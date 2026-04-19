# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter


# ======================================
# Age → Age Group 변환 함수
# ======================================
def age_to_group(age):
    if age < 10: return 0
    elif age < 20: return 1
    elif age < 30: return 2
    elif age < 40: return 3
    elif age < 50: return 4
    elif age < 60: return 5
    else: return 6


# ======================================
# 허용 오차 정확도 계산 함수
# ======================================
def tolerance_accuracy(pred_groups, true_groups, tolerance=1):
    """
    tolerance 범위 내 예측을 정답으로 인정
    tolerance=1: ±1 연령대 허용
    tolerance=2: ±2 연령대 허용
    """
    correct = sum(abs(p - t) <= tolerance for p, t in zip(pred_groups, true_groups))
    return correct / len(true_groups)


# ======================================
# UTKFace Dataset
# ======================================
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob.glob(os.path.join(root_dir, "**/*.jpg.chip.jpg"), recursive=True)
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        filename = os.path.basename(img_path)
        age = int(filename.split("_")[0])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)


# ======================================
# WeightedRandomSampler 생성
# ======================================
def create_weighted_sampler(dataset):
    print("Calculating class weights...")
    ages = []
    for i in range(len(dataset)):
        _, age = dataset[i]
        ages.append(age_to_group(int(age.item())))
    
    counts = Counter(ages)
    print("Class distribution:", dict(counts))
    
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[age_group] for age_group in ages]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ======================================
# ResNet50
# ======================================
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1)
    )
    return model


# ======================================
# 학습 함수 (±1, ±2 정확도 추가)
# ======================================
def train_model(model, train_loader, val_loader, device, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_rmses, val_rmses = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    train_accs_1, val_accs_1 = [], []  # ±1 정확도
    train_accs_2, val_accs_2 = [], []  # ±2 정확도

    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for images, ages in train_loader:
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(ages.cpu().numpy())

        # Train Metrics
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_mae_val = np.mean(np.abs(train_preds - train_targets))
        train_rmse_val = np.sqrt(np.mean((train_preds - train_targets)**2))
        train_maes.append(train_mae_val)
        train_rmses.append(train_rmse_val)
        
        train_pred_groups = [age_to_group(x) for x in train_preds]
        train_true_groups = [age_to_group(x) for x in train_targets]
        train_f1 = f1_score(train_true_groups, train_pred_groups, average='macro', zero_division=0)
        train_acc = accuracy_score(train_true_groups, train_pred_groups)
        train_acc_1 = tolerance_accuracy(train_pred_groups, train_true_groups, tolerance=1)
        train_acc_2 = tolerance_accuracy(train_pred_groups, train_true_groups, tolerance=2)
        
        train_f1s.append(train_f1)
        train_accs.append(train_acc)
        train_accs_1.append(train_acc_1)
        train_accs_2.append(train_acc_2)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, ages in val_loader:
                images = images.to(device, non_blocking=True)
                ages = ages.to(device, non_blocking=True)

                outputs = model(images).squeeze()
                loss = criterion(outputs, ages)
                val_running_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(ages.cpu().numpy())

        # Validation Metrics
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_mae_val = np.mean(np.abs(val_preds - val_targets))
        val_rmse_val = np.sqrt(np.mean((val_preds - val_targets)**2))
        val_maes.append(val_mae_val)
        val_rmses.append(val_rmse_val)
        
        val_pred_groups = [age_to_group(x) for x in val_preds]
        val_true_groups = [age_to_group(x) for x in val_targets]
        val_f1 = f1_score(val_true_groups, val_pred_groups, average='macro', zero_division=0)
        val_acc = accuracy_score(val_true_groups, val_pred_groups)
        val_acc_1 = tolerance_accuracy(val_pred_groups, val_true_groups, tolerance=1)
        val_acc_2 = tolerance_accuracy(val_pred_groups, val_true_groups, tolerance=2)
        
        val_f1s.append(val_f1)
        val_accs.append(val_acc)
        val_accs_1.append(val_acc_1)
        val_accs_2.append(val_acc_2)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = epoch_time * (num_epochs - epoch - 1)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae_val:.4f}, RMSE: {train_rmse_val:.4f}")
        print(f"        Acc: {train_acc:.4f}, Acc(±1): {train_acc_1:.4f}, Acc(±2): {train_acc_2:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, MAE: {val_mae_val:.4f}, RMSE: {val_rmse_val:.4f}")
        print(f"        Acc: {val_acc:.4f}, Acc(±1): {val_acc_1:.4f}, Acc(±2): {val_acc_2:.4f}, F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s | Elapsed: {elapsed:.2f}s | Remaining: {remaining:.2f}s")
        print(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break

    # 최종 평가
    print("\n" + "="*70)
    print("=== Final Metrics ===")
    print(f"Best Val Loss: {min(val_losses):.4f} at Epoch {val_losses.index(min(val_losses))+1}")
    print(f"Best Val MAE: {min(val_maes):.4f} at Epoch {val_maes.index(min(val_maes))+1}")
    print(f"Best Val RMSE: {min(val_rmses):.4f} at Epoch {val_rmses.index(min(val_rmses))+1}")
    print(f"Best Val F1: {max(val_f1s):.4f} at Epoch {val_f1s.index(max(val_f1s))+1}")
    print(f"\n--- Accuracy Metrics ---")
    print(f"Best Val Accuracy (Exact):     {max(val_accs):.4f} ({max(val_accs)*100:.2f}%)")
    print(f"Best Val Accuracy (±1 group):  {max(val_accs_1):.4f} ({max(val_accs_1)*100:.2f}%)")
    print(f"Best Val Accuracy (±2 groups): {max(val_accs_2):.4f} ({max(val_accs_2)*100:.2f}%)")
    print("="*70)

    print("\n=== Classification Report (Last Epoch) ===")
    age_group_names = ['0-9','10-19','20-29','30-39','40-49','50-59','60+']
    print(classification_report(val_true_groups, val_pred_groups, target_names=age_group_names))

    # 그래프 출력
    plot_training_results(train_losses, val_losses, train_maes, val_maes, 
                         train_rmses, val_rmses, train_f1s, val_f1s, 
                         train_accs, val_accs, train_accs_1, val_accs_1,
                         train_accs_2, val_accs_2)

    return model


# ======================================
# 학습 결과 그래프 출력 (나이 ±1, ±2 정확도 추가)
# ======================================
def plot_training_results(train_losses, val_losses, train_maes, val_maes,
                         train_rmses, val_rmses, train_f1s, val_f1s,
                         train_accs, val_accs, train_accs_1, val_accs_1,
                         train_accs_2, val_accs_2):
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Loss
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    axes[0, 0].plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, train_maes, 'b-o', label='Train MAE', markersize=4)
    axes[0, 1].plot(epochs, val_maes, 'r-o', label='Val MAE', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (years)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 2].plot(epochs, train_rmses, 'b-o', label='Train RMSE', markersize=4)
    axes[0, 2].plot(epochs, val_rmses, 'r-o', label='Val RMSE', markersize=4)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('RMSE (years)')
    axes[0, 2].set_title('Root Mean Squared Error')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, train_f1s, 'b-o', label='Train F1', markersize=4)
    axes[1, 0].plot(epochs, val_f1s, 'r-o', label='Val F1', markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score (Age Group Classification)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy (Exact)
    axes[1, 1].plot(epochs, train_accs, 'b-o', label='Train Acc', markersize=4)
    axes[1, 1].plot(epochs, val_accs, 'r-o', label='Val Acc', markersize=4)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy (Exact Match)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy (±1)
    axes[1, 2].plot(epochs, train_accs_1, 'b-o', label='Train Acc(±1)', markersize=4)
    axes[1, 2].plot(epochs, val_accs_1, 'r-o', label='Val Acc(±1)', markersize=4)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy (±1 Age Group)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Accuracy (±2)
    axes[2, 0].plot(epochs, train_accs_2, 'b-o', label='Train Acc(±2)', markersize=4)
    axes[2, 0].plot(epochs, val_accs_2, 'r-o', label='Val Acc(±2)', markersize=4)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('Accuracy (±2 Age Groups)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Gap (Overfitting 지표)
    train_val_gap = np.array(train_losses) - np.array(val_losses)
    axes[2, 1].plot(epochs, train_val_gap, 'g-o', label='Train-Val Gap', markersize=4)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Loss Gap')
    axes[2, 1].set_title('Overfitting Monitor (Train - Val Loss)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    final_vals = [val_accs[-1], val_accs_1[-1], val_accs_2[-1]]
    labels = ['Exact', '±1 group', '±2 groups']
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    bars = axes[2, 2].bar(labels, final_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[2, 2].set_ylabel('Accuracy')
    axes[2, 2].set_title('Final Val Accuracy Comparison')
    axes[2, 2].set_ylim([0, 1])
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, val in zip(bars, final_vals):
        height = bar.get_height()
        axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}\n({val*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_results_with_tolerance.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graph saved as 'training_results_with_tolerance.png'")
    plt.show()


# ======================================
# main 보호 (Windows 필수)(mac은 잘 모르겠음)
# ======================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\ohjun\Downloads\UTKFace-20251130T092716Z-1-001\UTKFace"
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    NUM_WORKERS = 4

    # 데이터 증강 (Train용)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Validation용 (증강 없음)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 데이터셋 로드
    print("Loading dataset...")
    full_dataset = UTKFaceDataset(DATA_PATH, transform=None)
    
    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    # 각각 다른 transform 적용
    train_dataset = torch.utils.data.Subset(
        UTKFaceDataset(DATA_PATH, transform=transform_train),
        train_indices.indices
    )
    val_dataset = torch.utils.data.Subset(
        UTKFaceDataset(DATA_PATH, transform=transform_val),
        val_indices.indices
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # WeightedRandomSampler 생성
    full_train_dataset = UTKFaceDataset(DATA_PATH, transform=None)
    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    sampler = create_weighted_sampler(train_subset)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}\n")

    model = build_model()
    model = model.to(device)
    
    # 학습 시작
    train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS)
