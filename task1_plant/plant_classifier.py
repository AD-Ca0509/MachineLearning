# -*- coding: utf-8 -*-
"""
植物物种分类 —— 5 分类图像分类（v5：终极冲分版）

本版升级：
  1. 混合集成：3 种不同架构（ConvNeXt-Tiny + EfficientNet-B2 + ResNet50）
     各做 5 折交叉验证，共 15 个模型一起投票
  2. 更高分辨率 288
  3. MixUp + CutMix 数据增强（小数据集涨分最有效的手段之一）
  4. 保留：两阶段训练 + 余弦衰减 + 标签平滑 + 测试时增强(TTA)

【数据位置】D:/plant/2026task1
【运行】在 VSCode 终端执行：  python D:/plant/plant_classifier.py
【注意】本版要训练 15 个模型，约 40~60 分钟，请耐心等待。
"""

import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# =====================================================================
# 配置区
# =====================================================================

BASE = r"D:\plant\2026task1"
TRAIN_DIR = os.path.join(BASE, "dataset-for-task2", "dataset-for-task2", "train")
TEST_DIR = os.path.join(BASE, "dataset-for-task2", "dataset-for-task2", "test")
SUB_PATH = os.path.join(BASE, "submission-for-task2.csv")
OUTPUT_DIR = r"D:\plant"

# 参与集成的 3 种架构（不同架构互相纠错，比同一种投多次更强）
MODEL_NAMES = ["convnext_tiny", "efficientnet_b2", "resnet50"]

IMG_SIZE = 288
BATCH_SIZE = 24

N_FOLDS = 5
HEAD_EPOCHS = 4
FINE_EPOCHS = 14
HEAD_LR = 1e-3
FINE_LR = 1e-4
LABEL_SMOOTHING = 0.1
USE_TTA = True

SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# =====================================================================
# 基础工具
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class PlantDataset(Dataset):
    def __init__(self, items, transform, has_label=True):
        self.items = items
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.has_label:
            path, label = self.items[idx]
        else:
            path = self.items[idx][0]
            label = -1
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        if self.has_label:
            return image, label
        return image


# =====================================================================
# 模型
# =====================================================================

def build_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")
    return model


def head_parameters(model_name, model):
    if model_name == "resnet50":
        return model.fc.parameters()
    return model.classifier.parameters()


def set_head_only_trainable(model_name, model):
    for p in model.parameters():
        p.requires_grad = False
    for p in head_parameters(model_name, model):
        p.requires_grad = True


def set_all_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


# =====================================================================
# 训练 / 验证 / 预测
# =====================================================================

def mixup(x, y, alpha=0.2):
    """MixUp：把两张图按比例线性混合，标签也按同样比例混合"""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix(x, y, alpha=1.0):
    """CutMix：把一张图的一块矩形区域"贴"到另一张图上，标签按面积比例混合"""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))
    x1 = int(np.clip(cx - rw // 2, 0, W))
    x2 = int(np.clip(cx + rw // 2, 0, W))
    y1 = int(np.clip(cy - rh // 2, 0, H))
    y2 = int(np.clip(cy + rh // 2, 0, H))
    x[:, :, y1:y2, x1:x2] = x[index][:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
    return x, y, y[index], lam


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 每个 batch 随机选 MixUp 或 CutMix（各 50% 概率），增强泛化、防过拟合
        if random.random() < 0.5:
            mixed_x, ya, yb, lam = mixup(images, labels)
        else:
            mixed_x, ya, yb, lam = cutmix(images, labels)

        outputs = model(mixed_x)
        loss = lam * criterion(outputs, ya) + (1 - lam) * criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()  # 粗略估计（混合后标签已不精确）
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_probs(model, loader, device, use_tta=True):
    model.eval()
    probs = []
    for images in tqdm(loader, desc="predict", leave=False):
        images = images.to(device)
        p = torch.softmax(model(images), dim=1)
        if use_tta:
            p += torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            p /= 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


# =====================================================================
# 主流程
# =====================================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 类别
    classes = sorted([
        c for c in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, c))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"共 {num_classes} 类: {classes}")

    # 2. 训练集
    train_items = []
    for cls in classes:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        for f in sorted(os.listdir(cls_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                train_items.append((os.path.join(cls_dir, f), class_to_idx[cls]))
    labels = np.array([it[1] for it in train_items])
    print(f"训练图片总数: {len(train_items)}")

    # 3. 提交模板 + 测试集
    sub = pd.read_csv(SUB_PATH)
    id_col = sub.columns[0]
    label_col = sub.columns[1]

    test_paths = []
    missing_ids = []
    for img_id in sub[id_col]:
        p = os.path.join(TEST_DIR, str(img_id))
        if not os.path.exists(p):
            base = os.path.splitext(str(img_id))[0]
            found = None
            for ext in IMG_EXTS:
                cand = os.path.join(TEST_DIR, base + ext)
                if os.path.exists(cand):
                    found = cand
                    break
            p = found
        if p is None:
            missing_ids.append(img_id)
        test_paths.append(p)
    print(f"匹配到测试图片 {len(test_paths) - len(missing_ids)} 张，缺失 {len(missing_ids)} 张")

    valid_items = [(p,) for p in test_paths if p is not None]
    test_ds = PlantDataset(valid_items, eval_transform, has_label=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. 交叉验证（每个架构都做 5 折），最后所有模型一起投票
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_models = len(MODEL_NAMES) * N_FOLDS
    test_probs = np.zeros((len(valid_items), num_classes))
    all_accs = []

    for model_name in MODEL_NAMES:
        print(f"\n########## 模型: {model_name} ##########")
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"--- Fold {fold + 1}/{N_FOLDS} ---")

            train_subset = [train_items[i] for i in tr_idx]
            val_subset = [train_items[i] for i in va_idx]
            train_ds = PlantDataset(train_subset, train_transform, has_label=True)
            val_ds = PlantDataset(val_subset, eval_transform, has_label=True)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = build_model(model_name, num_classes).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            best_acc = 0.0

            # 第一阶段：训练分类头
            set_head_only_trainable(model_name, model)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)
            for epoch in range(1, HEAD_EPOCHS + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f"  [头]   {epoch}/{HEAD_EPOCHS}  val_acc={val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

            # 第二阶段：全网络微调
            set_all_trainable(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=FINE_LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_EPOCHS)
            for epoch in range(1, FINE_EPOCHS + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step()
                print(f"  [微调] {epoch}/{FINE_EPOCHS}  val_acc={val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

            all_accs.append(best_acc)
            print(f"  Fold {fold + 1} 最佳: {best_acc:.4f}")

            # 预测测试集，累积概率
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device))
            test_probs += predict_probs(model, test_loader, device, use_tta=USE_TTA)

    # 5. 平均所有模型（15 个）的概率
    test_probs /= n_models
    pred_indices = test_probs.argmax(1)

    print(f"\n全部 {n_models} 个模型的平均验证准确率: {np.mean(all_accs):.4f}")

    # 6. 写回提交文件
    valid_pred_iter = iter(pred_indices)
    final_labels = []
    for p in test_paths:
        if p is not None:
            final_labels.append(idx_to_class[int(next(valid_pred_iter))])
        else:
            final_labels.append(classes[0])

    sub[label_col] = final_labels
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"提交文件已生成: {out_path}")
    print(sub.head())


if __name__ == "__main__":
    main()
