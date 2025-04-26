import os
import random
import logging
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda import amp
from sklearn.metrics import classification_report, balanced_accuracy_score

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ------------------------
# Configuration
# ------------------------


@dataclass
class Config:
    train_dir: str = 'train_dir'
    val_dir: str = 'val_dir'
    batch_size: int = 16
    orig_size: Tuple[int, int] = (600, 450)
    lr: float = 1e-4
    epochs: int = 40
    num_workers: int = 4
    patience: int = 5
    seed: int = 42
    num_classes: int = 7
    specialist_backbones: List[str] = field(default_factory=lambda: [
        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5'
    ])
    generalist_backbone: str = 'efficientnet_b2'
    top_k: int = 2
    lambda_bal: float = 0.01
    generalist_bias: float = 0.5


CONFIG = Config()

# Setup logging and seeds
o
os.makedirs('checkpoints', exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
random.seed(CONFIG.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# Dataset
# ------------------------


class ClassificationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = root_dir
        self.transform = transform
        # discover classes
        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            class_dir = os.path.join(self.root, c)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(
                        (os.path.join(class_dir, fname), self.class_to_idx[c]))
        self.labels = [label for _, label in self.samples]
        logging.info(f"Loaded {len(self.samples)} samples from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        if self.transform:
            img_t = self.transform(image=img_np)['image']
        else:
            img_t = ToTensorV2()(image=img_np)['image']
        return img_t, label


# ------------------------
# Transforms
# ------------------------
train_transform = A.Compose([
    A.Resize(CONFIG.orig_size[1], CONFIG.orig_size[0]),
    A.RandomResizedCrop(
        CONFIG.orig_size[1], CONFIG.orig_size[0], scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CONFIG.orig_size[1], CONFIG.orig_size[0]),
    A.Normalize(),
    ToTensorV2()
])

# ------------------------
# Expert with Self-Attn
# ------------------------


class BaseExpert(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        # CNN feature extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True
        )
        feat_info = self.backbone.feature_info.info[-1]
        self.channels = feat_info['num_chs']
        # small Transformer encoder for self-attention
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=8,
            dim_feedforward=4*self.channels,
            dropout=0.1,
            batch_first=True
        )
        # final classifier
        self.classifier = nn.Linear(self.channels, CONFIG.num_classes)

    def forward(self, x):
        # extract spatial features
        feat_map = self.backbone(x)[-1]          # (B, C, H, W)
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        attn_out = self.trans_enc(tokens)           # (B, H*W, C)
        pooled = attn_out.mean(dim=1)               # (B, C)
        logits = self.classifier(pooled)            # (B, num_classes)
        return pooled, logits

# ------------------------
# Mixture-of-Experts with Generalist
# ------------------------


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        # create specialist experts
        self.spec_experts = nn.ModuleList([
            BaseExpert(bk) for bk in CONFIG.specialist_backbones
        ])
        # create a generalist expert
        self.generalist = BaseExpert(CONFIG.generalist_backbone)
        # combine into single ModuleList
        self.experts = nn.ModuleList(
            list(self.spec_experts) + [self.generalist])
        self.num_specs = len(self.spec_experts)
        self.num_experts = len(self.experts)
        # gating network takes concatenated pooled features
        total_dim = self.num_experts * self.spec_experts[0].channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim//2, self.num_experts)
        )

    def forward(self, x, epoch=None):
        # gather features and logits from all experts
        feats, logits = [], []
        for expert in self.experts:
            f, l = expert(x)
            feats.append(f)
            logits.append(l)
        # concatenate for gating
        feat_tensor = torch.cat(feats, dim=1)    # (B, E*C)
        scores = self.gate_mlp(feat_tensor)       # (B, E)
        # bias generalist to ensure minimum floor
        scores[:, -1] += CONFIG.generalist_bias
        # top-k selection
        topk_vals, topk_idx = scores.topk(CONFIG.top_k, dim=1)
        mask = torch.zeros_like(scores).scatter(1, topk_idx, 1)
        # annealed Gumbel softmax
        temp = max(1.0 - 0.045*(epoch or 0), 0.1)
        gate_weights = torch.zeros_like(scores)
        soft = F.softmax(topk_vals / temp, dim=1)
        gate_weights.scatter_(1, topk_idx, soft)
        # load-balance penalty on specialists only
        avg_gate_spec = gate_weights[:, :self.num_specs].mean(dim=0)
        load_loss = CONFIG.lambda_bal * \
            ((avg_gate_spec - 1.0/self.num_specs)**2).sum()
        # combine logits
        logits_stack = torch.stack(logits, dim=1)   # (B, E, C)
        gw = gate_weights.unsqueeze(-1)            # (B, E, 1)
        cls_out = (gw * logits_stack).sum(dim=1)   # (B, C)
        return cls_out, load_loss

# ------------------------
# Sampler
# ------------------------


def make_balanced_sampler(dataset: ClassificationDataset) -> WeightedRandomSampler:
    counts = np.bincount(dataset.labels, minlength=CONFIG.num_classes)
    weights = 1.0 / counts
    sample_w = [weights[l] for l in dataset.labels]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

# ------------------------
# Training & Evaluation
# ------------------------


def train_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc=f"Train E{epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        with amp.autocast():
            cls_logits, load_loss = model(imgs, epoch)
            cls_loss = F.cross_entropy(cls_logits, labels)
            loss = cls_loss + load_loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    logging.info(f"Epoch {epoch} Train Loss: {avg:.4f}")
    return avg


def evaluate(model, loader, split="Val"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=split):
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    logging.info(f"{split} Report:\n" +
                 classification_report(all_labels, all_preds))
    bal = balanced_accuracy_score(all_labels, all_preds)
    logging.info(f"{split} Balanced Acc: {bal:.4f}")
    return bal


# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    # Datasets & loaders
    train_ds = ClassificationDataset(
        CONFIG.train_dir, transform=train_transform)
    val_ds = ClassificationDataset(CONFIG.val_dir,   transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG.batch_size,
        sampler=make_balanced_sampler(train_ds),
        num_workers=CONFIG.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG.batch_size,
        shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True
    )

    # Model, optimizer, scheduler, scaler
    model = MoE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=CONFIG.patience//2, factor=0.5
    )
    scaler = amp.GradScaler()

    best_score, patience = 0.0, 0
    for ep in range(1, CONFIG.epochs+1):
        train_epoch(model, train_loader, optimizer, scaler, ep)
        val_acc = evaluate(model, val_loader, split="Val")
        scheduler.step(val_acc)
        if val_acc > best_score:
            best_score, patience = val_acc, 0
            torch.save(model.state_dict(), os.path.join(
                'checkpoints', 'best.pth'))
            logging.info("Saved best model")
        else:
            patience += 1
            if patience >= CONFIG.patience:
                logging.info("Early stopping")
                break

    # Final evaluation on validation
    logging.info("Final Evaluation")
    evaluate(model, val_loader, split="Test")
