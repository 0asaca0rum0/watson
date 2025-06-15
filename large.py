import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
import random
import logging
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda import amp
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
# suppress sklearn undefined metric warnings when no samples predicted for a class
def _ignore_sklearn_warnings():
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

_ignore_sklearn_warnings()

# ------------------------
# Configuration
# ------------------------
@dataclass
class Config:
    train_images_dir: str = './content/train'
    train_gt_csv: str = './content/train.csv'

    input_size: Tuple[int, int] = (256, 256)
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 30
    patience: int = 10
    num_workers: int = 4
    seed: int = 42

    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ['benign', 'malignant'])

    specialist_backbones: List[str] = field(default_factory=lambda: [
        'convnext_base_in22ft1k',
        'tf_efficientnetv2_l',
        'mobilenetv3_large_100'
    ])
    generalist_backbone: str = 'convnext_base_in22ft1k'
    lambda_bal: float = 0.02

CONFIG = Config()

# ------------------------
# Setup
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# Dataset & Transforms
# ------------------------
class ISICDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['benign_malignant'].isin(CONFIG.class_names)]
        self.images_dir = images_dir
        self.transform = transform
        self.label_col = 'benign_malignant'
        self.class_to_idx = {cls: i for i, cls in enumerate(CONFIG.class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, f"{row['image_name']}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = image.resize(CONFIG.input_size)
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        label = self.class_to_idx[row[self.label_col]]
        return image, label


def get_transforms(train: bool = True):
    h, w = CONFIG.input_size
    if train:
        return A.Compose([
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

# ------------------------
# Model Architecture
# ------------------------
class Expert(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=True, features_only=True)
        ch = self.net.feature_info.info[-1]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, CONFIG.num_classes)

    def forward(self, x):
        fmap = self.net(x)[-1]
        vec = self.pool(fmap).flatten(1)
        logits = self.classifier(vec)
        return vec, logits

class DynamicMoE(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate experts
        self.experts = nn.ModuleList(
            [Expert(b) for b in CONFIG.specialist_backbones] +
            [Expert(CONFIG.generalist_backbone)]
        )
        # Compute gating input dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, *CONFIG.input_size)
            dims = [e(dummy)[0].shape[1] for e in self.experts]
            total_ch = sum(dims)
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(total_ch, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, len(self.experts))
        )

    def forward(self, x, epoch: int = 0):
        feats, logits = zip(*(e(x) for e in self.experts))
        comb = torch.cat(feats, dim=1)
        tau = max(1.0 - 0.05 * epoch, 0.2)
        w = F.gumbel_softmax(self.gate(comb), tau=tau, hard=True, dim=1)
        # Load balancing loss
        spec_w = w[:, :-1]
        load_loss = CONFIG.lambda_bal * torch.var(spec_w.mean(dim=0))
        # Weighted sum of logits
        out = sum(w[:, i].unsqueeze(-1) * logits[i] for i in range(len(self.experts)))
        return out, load_loss

# ------------------------
# Training & Evaluation
# ------------------------
def train_epoch(model, loader, optimizer, scaler, epoch: int):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"Train Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with amp.autocast():
            out, load = model(images, epoch)
            loss = F.cross_entropy(out, labels) + load
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    oa = accuracy_score(all_labels, all_preds)
    ba = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, oa, ba


def validate(model, loader, epoch: int):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Val Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            out, load = model(images, epoch)
            loss = F.cross_entropy(out, labels) + load
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    oa = accuracy_score(all_labels, all_preds)
    ba = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CONFIG.class_names, zero_division=0)
    logging.info(f"Validation Report:\n{report}")
    return avg_loss, oa, ba

# ------------------------
# Main
# ------------------------
def main():
    # Dataset and loaders
    full_dataset = ISICDataset(CONFIG.train_gt_csv, CONFIG.train_images_dir, transform=get_transforms(True))
    # Stratified split into train/val
    labels = [full_dataset.df.iloc[i][full_dataset.label_col] for i in range(len(full_dataset))]
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        random_state=CONFIG.seed,
        stratify=labels
    )
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True)

    # Model, optimizer, scaler
    model = DynamicMoE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr)
    scaler = amp.GradScaler()

    # History tracking
    history = {'epoch': [], 'train_loss': [], 'train_oa': [], 'train_ba': [], 'val_loss': [], 'val_oa': [], 'val_ba': []}
    best_ba = 0
    epochs_no_improve = 0

    for epoch in range(1, CONFIG.epochs + 1):(1, CONFIG.epochs + 1):
        train_loss, train_oa, train_ba = train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_oa, val_ba = validate(model, val_loader, epoch)

        # Log
        logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, OA={train_oa:.4f}, BA={train_ba:.4f} | "
                     f"Val Loss={val_loss:.4f}, OA={val_oa:.4f}, BA={val_ba:.4f}")

        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_oa'].append(train_oa)
        history['train_ba'].append(train_ba)
        history['val_loss'].append(val_loss)
        history['val_oa'].append(val_oa)
        history['val_ba'].append(val_ba)

        # Checkpoint & early stopping
        if val_ba > best_ba:
            best_ba = val_ba
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            logging.info("-- New best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG.patience:
                logging.info("-- Early stopping triggered.")
                break

    # Save metrics to CSV
    pd.DataFrame(history).to_csv('metrics.csv', index=False)

    # Plot metrics
    def plot_metric(name, ylabel):
        plt.figure()
        plt.plot(history['epoch'], history[f'train_{name}'], label='Train')
        plt.plot(history['epoch'], history[f'val_{name}'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{name}_plot.png')
        plt.close()

    plot_metric('loss', 'Loss')
    plot_metric('oa', 'Overall Accuracy')
    plot_metric('ba', 'Balanced Accuracy')

if __name__ == '__main__':
    main()
