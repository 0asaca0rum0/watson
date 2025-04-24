import os
import random
import gc
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics.classification import MulticlassAUROC
from torchmetrics import JaccardIndex
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score)
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda import amp
from torch.utils.data import WeightedRandomSampler

# CONFIGURATION
CONFIG = {
    'batch_size': 8,
    'orig_size': (600, 450),
    'lr': 2e-4,
    'epochs': 50,
    'num_workers': 4,
    'num_experts': 4,
    'patience': 5,
    'seed': 42,
    'test_ratio': 0.2,
    'val_ratio': 0.2
}

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
# AUGMENTATIONS
train_transform = A.Compose([
    A.PadIfNeeded(height=CONFIG['orig_size'][1],
                  width=CONFIG['orig_size'][0], border_mode=0),
    A.Rotate(limit=30, border_mode=0, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=0, border_mode=0, p=0.5),
    ToTensorV2()
])
val_test_transform = A.Compose([
    A.PadIfNeeded(height=CONFIG['orig_size'][1],
                  width=CONFIG['orig_size'][0], border_mode=0),
    ToTensorV2()
])

# DATASET
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.transform = transform
        self.classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = Image.open(os.path.join(
                self.img_dir, row['image'] + '.jpg')).convert('RGB')
            mask = Image.open(os.path.join(
                self.mask_dir, row['image'] + '_segmentation.png')).convert('L')
            img, mask = np.array(img), np.array(mask)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            label = int((row[self.classes].values == 1).argmax())
            mask = (mask > 0).long()
            return img, mask.unsqueeze(0), label
        except Exception as e:
            logging.error(f"Error loading sample at index {idx}: {e}")
            return torch.zeros(3, *CONFIG['orig_size']), torch.zeros(1, *CONFIG['orig_size']), 0

# BALANCED SAMPLER
def make_balanced_sampler(dataset, indices):
    # extract labels for the given indices
    labels = np.array([
        int((dataset.df.iloc[i][dataset.classes].values == 1).argmax())
        for i in indices
    ])
    # compute class counts and weights
    class_counts = np.bincount(labels, minlength=len(dataset.classes))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)

# MODEL DEFINITIONS

class Expert(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=True, num_classes=0)
        self.head_cls = nn.Linear(self.backbone.num_features, nc)
        self.head_seg = nn.Sequential(
            nn.Conv2d(self.backbone.num_features, 64, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        cls = self.head_cls(feat)
        seg = self.head_seg(feat.unsqueeze(-1).unsqueeze(-1))
        return cls, seg


class MoE(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.experts = nn.ModuleList([Expert(nc)
                                     for _ in range(CONFIG['num_experts'])])
        self.gate = nn.Sequential(nn.Linear(nc * CONFIG['num_experts'], 128),
                                  nn.ReLU(), nn.Linear(128, CONFIG['num_experts']))

    def forward(self, x):
        outs = [e(x) for e in self.experts]
        cls_stack = torch.stack([o[0] for o in outs], dim=1)
        seg_stack = torch.stack([o[1] for o in outs], dim=1)
        feats = torch.cat([o[0] for o in outs], dim=1)
        w = F.gumbel_softmax(self.gate(feats), hard=True, dim=1)
        cls = (cls_stack * w.unsqueeze(-1)).sum(1)
        seg = (seg_stack * w.view(-1, CONFIG['num_experts'], 1, 1, 1)).sum(1)
        return cls, seg

# TRAIN / EVAL


def train_one_epoch(model, loader, opt, scaler):
    model.train()
    tot_loss = 0
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(
            device), masks.to(device), labels.to(device)
        with amp.autocast():
            cls, seg = model(imgs)
            loss = F.cross_entropy(
                cls, labels) + F.binary_cross_entropy_with_logits(seg, masks.float())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot_loss += loss.item()
    return tot_loss / len(loader)


def evaluate(model, loader, split_name="Validation"):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for imgs, _, labels in loader:
            imgs = imgs.to(device)
            cls, _ = model(imgs)
            pred = cls.argmax(1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(labels.numpy())

    print(f"\n{split_name} Set Evaluation:")
    print(classification_report(all_true, all_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_true, all_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(all_true, all_pred))
    print("Precision:", precision_score(all_true, all_pred, average='macro'))
    print("Recall:", recall_score(all_true, all_pred, average='macro'))
    print("F1 Score:", f1_score(all_true, all_pred, average='macro'))
    print("Overall Accuracy:", accuracy_score(all_true, all_pred))

# Stacked ensemble evaluation (classification only)
def ensemble_evaluate(model_cfgs, loader):
    device = get_device()
    models = []
    for cls, ckpt in model_cfgs:
        m = cls(nc=7).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, _, labels in loader:
            imgs = imgs.to(device)
            # sum softmax probabilities
            probs = sum(F.softmax(m(imgs)[0], dim=1) for m in models)
            preds = (probs / len(models)).argmax(1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(preds)
    print("\nStacked Ensemble Results:")
    print(classification_report(all_true, all_pred, digits=4))
    print("Balanced Acc:", balanced_accuracy_score(all_true, all_pred))

def main():
    try:
        random.seed(CONFIG['seed'])
        np.random.seed(CONFIG['seed'])
        torch.manual_seed(CONFIG['seed'])

        ds = HAM10000Dataset("GroundTruth.csv", "images", "masks", None)
        labels = np.array(ds.df[ds.classes].values.argmax(1))

        sss = StratifiedShuffleSplit(
            1, test_size=CONFIG['test_ratio'], random_state=CONFIG['seed'])
        trv, te = next(sss.split(np.arange(len(ds)), labels))

        sss = StratifiedShuffleSplit(
            1, test_size=CONFIG['val_ratio'] / (1 - CONFIG['test_ratio']), random_state=CONFIG['seed'])
        tr, va = next(sss.split(trv, labels[trv]))

        train_set = Subset(HAM10000Dataset(
            "GroundTruth.csv", "images", "masks", train_transform), trv[tr])
        val_set = Subset(HAM10000Dataset("GroundTruth.csv",
                         "images", "masks", val_test_transform), trv[va])
        test_set = Subset(HAM10000Dataset("GroundTruth.csv",
                          "images", "masks", val_test_transform), te)

        train_loader = DataLoader(
            train_set, CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(
            val_set, CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        test_loader = DataLoader(
            test_set, CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

        model = MoE(nc=7).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        scaler = amp.GradScaler(enabled=True)
        writer = SummaryWriter()

        for e in range(CONFIG['epochs']):
            loss = train_one_epoch(model, train_loader, opt, scaler)
            print(f"Epoch {e + 1}/{CONFIG['epochs']} Loss: {loss:.4f}")
            evaluate(model, val_loader, split_name="Validation")

        print("Final Test Results:")
        evaluate(model, test_loader, split_name="Test")

        # deploy ensemble of three experts
        ensemble_evaluate([
            (Expert, "checkpoints/effb4_best.pth"),
            (Expert, "checkpoints/effb5_best.pth"),
            (Expert, "checkpoints/swin_small_best.pth"),
        ], test_loader)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")


if __name__ == '__main__':
    main()
