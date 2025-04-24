import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import os
import gc                            # added
import torchmetrics                  # added for AUROC/IoU
import logging
import warnings
# new for reproducibility
import random
# new for stratified splits
from sklearn.model_selection import StratifiedShuffleSplit
import torch.cuda.amp as amp
# NEW: Import for Gumbel softmax and DropPath
from torch.nn.modules.dropout import Dropout
from torch import nn

# Configure data dirs
INPUT_DIR = "./ham1000-classification-segmentation"
OUTPUT_DIR = "./working"
os.makedirs(os.path.join(OUTPUT_DIR, "model_checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "splits"), exist_ok=True)

# ensure required packages (for notebook/Kaggle cell)
try:
    import torchmetrics
    import timm
except ImportError:
    os.system("pip install -q torchmetrics timm")

# =====================
# Device setup (GPU/CPU only)
# =====================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# silence torch_xla TPU port errors globally
logging.getLogger('torch_xla').setLevel(logging.ERROR)

# Enable cuDNN autotuner for improved performance
torch.backends.cudnn.benchmark = True

# Clean RAM/VRAM at start
gc.collect()                        # free Python objects
if torch.cuda.is_available():
    torch.cuda.empty_cache()        # free GPU memory

# reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# enable TF32 for faster matmuls on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ensure INPUT_DIR exists
assert os.path.isdir(INPUT_DIR), f"INPUT_DIR not found: {INPUT_DIR}"

# consolidate device selection
device = get_device()
print(f"Using device: {device}")

# =====================
# Unified Configuration
# =====================
CONFIG = {
    "batch_size": 128,
    "image_size": 160,
    "base_lr": 2e-3,
    "warmup_epochs": 2,
    "max_epochs": 30,
    "accum_steps": 2,
    "mixed_precision": True,
    "grad_clip": 1.0,
    "num_workers": 6,
    "cache_images": False,
    "train_task": "both",
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "patience": 6,
    "seed": SEED,
    "compile_model": True,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,
    "ortho_weight": 0.01,
    "stochastic_depth": 0.2,
    "aug_level": "strong_v2",
    'num_experts': 4
}
# add number of experts
CONFIG['num_experts'] = 4

# =====================
# Augmentation Pipeline
# =====================
from torchvision.transforms import functional as F

# Resize with aspect ratio preserved and padding
def resize_with_padding(image, size):
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    pad_w, pad_h = size - new_w, size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return F.pad(image, padding, fill=0)

# Augmentation for training set
train_augmentations = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, CONFIG['image_size'])),  # Resize with padding
    transforms.RandomRotation(30),  # Random rotation ±30°
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.RandomVerticalFlip(),  # Vertical flip
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Transforms for validation/test sets (resize + normalize only)
val_test_augmentations = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, CONFIG['image_size'])),  # Resize with padding
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# build per-expert transform lists
train_transform_list = [train_augmentations for _ in range(CONFIG['num_experts'])]
val_transform_list   = [val_test_augmentations for _ in range(CONFIG['num_experts'])]

# =====================
# Hybrid Sampling Strategy
# =====================
from torch.utils.data import WeightedRandomSampler, Subset

def create_balanced_sampler(dataset):
    # support Dataset or Subset
    if isinstance(dataset, Subset):
        base, idxs = dataset.dataset, dataset.indices
        labels = np.array(base.labels)[idxs]
    else:
        labels = np.array(dataset.labels)
    counts = np.bincount(labels)
    total = len(labels)
    weights = [ total/(len(counts)*c) for c in counts ]
    sample_w = [ weights[int(l)] for l in labels ]
    return WeightedRandomSampler(sample_w, num_samples=total, replacement=True)

# =====================
# Dataset (multi‐view)
# =====================
class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, transforms_list):
        self.root_dir = root_dir
        self.transforms_list = transforms_list
        # ...existing code...
        self.resize = transforms.Resize(
            (CONFIG['image_size'], CONFIG['image_size']))  # moved here
        csv_path = os.path.join(root_dir, "GroundTruth.csv")
        df = pd.read_csv(csv_path)
        classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        for _, row in df.iterrows():
            img = row['image']
            self.image_paths.append(os.path.join(
                root_dir, 'images', img + '.jpg'))
            seg = img + '_segmentation'
            self.mask_paths.append(os.path.join(
                root_dir, 'masks', seg + '.png'))
            self.labels.append(
                next(i for i, c in enumerate(classes) if row[c] == 1))
        self.cache = {} if CONFIG['cache_images'] else None  # Use dict instead of list for selective caching

    def __len__(self): return len(self.image_paths)

    def _load_item(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        return self.resize(img), self.resize(mask)

    def __getitem__(self, idx):
        img, mask = self._load_item(idx)
        # generate multi‐view inputs
        if self.transforms_list:
            views = [tf(img.copy()) for tf in self.transforms_list]
        else:
            views = [transforms.ToTensor()(img)]
        mask_t = transforms.functional.pil_to_tensor(mask)
        mask_t = (mask_t > 0).long().squeeze(0)
        return views, mask_t, self.labels[idx]

# =====================
# DataLoader helper
# =====================
def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type=='cuda')
    )

def train_model():
    # use global transform lists to allow in-place updates (e.g., progressive resizing)
    global train_transform_list, val_transform_list
    device = get_device()
    # full dataset for splits
    ds = HAM10000Dataset(INPUT_DIR, None)
    labels = np.array(ds.labels)
    idxs = np.arange(len(ds))
    # stratified split
    sss = StratifiedShuffleSplit(1, test_size=CONFIG['test_ratio'], random_state=CONFIG['seed'])
    trval_idx, test_idx = next(sss.split(idxs, labels))
    sss = StratifiedShuffleSplit(1, test_size=CONFIG['val_ratio']/(1-CONFIG['test_ratio']), random_state=CONFIG['seed'])
    train_idx, val_idx = next(sss.split(trval_idx, labels[trval_idx]))

    # Save indices
    np.save(os.path.join(OUTPUT_DIR, 'splits/train_idx.npy'), train_idx)
    np.save(os.path.join(OUTPUT_DIR, 'splits/val_idx.npy'), val_idx)
    np.save(os.path.join(OUTPUT_DIR, 'splits/test_idx.npy'), test_idx)

    # subsets with proper transforms
    train_ds = Subset(HAM10000Dataset(INPUT_DIR, train_transform_list), train_idx)
    val_ds   = Subset(HAM10000Dataset(INPUT_DIR, val_transform_list),   val_idx)
    test_ds  = Subset(HAM10000Dataset(INPUT_DIR, val_transform_list),   test_idx)

    # balanced sampler + loaders
    sampler     = create_balanced_sampler(train_ds)
    train_loader= make_loader(train_ds, sampler=sampler)
    val_loader  = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    # instantiate model on GPU
    model = MoE(num_classes=7).to(device).to(memory_format=torch.channels_last)

    # safe checkpoint load
    ckpt_path = os.path.join(OUTPUT_DIR, 'model_checkpoints', 'best_model.pth')
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("No checkpoint found, training from scratch.")

    # early stopping setup
    best_score = -np.inf
    no_improve = 0

    try:
        train_len = int(0.8*len(ds))
        train_ds, val_ds = random_split(ds, [train_len, len(ds)-train_len])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True,
                                  collate_fn=collate_moe)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                num_workers=CONFIG['num_workers'], pin_memory=True,
                                collate_fn=collate_moe)

        model = MoE(num_classes=7).to(device).to(
            memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CONFIG['base_lr'], weight_decay=0.05)
        scaler = amp.GradScaler(enabled=CONFIG['mixed_precision'])
        total_steps = CONFIG['max_epochs'] * \
            (len(train_loader)//CONFIG['accum_steps'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG['base_lr'],
            total_steps=total_steps,
            pct_start=CONFIG['warmup_epochs']/CONFIG['max_epochs']
        )

        writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, 'logs'))

        history = {'train_loss': [], 'val_acc': []}

        # set up richer metrics
        auroc = torchmetrics.classification.MulticlassAUROC(
            num_classes=7).to(device)
        iou = torchmetrics.JaccardIndex(task="binary").to(device)

        # Apply torch.compile if available (PyTorch 2.0+)
        if CONFIG['compile_model'] and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print("Model successfully compiled with torch.compile")
            except Exception as e:
                print(f"Failed to compile model: {e}")

        for epoch in range(CONFIG['max_epochs']):
            if epoch == 10:
                print("Progressive resizing: updating image_size to 192")
                CONFIG['image_size'] = 192
                # Rebuild strong_aug for new resolution (other transforms remain the same)
                strong_aug = transforms.Compose([
                    transforms.RandAugment(num_ops=2, magnitude=15),
                    transforms.RandomResizedCrop(CONFIG['image_size'], scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomErasing(p=0.5)
                ])
                train_transform_list = [strong_aug for _ in range(CONFIG['num_experts'])]
                val_transform_list   = [strong_aug for _ in range(CONFIG['num_experts'])]
                # Optionally, reinitialize datasets/loaders here
            model.train()
            running_loss = 0.0
            for i, (views, masks, labels) in enumerate(tqdm(train_loader)):
                try:
                    views = [v.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last) for v in views]
                    masks = masks.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with amp.autocast(enabled=CONFIG['mixed_precision'] and not TPU_AVAILABLE):
                        cls_logits, seg_logits = model(views)
                        loss_cls = criterion(cls_logits, labels)
                        if CONFIG['train_task'] == 'both':
                            tgt = masks.unsqueeze(1).float()  # [B,1,H,W]
                            loss_seg = F.binary_cross_entropy_with_logits(
                                seg_logits, tgt)
                        else:
                            loss_seg = 0.0
                        loss = (loss_cls + loss_seg) / CONFIG['accum_steps']
                    # Scale the loss and backpropagate
                    scaler.scale(loss).backward()
                    if (i+1) % CONFIG['accum_steps'] == 0:
                        if TPU_AVAILABLE:
                            xm.optimizer_step(optimizer, barrier=True)
                            xm.mark_step()
                        else:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                        # Log gradient histograms for sanity checks
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f"grad/{name}", param.grad.cpu(), epoch)
                    running_loss += loss.item()
                    if i % 10 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("OOM encountered; clearing cache.")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
            avg_loss = running_loss/len(train_loader)
            history['train_loss'].append(avg_loss)
            writer.add_scalar('Loss/train', avg_loss, epoch)

            model.eval()
            correct = 0
            total = 0
            auroc.reset()
            iou.reset()
            with torch.no_grad():
                for views, masks, labels in val_loader:
                    views = [v.to(device).to(memory_format=torch.channels_last)
                             for v in views]
                    labels = labels.to(device)
                    masks = masks.to(device)
                    with amp.autocast(enabled=CONFIG['mixed_precision']):
                        logits, seg_logits = model(views)
                    preds = logits.argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    auroc.update(logits.softmax(dim=1), labels)
                    iou.update(
                        (seg_logits.sigmoid() > 0.5).long().squeeze(1), masks)

            val_acc = correct/total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(writer.log_dir, 'best_model.pth')
                )
            val_auroc = auroc.compute()
            val_iou = iou.compute()
            history['val_acc'].append(val_acc)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('AUROC/val', val_auroc, epoch)
            writer.add_scalar('IoU/val', val_iou, epoch)

            # Log expert diversity score: mean pairwise cosine similarity of expert fc weights
            expert_weights = torch.stack([F.normalize(ex.fc.weight, dim=1) for ex in model.experts])
            diversity = 0
            cnt = 0
            for i in range(expert_weights.size(0)):
                for j in range(i+1, expert_weights.size(0)):
                    diversity += (expert_weights[i] * expert_weights[j]).sum(dim=1).mean()
                    cnt += 1
            diversity_score = diversity / cnt if cnt > 0 else 0
            writer.add_scalar('ExpertDiversity', diversity_score, epoch)

            print(
                f"Epoch {epoch+1}/{CONFIG['max_epochs']} - "
                f"loss: {avg_loss:.4f}, "
                f"val_acc: {val_acc:.4f}, "
                f"val_auroc: {val_auroc:.4f}, "
                f"val_iou: {val_iou:.4f}"
            )

            score = val_auroc + val_iou   # combined monitoring metric
            if score > best_score:
                best_score = score
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(
                    OUTPUT_DIR, 'model_checkpoints/best_model.pth'))  # checkpoint
            else:
                no_improve += 1
            if no_improve >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        writer.close()
        # save final model weights
        torch.save(
            model.state_dict(),
            os.path.join(writer.log_dir, 'final_model.pth')
        )

        # rollback to best checkpoint
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'model_checkpoints', 'best_model.pth')))

        # Plot metrics
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Training Loss and Validation Accuracy')
        plt.show()

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for views, _, labels in val_loader:
                views = [v.to(device).to(memory_format=torch.channels_last)
                         for v in views]
                logits, _ = model(views)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure()
        plt.matshow(cm)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.show()

        # ---------
        # Held-out test evaluation
        # ---------
        model.eval()
        all_preds, all_labels = [], []
        all_iou = []
        with torch.no_grad():
            for views, masks, labels in test_loader:
                views = [v.to(device).to(memory_format=torch.channels_last)
                         for v in views]
                logits, seg_logits = model(views)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                # per-image IoU
                pred_masks = (seg_logits.sigmoid() > 0.5).long().squeeze(1)
                for pm, gm in zip(pred_masks.cpu(), masks):
                    all_iou.append(torchmetrics.functional.jaccard_index(
                        pm, gm, task="binary").item())

        # per-class metrics
        from sklearn.metrics import classification_report
        print("Test Classification Report:")
        print(classification_report(all_labels, all_preds, digits=4))
        print(f"Mean Test IoU: {np.mean(all_iou):.4f}")
        print("Test Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    except Exception as e:
        logging.error(f"Training aborted due to error: {e}")
        return


# NEW: Implement Focal Loss with label smoothing
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Expected to be tensor of shape (C,)
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing if needed
        if self.smoothing > 0:
            num_classes = inputs.size(1)
            smooth_target = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
            targets = smooth_target
        else:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt * targets
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        loss = focal_loss.sum(dim=1)
        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        else:
            return loss

# NEW: Define DropPath for stochastic depth
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        logging.critical(f"Fatal error in main: {e}")
