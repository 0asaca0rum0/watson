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
import torch.cuda.amp as amp  # Use proper amp import
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

# helper for optional TPU/XLA import & logging


def import_xla():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        logging.getLogger('torch_xla').setLevel(logging.ERROR)
        return xm, pl
    except ImportError:
        return None, None


xm, pl = import_xla()

# choose device (TPU if available else GPU/CPU)


def get_device():
    if xm and (os.getenv('COLAB_TPU_ADDR') or os.getenv('TPU_NAME')):
        return xm.xla_device()
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
    "batch_size": 128,              # increased for faster processing given 15GB VRAM
    "image_size": 160,             # kept same for speed/quality trade-off
    "base_lr": 2e-3,
    "warmup_epochs": 2,
    "max_epochs": 30,
    "accum_steps": 2,
    "mixed_precision": True,       # Ensure mixed precision is enabled
    "grad_clip": 1.0,
    "num_workers": 6,              # increased worker count leveraging 12GB RAM
    "cache_images": False,         # Disable caching for large datasets
    "train_task": "both",
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "patience": 6,       # early stopping patience
    "seed": SEED,
    "compile_model": True,         # Enable torch.compile if available (PyTorch 2.0+)
    "focal_gamma": 2.0,          # Focal Loss parameter
    "label_smoothing": 0.1,      # Label smoothing factor
    "ortho_weight": 0.01,        # Weight for expert orthogonality constraint
    "stochastic_depth": 0.2,     # Probability for stochastic depth
    "aug_level": "strong_v2"     # New augmentation preset
}

# TPU/GPU detection
# only attempt XLA import when TPU env vars are set
TPU_AVAILABLE = False
if os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_NAME'):
    logging.getLogger("torch_xla").setLevel(logging.ERROR)
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        _ = xm.xrt_world_size()       # verify TPU availability
        TPU_AVAILABLE = True
    except Exception:
        TPU_AVAILABLE = False

if TPU_AVAILABLE:
    replicas = xm.xrt_world_size()   # number of TPU cores
    CONFIG["batch_size"] = 16 * replicas  # adapt batch size
    device = xm.xla_device()

# custom collate to batch multi‑view inputs


def collate_moe(batch):
    views = list(zip(*[item[0] for item in batch]))
    views = [torch.stack(vs, dim=0) for vs in views]
    masks = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return views, masks, labels


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

# =====================
# Hybrid Sampling Strategy
# =====================
from torch.utils.data import WeightedRandomSampler, Subset

def create_balanced_sampler(dataset):
    # support both Dataset and Subset wrappers
    if isinstance(dataset, Subset):
        base = dataset.dataset
        idxs = dataset.indices
        labels = np.array(base.labels)[idxs]
    else:
        labels = np.array(dataset.labels)
    # compute class weights inversely proportional to frequency
    class_counts = np.bincount(labels)
    total = len(labels)
    weights_per_class = [ total/(len(class_counts)*c) for c in class_counts ]
    sample_weights = [ weights_per_class[int(l)] for l in labels ]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# =====================
# Dataset Modifications
# =====================
class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, transforms_list, augment=False):
        self.root_dir = root_dir
        self.transforms_list = transforms_list
        self.augment = augment
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
        if self.transforms_list:
            if self.augment:
                img = train_augmentations(img)  # Apply training augmentations
            else:
                img = val_test_augmentations(img)  # Apply validation/test transforms
            mask_tensor = transforms.functional.pil_to_tensor(mask)
            mask_tensor = (mask_tensor > 0).long().squeeze(0)
            return img, mask_tensor, self.labels[idx]
        else:
            return img, mask, self.labels[idx]

# =====================
# MoE Model with Checkpointing
# =====================


class ExpertModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # use smaller backbone
        self.encoder = timm.create_model(
            'efficientnet_b0', pretrained=True, features_only=True)
        if hasattr(self.encoder, 'set_gradient_checkpointing'):
            self.encoder.set_gradient_checkpointing(True)
        ch = self.encoder.feature_info[-1]['num_chs']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, num_classes)
        self.seg_head = nn.Sequential(
            nn.Conv2d(ch, 128, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1),  # binary segmentation
            nn.Upsample(size=(
                CONFIG['image_size'], CONFIG['image_size']), mode='bilinear', align_corners=False)
        )
        self.stoch_depth = CONFIG['stochastic_depth']

    def forward(self, x):
        # Use JIT tracing for encoder if not using gradient checkpointing
        feats = self.encoder(x)[-1]
        # Reduce memory usage by freeing intermediate activations
        pooled = self.avgpool(feats).flatten(1)
        cls_out = self.fc(pooled)
        seg_out = self.seg_head(feats)
        # Apply stochastic depth
        cls_out = drop_path(cls_out, self.stoch_depth, self.training)
        return cls_out, seg_out, pooled


class MoE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.experts = nn.ModuleList(
            [ExpertModel(num_classes) for _ in range(CONFIG['num_experts'])])
        feat_dim = self.experts[0].fc.in_features * CONFIG['num_experts']
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),              # Increased dropout to 0.5
            nn.Linear(256, CONFIG['num_experts'])
            # No Softmax here; will use Gumbel-Softmax in forward
        )

    def forward(self, views):
        # Ensure views length == num_experts
        assert len(views) == CONFIG['num_experts'], \
            f"Expected {CONFIG['num_experts']} views, but got {len(views)}"
        outs = [e(v) for e, v in zip(self.experts, views)]
        cls = torch.stack([o[0] for o in outs], dim=1)
        seg = torch.stack([o[1] for o in outs], dim=1)
        feats = torch.cat([o[2] for o in outs], dim=1)
        gate_logits = self.gate(feats)
        # Use Gumbel-Softmax with hard sampling for sparsity
        weights = F.gumbel_softmax(gate_logits, tau=1, hard=True, dim=1)
        cls_out = (cls * weights.unsqueeze(-1)).sum(1)
        seg_out = (
            seg * weights.view(-1, CONFIG['num_experts'], 1, 1, 1)).sum(1)
        # NEW: Compute expert orthogonality loss and attach as an attribute
        expert_weights = torch.stack(
            [F.normalize(ex.fc.weight, dim=1) for ex in self.experts])
        ortho_loss = 0
        for i in range(expert_weights.size(0)):
            for j in range(i+1, expert_weights.size(0)):
                ortho_loss += torch.mean(
                    (expert_weights[i] * expert_weights[j]).sum(dim=1))
        self.ortho_loss = CONFIG['ortho_weight'] * ortho_loss
        return cls_out, seg_out

# =====================
# Training & Visualization
# =====================

# helper to build DataLoader with optimal settings


def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
        prefetch_factor=2
    )


def train_model():
    best_val_acc = 0.0
    logging.info(f"Training on device: {device}")
    try:
        # full dataset without per-sample transforms for stratified splitting
        ds = HAM10000Dataset(INPUT_DIR, None)
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    # Build datasets with augmentations
    full_ds = HAM10000Dataset(INPUT_DIR, None)
    labels = np.array(full_ds.labels)
    idxs = np.arange(len(full_ds))

    # Stratified splits
    sss = StratifiedShuffleSplit(1, test_size=CONFIG['test_ratio'], random_state=SEED)
    trval_idx, test_idx = next(sss.split(idxs, labels))
    sss = StratifiedShuffleSplit(1, test_size=CONFIG['val_ratio']/(1-CONFIG['test_ratio']), random_state=SEED)
    train_idx, val_idx = next(sss.split(trval_idx, labels[trval_idx]))

    # Save indices
    np.save(os.path.join(OUTPUT_DIR, 'splits/train_idx.npy'), train_idx)
    np.save(os.path.join(OUTPUT_DIR, 'splits/val_idx.npy'), val_idx)
    np.save(os.path.join(OUTPUT_DIR, 'splits/test_idx.npy'), test_idx)

    # Build subsets with augmentations
    train_ds = torch.utils.data.Subset(HAM10000Dataset(INPUT_DIR, None, augment=True), train_idx)
    val_ds = torch.utils.data.Subset(HAM10000Dataset(INPUT_DIR, None, augment=False), val_idx)
    test_ds = torch.utils.data.Subset(HAM10000Dataset(INPUT_DIR, None, augment=False), test_idx)

    # Create sampler for balanced sampling
    sampler = create_balanced_sampler(train_ds)

    # Build loaders
    train_loader = make_loader(train_ds, sampler=sampler)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    if TPU_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader,   device)
        test_loader = pl.MpDeviceLoader(test_loader,  device)

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
