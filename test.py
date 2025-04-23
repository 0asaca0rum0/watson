import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, jaccard_score, roc_auc_score
import matplotlib.pyplot as plt

# MoE class defined earlier in this notebook, no external import needed

# --------- Configuration ---------
CONFIG = {
    'num_experts': 2,
    'image_size': 192,
    'checkpoint_path': '/content/runs/ham10000_experiment/best_model.pth',
    'image_dir': '/content/ham1000-segmentation-and-classification/images',
    'mask_dir': '/content/ham1000-segmentation-and-classification/masks',
    'ground_truth_csv': '/content/ham1000-segmentation-and-classification/GroundTruth.csv',
    'max_samples': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# --------- Model setup ---------
device = CONFIG['device']
model = MoE(num_classes=7).to(device)
model.eval()

ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device)
state_dict = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state_dict, strict=False)

# --------- Transforms ---------
transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
])

# --------- Load ground truth ---------
df = pd.read_csv(CONFIG['ground_truth_csv'])
df = df.sample(frac=1).reset_index(drop=True)   # randomize image order
image_col = df.columns[0]
class_cols = df.columns.tolist()[1:]   # ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']

y_true_cls, y_pred_cls, y_scores = [], [], []   # add y_scores for probabilities
iou_scores = []

# --------- Evaluation loop ---------
for idx, row in df.iterrows():
    if idx >= CONFIG['max_samples']:
        break

    filename = row[image_col]
    # ensure extension
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = f"{filename}.jpg"
    # determine true class from one-hot encoding
    true_cls = int(np.argmax([row[c] for c in class_cols]))
    y_true_cls.append(true_cls)

    # Load image
    img_path = os.path.join(CONFIG['image_dir'], filename)
    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)

    # Multi-view
    views = [inp.clone() for _ in range(CONFIG['num_experts'])]
    with torch.no_grad():
        cls_logits, seg_logits = model(views)
    # collect softmax probabilities
    probs = torch.softmax(cls_logits, dim=1).cpu().numpy().flatten()   # add
    y_scores.append(probs)                                              # add
    pred_cls = cls_logits.argmax(1).item()
    y_pred_cls.append(pred_cls)

    # Segmentation prediction
    pred_mask = seg_logits.argmax(1).squeeze(0).cpu().numpy()

    # Load GT mask (use '_segmentation' suffix)
    mask_name = os.path.splitext(filename)[0] + '_segmentation.png'
    mask_path = os.path.join(CONFIG['mask_dir'], mask_name)
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found: {mask_path}, skipping IoU computation.")
    else:
        # load and resize GT mask to CONFIG['image_size']
        gt_img = Image.open(mask_path)
        gt_img_resized = gt_img.resize(
            (CONFIG['image_size'], CONFIG['image_size']),
            resample=Image.NEAREST
        )
        gt_mask = np.array(gt_img_resized)
        iou = jaccard_score(
            gt_mask.flatten(),
            pred_mask.flatten(),
            average='macro',
            zero_division=0
        )
        iou_scores.append(iou)

# --------- Classification metrics ---------
accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_cls, y_pred_cls, average='weighted', zero_division=0)
cm = confusion_matrix(y_true_cls, y_pred_cls)

# compute ROC AUC
roc_auc = roc_auc_score(
    y_true_cls,
    np.stack(y_scores, axis=0),
    multi_class='ovr',
    average='weighted'
)

print("Classification Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")   # add ROC AUC

# --------- Segmentation metrics ---------
mean_iou = np.mean(iou_scores)
print("\nSegmentation Metrics:")
print(f"Mean IoU:  {mean_iou:.4f}")

# --------- Confusion matrix ---------
print("\nConfusion Matrix:")
print(cm)

# --------- Evaluation Summary ---------
threshold_acc = 0.8
threshold_iou = 0.5
threshold_roc = 0.85   # add threshold for ROC
print("\nEvaluation Summary:")
print(f"Accuracy ≥ {threshold_acc}: {accuracy >= threshold_acc}")
print(f"Mean IoU ≥ {threshold_iou}: {mean_iou >= threshold_iou}")
print(f"ROC AUC ≥ {threshold_roc}: {roc_auc >= threshold_roc}")   # add ROC threshold check
if accuracy >= threshold_acc and mean_iou >= threshold_iou and roc_auc >= threshold_roc:
    print("Model performance is good.")
elif (accuracy >= threshold_acc or mean_iou >= threshold_iou or roc_auc >= threshold_roc):
    print("Model performance is acceptable but could improve.")
else:
    print("Model performance is below expectations.")
