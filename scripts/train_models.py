"""
Phase 3: CNN Training Pipeline for Knee Osteoporosis Classification

Models:
1. Custom Baseline CNN
2. ResNet18 (transfer learning)
3. EfficientNet-B0 (transfer learning)
4. DenseNet121 (transfer learning)

Dataset: dataset/strict_clean_knee_osteoporosis/
         894 images, 3 classes, 70/15/15 split

IMPORTANT:
- Does NOT modify the dataset.
- Does NOT modify the React/FastAPI application.
- Saves all outputs to models/checkpoints/<model_name>/
"""

import os
import sys
import json
import time
import copy
import random
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models, datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42
NUM_CLASSES = 3
CLASS_NAMES = ['Normal', 'Osteopenia', 'Osteoporosis']
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows compatibility
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 0.001
GRAD_CLIP = 1.0

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(WORKSPACE_ROOT, 'dataset', 'strict_clean_knee_osteoporosis')
CHECKPOINT_ROOT = os.path.join(WORKSPACE_ROOT, 'models', 'checkpoints')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# DATA LOADING
# =====================================================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def load_datasets():
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'train'), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'val'), transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, 'test'), transform=val_transform
    )

    # Verify class order matches our expected names
    assert train_dataset.classes == CLASS_NAMES, \
        f"Class order mismatch: {train_dataset.classes} != {CLASS_NAMES}"

    return train_dataset, val_dataset, test_dataset


def get_class_weights(dataset):
    """Compute inverse-frequency class weights."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    total = len(targets)
    weights = total / (NUM_CLASSES * class_counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset):
    """Create a WeightedRandomSampler for class-balanced mini-batches."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    return train_loader, val_loader, test_loader


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

class CustomBaselineCNN(nn.Module):
    """Lightweight custom CNN for baseline comparison (~1.2M params)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x224x224 -> 32x112x112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x112x112 -> 64x56x56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=NUM_CLASSES):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_densenet121(num_classes=NUM_CLASSES):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


MODEL_BUILDERS = {
    'custom_cnn': lambda: CustomBaselineCNN(),
    'resnet18': build_resnet18,
    'efficientnet_b0': build_efficientnet_b0,
    'densenet121': build_densenet121,
}


def freeze_backbone(model, model_name):
    """Freeze all layers except the classification head."""
    if model_name == 'custom_cnn':
        return  # Train all layers for custom CNN

    for param in model.parameters():
        param.requires_grad = False

    if model_name == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'efficientnet_b0':
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'densenet121':
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model, model_name, phase='head'):
    """Create optimizer with appropriate LR per phase."""
    if model_name == 'custom_cnn':
        return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if phase == 'head':
        # Only head parameters are trainable
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    else:
        # Discriminative LR: backbone lower, head higher
        if model_name == 'resnet18':
            backbone_params = [p for n, p in model.named_parameters()
                               if not n.startswith('fc')]
            head_params = list(model.fc.parameters())
        elif model_name == 'efficientnet_b0':
            backbone_params = [p for n, p in model.named_parameters()
                               if not n.startswith('classifier')]
            head_params = list(model.classifier.parameters())
        elif model_name == 'densenet121':
            backbone_params = [p for n, p in model.named_parameters()
                               if not n.startswith('classifier')]
            head_params = list(model.classifier.parameters())
        else:
            return optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        return optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4},
        ], weight_decay=1e-4)


# =====================================================================
# TRAINING LOOP
# =====================================================================

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(model, model_name, train_loader, val_loader, criterion, device, checkpoint_dir, resume=False):
    """Full training loop with phase-based unfreezing for transfer learning and safe checkpoint resumption."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, 'best_model.pth')
    last_ckpt_path = os.path.join(checkpoint_dir, 'last_epoch.pth')

    start_epoch = 1
    best_val_f1 = 0.0
    best_model_state = None
    training_log = {
        'model_name': model_name,
        'start_time': datetime.now().isoformat(),
        'config': {
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
            'seed': SEED,
            'img_size': IMG_SIZE,
            'architecture': model_name,
        },
        'epochs': []
    }

    # Phase A: Head-only (transfer learning models) or full (custom CNN)
    is_transfer = model_name != 'custom_cnn'
    head_epochs = 10 if is_transfer else 0

    if is_transfer:
        freeze_backbone(model, model_name)

    optimizer = get_optimizer(model, model_name, phase='head' if is_transfer else 'full')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-7
    )
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Handle resumption if requested
    if resume:
        resume_file = last_ckpt_path if os.path.exists(last_ckpt_path) else (best_ckpt_path if os.path.exists(best_ckpt_path) else None)
        if resume_file:
            print(f"  >>> Resuming {model_name} from checkpoint: {resume_file} <<<", flush=True)
            ckpt = torch.load(resume_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"  Warning: could not restore optimizer state: {e}", flush=True)
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_f1 = ckpt.get('best_val_f1', 0.0)
            best_model_state = copy.deepcopy(model.state_dict())
            early_stopping.best_score = best_val_f1
            print(f"  Resumed at epoch {start_epoch}, previous best val F1: {best_val_f1:.4f}", flush=True)

    total_epochs_trained = start_epoch - 1

    print(f"\n{'='*60}", flush=True)
    print(f"  Training: {model_name}", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # Phase transition: unfreeze at epoch head_epochs+1
        if is_transfer and epoch == head_epochs + 1:
            print(f"\n  >>> Epoch {epoch}: Unfreezing all layers for fine-tuning <<<", flush=True)
            unfreeze_all(model)
            optimizer = get_optimizer(model, model_name, phase='finetune')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-7
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {trainable:,}", flush=True)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_data = {
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'train_acc': round(train_acc, 4),
            'train_f1': round(train_f1, 4),
            'val_loss': round(val_loss, 5),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'lr': current_lr,
        }
        training_log['epochs'].append(epoch_data)
        total_epochs_trained = epoch

        # Checkpoint best model based on VALIDATION macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'class_names': CLASS_NAMES,
                'training_config': training_log['config'],
            }, best_ckpt_path)

        # Print progress with immediate flush
        marker = ' *BEST*' if val_f1 >= best_val_f1 else ''
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train L={train_loss:.4f} A={train_acc:.3f} F1={train_f1:.3f} | "
              f"Val L={val_loss:.4f} A={val_acc:.3f} F1={val_f1:.3f} | "
              f"LR={current_lr:.2e}{marker}", flush=True)

        # Early stopping based on VALIDATION macro F1
        if early_stopping.step(val_f1):
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})", flush=True)
            break

    # Save last epoch
    torch.save({
        'epoch': total_epochs_trained,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'class_names': CLASS_NAMES,
        'training_config': training_log['config'],
    }, last_ckpt_path)

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_log['end_time'] = datetime.now().isoformat()
    training_log['total_epochs'] = total_epochs_trained
    training_log['best_val_f1'] = round(best_val_f1, 4)

    with open(os.path.join(checkpoint_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n  Finished {model_name}: best_val_f1={best_val_f1:.4f} at epoch {total_epochs_trained}", flush=True)

    return model, training_log


# =====================================================================
# EVALUATION & VISUALIZATION (VALIDATION & HELD-OUT TEST SEPARATED)
# =====================================================================

def compute_bootstrap_ci(y_true, y_pred, y_probs, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(SEED)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue
    if not scores:
        return 0, 0, 0
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper


def evaluate_model(model, loader, criterion, device, checkpoint_dir, model_name, split_name='val'):
    """Evaluation with metrics, plots, and bootstrap CIs for a specific split (val or test)."""
    print(f"\n  Evaluating {model_name} on {split_name} set...", flush=True)
    split_loss, split_acc, split_f1, preds, labels, probs = evaluate(
        model, loader, criterion, device
    )

    # Core metrics
    metrics = {
        'split': split_name,
        'loss': round(split_loss, 5),
        'accuracy': round(split_acc, 4),
        'macro_precision': round(precision_score(labels, preds, average='macro', zero_division=0), 4),
        'macro_recall': round(recall_score(labels, preds, average='macro', zero_division=0), 4),
        'macro_f1': round(split_f1, 4),
    }

    # Per-class metrics (Precision, Recall/Sensitivity, Specificity, F1-Score)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_cls = (2 * prec * sensitivity / (prec + sensitivity)) if (prec + sensitivity) > 0 else 0.0
        per_class[cls] = {
            'precision': round(prec, 4),
            'sensitivity': round(sensitivity, 4),
            'recall': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'f1_score': round(f1_cls, 4),
            'support': int(tp + fn),
        }
    metrics['per_class'] = per_class
    metrics['confusion_matrix'] = cm.tolist()

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        metrics['roc_auc_macro'] = round(roc_auc_macro, 4)

        # Per-class ROC-AUC
        for i, cls in enumerate(CLASS_NAMES):
            binary_labels = (labels == i).astype(int)
            auc_i = roc_auc_score(binary_labels, probs[:, i])
            metrics['per_class'][cls]['roc_auc'] = round(auc_i, 4)
    except Exception as e:
        metrics['roc_auc_macro'] = None
        print(f"    Warning: ROC-AUC computation failed: {e}", flush=True)

    # Expected Calibration Error (ECE)
    n_bins = 10
    confidences = probs.max(axis=1)
    accuracies_bin = (preds == labels).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies_bin[mask].mean()
            ece += mask.sum() / len(labels) * abs(avg_acc - avg_conf)
    metrics['expected_calibration_error'] = round(ece, 4)

    # Bootstrap CIs for key metrics
    print(f"    Computing bootstrap confidence intervals...", flush=True)
    for metric_name, metric_fn in [
        ('accuracy', lambda y, p: accuracy_score(y, p)),
        ('macro_f1', lambda y, p: f1_score(y, p, average='macro', zero_division=0)),
        ('macro_precision', lambda y, p: precision_score(y, p, average='macro', zero_division=0)),
        ('macro_recall', lambda y, p: recall_score(y, p, average='macro', zero_division=0)),
    ]:
        mean, lower, upper = compute_bootstrap_ci(labels, preds, probs, metric_fn)
        metrics[f'{metric_name}_ci95'] = [round(lower, 4), round(upper, 4)]

    # Save metrics JSON
    metrics_path = os.path.join(checkpoint_dir, f'{split_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # --- PLOTS ---

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} — Confusion Matrix ({split_name.capitalize()} Set)')
    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f'{split_name}_confusion_matrix.png'), dpi=150)
    plt.close(fig)

    # 2. ROC curves (one-vs-rest)
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASS_NAMES):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc_i:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} — One-vs-Rest ROC Curves ({split_name.capitalize()} Set)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f'{split_name}_roc_curves.png'), dpi=150)
    plt.close(fig)

    # 3. Calibration reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_means_conf = []
    bin_means_acc = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_means_conf.append(confidences[mask].mean())
            bin_means_acc.append(accuracies_bin[mask].mean())
    ax.bar(bin_means_conf, bin_means_acc, width=0.08, alpha=0.7, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{model_name} — Calibration Diagram ({split_name.capitalize()})\nECE={ece:.4f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, f'{split_name}_calibration.png'), dpi=150)
    plt.close(fig)

    print(f"    {split_name.capitalize()} Acc={split_acc:.4f}  Macro F1={split_f1:.4f}  "
          f"AUC={metrics.get('roc_auc_macro', 'N/A')}", flush=True)

    return metrics


def plot_training_curves(training_log, checkpoint_dir):
    """Plot training/validation loss, accuracy, and F1 curves."""
    epochs_data = training_log.get('epochs', [])
    if not epochs_data:
        return
    epochs = [e['epoch'] for e in epochs_data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_name = training_log['model_name']

    # Loss
    axes[0].plot(epochs, [e['train_loss'] for e in epochs_data], label='Train', color='#2196F3')
    axes[0].plot(epochs, [e['val_loss'] for e in epochs_data], label='Val', color='#F44336')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [e['train_acc'] for e in epochs_data], label='Train', color='#2196F3')
    axes[1].plot(epochs, [e['val_acc'] for e in epochs_data], label='Val', color='#F44336')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, [e['train_f1'] for e in epochs_data], label='Train', color='#2196F3')
    axes[2].plot(epochs, [e['val_f1'] for e in epochs_data], label='Val', color='#F44336')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title(f'{model_name} — Macro F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} Training Curves', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_evaluation_report(metrics, training_log, checkpoint_dir, model_name, split_name='val'):
    """Generate per-model evaluation report in Markdown."""
    cm = np.array(metrics['confusion_matrix'])
    report = f"""# {model_name} — {split_name.capitalize()} Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Split evaluated:** `{split_name}` (n={sum(cm.sum(axis=1))})  
**Epochs trained:** {training_log.get('total_epochs', 'N/A')}  
**Best validation F1:** {training_log.get('best_val_f1', metrics['macro_f1']):.4f}

## Results Summary

| Metric | Value | 95% CI |
|:---|:---:|:---:|
| Accuracy | {metrics['accuracy']:.4f} | [{metrics.get('accuracy_ci95', ['N/A', 'N/A'])[0]}, {metrics.get('accuracy_ci95', ['N/A', 'N/A'])[1]}] |
| Macro Precision | {metrics['macro_precision']:.4f} | [{metrics.get('macro_precision_ci95', ['N/A', 'N/A'])[0]}, {metrics.get('macro_precision_ci95', ['N/A', 'N/A'])[1]}] |
| Macro Recall | {metrics['macro_recall']:.4f} | [{metrics.get('macro_recall_ci95', ['N/A', 'N/A'])[0]}, {metrics.get('macro_recall_ci95', ['N/A', 'N/A'])[1]}] |
| Macro F1 | {metrics['macro_f1']:.4f} | [{metrics.get('macro_f1_ci95', ['N/A', 'N/A'])[0]}, {metrics.get('macro_f1_ci95', ['N/A', 'N/A'])[1]}] |
| ROC-AUC (macro) | {metrics.get('roc_auc_macro', 'N/A')} | — |
| ECE | {metrics['expected_calibration_error']:.4f} | — |

## Per-Class Performance

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for cls in CLASS_NAMES:
        pc = metrics['per_class'][cls]
        report += (f"| {cls} | {pc.get('precision', 'N/A')} | {pc['sensitivity']:.4f} | "
                   f"{pc['specificity']:.4f} | {pc.get('f1_score', 'N/A')} | {pc.get('roc_auc', 'N/A')} | {pc['support']} |\n")

    report += f"""
> [!NOTE]
> **Dataset & Clinical Disclaimer:**
> Evaluation performed on {split_name} split of `strict_clean_knee_osteoporosis` (n={sum(cm.sum(axis=1))}).
> 76.5% of images in the source dataset lack patient IDs and rely on unverified folder placement.
> Model confidences are research estimates and are not clinically validated diagnostic probabilities.

## Confusion Matrix

![Confusion Matrix]({split_name}_confusion_matrix.png)

## ROC Curves

![ROC Curves]({split_name}_roc_curves.png)

## Calibration

![Calibration]({split_name}_calibration.png)
"""
    if os.path.exists(os.path.join(checkpoint_dir, 'training_curves.png')):
        report += """
## Training Curves

![Training Curves](training_curves.png)
"""

    report_path = os.path.join(checkpoint_dir, f'{split_name}_evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved report: {report_path}", flush=True)


def generate_comparison_report(all_results, comparison_dir, split_name='val'):
    """Generate cross-model comparison report based on validation metrics."""
    report = f"""# Model Comparison Report ({split_name.capitalize()} Set)

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Evaluation Split:** `{split_name}`  
**Dataset:** strict_clean_knee_osteoporosis (Validation n=134)  
**Primary Model Selection Metric:** Validation Macro F1-Score  

## Comparison Summary

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC | ECE | Epochs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    best_f1 = -1
    best_model = ''
    for model_name, (metrics, training_log) in all_results.items():
        f1 = metrics['macro_f1']
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name
        report += (f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                   f"{metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | "
                   f"{metrics.get('roc_auc_macro', 'N/A')} | {metrics['expected_calibration_error']:.4f} | "
                   f"{training_log.get('total_epochs', 'N/A')} |\n")

    report += f"""
## Model Selection Recommendation

**Top-performing model by Validation Macro F1:** `{best_model}` (Val Macro F1 = {best_f1:.4f})

> [!IMPORTANT]
> All candidate comparisons above are performed STRICTLY on the **Validation Set (n=134)**.
> The **Held-Out Test Set (n=135)** remains completely untouched and blinded during model exploration.
> Test set evaluation should only be performed once the final architecture is locked and approved.
"""

    comparison_path = os.path.join(comparison_dir, f'{split_name}_comparison_report.md')
    with open(comparison_path, 'w') as f:
        f.write(report)
    print(f"  Saved comparison report: {comparison_path}", flush=True)
    return best_model


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CNN models for knee osteoporosis classification')
    parser.add_argument('--models', nargs='+', default=list(MODEL_BUILDERS.keys()),
                        help='Models to train (default: all 4)')
    parser.add_argument('--skip-completed', action='store_true', default=True,
                        help='Skip training if best_model.pth already exists, but include in validation comparison')
    parser.add_argument('--no-skip-completed', dest='skip_completed', action='store_false',
                        help='Force retraining even if checkpoint exists')
    parser.add_argument('--resume', type=str, default=None,
                        help='Model name to resume training from checkpoint (e.g. --resume custom_cnn)')
    parser.add_argument('--evaluate-test', type=str, default=None,
                        help='Explicitly evaluate ONLY the specified model on the held-out test set (DO NOT USE DURING TRAINING)')
    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {DATASET_DIR}", flush=True)

    # Load datasets
    print("\nLoading datasets...", flush=True)
    train_dataset, val_dataset, test_dataset = load_datasets()
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # NOTE ON CLASS BALANCING:
    # We use WeightedRandomSampler in train_loader to ensure balanced mini-batch sampling across classes.
    # Therefore, we use standard unweighted CrossEntropyLoss() to avoid double-weighting minority classes.
    criterion = nn.CrossEntropyLoss()
    print("  Loss: Standard CrossEntropyLoss (class balancing handled by WeightedRandomSampler)", flush=True)

    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    # -------------------------------------------------------------
    # STANDALONE HELD-OUT TEST EVALUATION (ONLY IF EXPLICITLY REQUESTED)
    # -------------------------------------------------------------
    if args.evaluate_test:
        model_name = args.evaluate_test
        if model_name not in MODEL_BUILDERS:
            print(f"Error: Unknown model {model_name}", flush=True)
            return

        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, model_name)
        best_ckpt = os.path.join(checkpoint_dir, 'best_model.pth')
        if not os.path.exists(best_ckpt):
            print(f"Error: No checkpoint found at {best_ckpt}", flush=True)
            return

        print(f"\n{'='*60}", flush=True)
        print(f"  EXPLICIT HELD-OUT TEST EVALUATION: {model_name}", flush=True)
        print(f"  Loading weights from: {best_ckpt}", flush=True)
        print(f"{'='*60}", flush=True)

        model = MODEL_BUILDERS[model_name]()
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)

        training_log = {}
        log_file = os.path.join(checkpoint_dir, 'training_log.json')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                training_log = json.load(f)
        else:
            training_log = {'total_epochs': ckpt.get('epoch', 'N/A'), 'best_val_f1': ckpt.get('best_val_f1', 0.0)}

        test_metrics = evaluate_model(
            model, test_loader, criterion, device, checkpoint_dir, model_name, split_name='test'
        )
        generate_evaluation_report(test_metrics, training_log, checkpoint_dir, model_name, split_name='test')
        print(f"\n  Held-out test evaluation completed for {model_name}.", flush=True)
        return

    # -------------------------------------------------------------
    # MODEL TRAINING AND VALIDATION-ONLY SELECTION PIPELINE
    # -------------------------------------------------------------
    all_results = {}

    for model_name in args.models:
        if model_name not in MODEL_BUILDERS:
            print(f"Unknown model: {model_name}", flush=True)
            continue

        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, model_name)
        best_ckpt = os.path.join(checkpoint_dir, 'best_model.pth')
        start_time = time.time()

        # Check if already trained and skip requested
        if args.skip_completed and os.path.exists(best_ckpt) and (args.resume != model_name):
            print(f"\n{'='*60}", flush=True)
            print(f"  Found existing checkpoint for {model_name}: {best_ckpt}", flush=True)
            print(f"  Skipping training. Loading best checkpoint for validation evaluation...", flush=True)
            print(f"{'='*60}", flush=True)

            model = MODEL_BUILDERS[model_name]()
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)

            training_log = {}
            log_file = os.path.join(checkpoint_dir, 'training_log.json')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    training_log = json.load(f)
            else:
                training_log = {
                    'model_name': model_name,
                    'total_epochs': ckpt.get('epoch', 'N/A'),
                    'best_val_f1': ckpt.get('best_val_f1', 0.0),
                    'config': ckpt.get('training_config', {}),
                    'note': 'Original checkpoint trained with previous loss configuration (double-weighted class loss)' if model_name == 'custom_cnn' else '',
                    'epochs': []
                }
                with open(log_file, 'w') as f:
                    json.dump(training_log, f, indent=2)

            val_metrics = evaluate_model(
                model, val_loader, criterion, device, checkpoint_dir, model_name, split_name='val'
            )
            generate_evaluation_report(val_metrics, training_log, checkpoint_dir, model_name, split_name='val')
            all_results[model_name] = (val_metrics, training_log)
            continue

        # Build fresh or resume model
        model = MODEL_BUILDERS[model_name]()
        model = model.to(device)

        should_resume = (args.resume == model_name)
        model, training_log = train_model(
            model, model_name, train_loader, val_loader, criterion, device, checkpoint_dir, resume=should_resume
        )

        # Plot training curves
        plot_training_curves(training_log, checkpoint_dir)

        # Evaluate strictly on VALIDATION set
        val_metrics = evaluate_model(
            model, val_loader, criterion, device, checkpoint_dir, model_name, split_name='val'
        )

        # Generate per-model validation report
        generate_evaluation_report(val_metrics, training_log, checkpoint_dir, model_name, split_name='val')

        elapsed = time.time() - start_time
        print(f"  Total time for {model_name}: {elapsed/60:.1f} min", flush=True)

        all_results[model_name] = (val_metrics, training_log)

    # Cross-model validation comparison and ranking
    if len(all_results) > 1:
        print("\n" + "=" * 60, flush=True)
        print("  Generating cross-model validation comparison...", flush=True)
        best = generate_comparison_report(all_results, CHECKPOINT_ROOT, split_name='val')
        print(f"  Top candidate model by Validation Macro F1: {best}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ALL TRAINING AND VALIDATION SELECTION COMPLETE", flush=True)
    print("  The held-out test set remains strictly untouched.", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()

