"""
Step 6 of 8: train CNN classifiers on the train split.

Input:  `data/splits/train.csv`, `val.csv`, `label_mapping.csv` and images in
        `DATA_ROOT`.
Output: best weights, a history CSV, and loss/accuracy curves in `OUTPUT_DIR`.

The script is safe for Kaggle: it uses CUDA with mixed precision when present,
and writes the best weights and history after EVERY epoch so a stopped session
does not lose progress. `--quick` trains 2 epochs on 200 images as a smoke test.
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    alexnet,
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
    resnet50,
)

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    DEFAULT_MODEL,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEARNING_RATE_PRETRAINED,
    LEARNING_RATE_SCRATCH,
    MODELS_DIR,
    NUM_WORKERS,
    OUTPUT_DIR,
    REPORTS_DIR,
    ROTATION_DEGREES,
    SEED,
    SPLITS_DIR,
    USE_TRAIN_AUGMENTATION,
    WARMUP_EPOCHS,
)
from src.dataset import EyeImageDataset  # noqa: E402
from src.utils import ensure_directories, get_device, set_seed  # noqa: E402

QUICK_TRAIN_IMAGES = 200
QUICK_VAL_IMAGES = 100
QUICK_EPOCHS = 2


class SimpleCNN(nn.Module):
    """A tiny hand-written CNN with three convolution blocks."""

    def __init__(self, number_of_classes):
        """Build three conv blocks, a small pooling step, and a final head."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Adaptive pooling keeps the head size fixed for any input image size.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(128 * 4 * 4, number_of_classes)

    def forward(self, images):
        """Return one score per class for each image in the batch."""
        features = self.pool(self.features(images))
        return self.classifier(features.flatten(start_dim=1))


def build_model(model_name, number_of_classes):
    """
    Build the requested model and return (model, classification_head).

    Pretrained models start from ImageNet weights. We keep the feature part
    frozen for a few warmup epochs so the new random head can learn first;
    after that we unfreeze the whole model.
    """
    if model_name == "simple_cnn":
        model = SimpleCNN(number_of_classes)
        return model, model.classifier, False

    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model, model.fc, True

    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model, model.fc, True

    if model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        old_head = model.classifier[1]
        model.classifier[1] = nn.Linear(old_head.in_features, number_of_classes)
        return model, model.classifier[1], True

    if model_name == "alexnet":
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        old_head = model.classifier[6]
        model.classifier[6] = nn.Linear(old_head.in_features, number_of_classes)
        return model, model.classifier[6], True

    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        old_head = model.classifier[1]
        model.classifier[1] = nn.Linear(old_head.in_features, number_of_classes)
        return model, model.classifier[1], True

    raise ValueError(f"Unknown model: {model_name}")


def make_transforms(image_size, is_training):
    """Build resize/normalize transforms; augmentation is train-only."""
    steps = [transforms.Resize((image_size, image_size))]

    if is_training and USE_TRAIN_AUGMENTATION:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(ROTATION_DEGREES),
                transforms.ColorJitter(
                    brightness=COLOR_JITTER_BRIGHTNESS,
                    contrast=COLOR_JITTER_CONTRAST,
                ),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def make_grad_scaler(enabled):
    """Return a GradScaler, with a fallback for older PyTorch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled):
    """Use float16 autocast on GPU, or do nothing on CPU."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def set_freeze(model, head, freeze):
    """Freeze the feature part while keeping the new head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = not freeze
    for parameter in head.parameters():
        parameter.requires_grad = True


def make_class_weights(train_labels, number_of_classes, device):
    """Give more weight to rare classes because the dataset is imbalanced."""
    counts = np.bincount(train_labels, minlength=number_of_classes)
    # max(counts, 1) avoids dividing by zero in a tiny --quick subset.
    weights = len(train_labels) / (number_of_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, is_training):
    """Train or validate for one epoch and return (average loss, accuracy)."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad() if not is_training else contextlib.nullcontext():
            with autocast_context(use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

        if is_training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        seen += batch_size

    return total_loss / seen, correct / seen


def save_curve(history, output_path):
    """Plot train/validation loss and accuracy from the history CSV."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train acc")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def main():
    """Train one CNN model, saving progress after every epoch."""
    parser = argparse.ArgumentParser(description="Train a CNN on eye images.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size.")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help="Epochs with only the new head trainable (0 disables this phase).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 epochs, 200 train images; outputs get a _quick suffix.",
    )
    args = parser.parse_args()

    print(f"Step 6/8: training {args.model}...")
    set_seed(SEED)
    ensure_directories(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)

    epochs = QUICK_EPOCHS if args.quick else args.epochs
    warmup_epochs = 0 if args.quick else min(args.warmup_epochs, epochs - 1)
    run_tag = f"{args.model}_quick" if args.quick else args.model
    best_model_path = MODELS_DIR / f"{run_tag}_best.pth"
    history_path = REPORTS_DIR / f"{run_tag}_history.csv"
    curve_path = FIGURES_DIR / f"{run_tag}_curves.png"

    if args.quick:
        print("QUICK MODE: 2 epochs, 200 train images, 100 val images.")

    train_frame = pd.read_csv(SPLITS_DIR / "train.csv")
    val_frame = pd.read_csv(SPLITS_DIR / "val.csv")
    label_mapping = pd.read_csv(SPLITS_DIR / "label_mapping.csv")
    number_of_classes = len(label_mapping)

    if args.quick:
        train_frame = train_frame.head(QUICK_TRAIN_IMAGES)
        val_frame = val_frame.head(QUICK_VAL_IMAGES)

    train_dataset = EyeImageDataset(
        train_frame, transform=make_transforms(args.img_size, is_training=True)
    )
    val_dataset = EyeImageDataset(
        val_frame, transform=make_transforms(args.img_size, is_training=False)
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = get_device()
    use_amp = device == "cuda"
    print(f"Device: {device}  (mixed precision: {use_amp})")
    print(f"Train images: {len(train_dataset)}  Val images: {len(val_dataset)}")

    model, head, is_pretrained = build_model(args.model, number_of_classes)
    model.to(device)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if is_pretrained and warmup_epochs > 0:
        # Warmup lets the new random head learn before we disturb the
        # ImageNet features with large gradients from the head.
        set_freeze(model, head, freeze=True)
        print(f"Phase 1: training only the new head for {warmup_epochs} epoch(s).")
    else:
        print("Training all layers from the start.")

    learning_rate = (
        LEARNING_RATE_SCRATCH if args.model == "simple_cnn" else LEARNING_RATE_PRETRAINED
    )

    def make_optimizer(current_model):
        trainable = [parameter for parameter in current_model.parameters() if parameter.requires_grad]
        return torch.optim.Adam(trainable, lr=learning_rate)

    optimizer = make_optimizer(model)
    class_weights = make_class_weights(train_frame["label"].to_numpy(), number_of_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = make_grad_scaler(use_amp)

    history_rows = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if is_pretrained and warmup_epochs > 0 and epoch == warmup_epochs + 1:
            # Phase 2: fine-tune the whole network with a fresh, smaller
            # optimizer that now includes the unfrozen feature layers.
            set_freeze(model, head, freeze=False)
            optimizer = make_optimizer(model)
            print("Phase 2: unfrozen all layers for fine-tuning.")

        phase_name = "warmup" if epoch <= warmup_epochs else "full"
        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, True
        )
        val_loss, val_accuracy = run_one_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_amp, False
        )

        history_rows.append(
            {
                "epoch": epoch,
                "phase": phase_name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        # Write progress NOW, not at the end: a stopped Kaggle session keeps
        # every completed epoch and the current best weights.
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(best_state, best_model_path)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} [{phase_name}] "
            f"train loss={train_loss:.4f} acc={train_accuracy:.3f} | "
            f"val loss={val_loss:.4f} acc={val_accuracy:.3f} | "
            f"time={elapsed:.1f}s"
        )
        print(f"  saved best weights: {best_model_path}")
        print(f"  saved history: {history_path}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    save_curve(pd.DataFrame(history_rows), curve_path)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Step 6 finished for {args.model}. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
