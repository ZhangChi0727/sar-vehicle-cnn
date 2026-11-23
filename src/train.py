"""Training and validation script for the SAR vehicle CNN classifier."""

import argparse
import os
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.dataset import SARTrainDataset
from src.model import ConvN


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one full training epoch and returns the average batch loss."""
    model.train()  # Enable BatchNorm training statistics.
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear gradients from the previous step.
        log_probs = model(images)      # Forward pass -> (B, 3) log-probabilities.
        loss = loss_fn(log_probs, labels)
        loss.backward()                # Backpropagate gradients.
        optimizer.step()               # Update model weights.

        total_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"  [Batch {batch_idx:03d}] loss = {loss.item():.6f}")

    return total_loss / max(len(dataloader), 1)


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates model accuracy and average loss on the validation split."""
    model.eval()  # Freeze BatchNorm running statistics for deterministic evaluation.
    total_loss = 0.0
    correct = 0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            total_loss += loss_fn(log_probs, labels).item() * images.size(0)
            preds = log_probs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, accuracy


def main() -> None:
    """Parses CLI arguments and runs the full training/validation pipeline."""
    parser = argparse.ArgumentParser(description="Train ConvN on MSTAR SAR images.")
    parser.add_argument("--train-txt", type=str, default="data/train.txt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize all SAR images to 32x32 tensors.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    train_ds = SARTrainDataset(args.train_txt, split="train", transform=train_transform)
    val_ds = SARTrainDataset(args.train_txt, split="val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConvN(num_classes=3).to(device)
    loss_fn = nn.NLLLoss()  # Paired with log_softmax output from ConvN.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)
        scheduler.step()

        print(
            f"  Summary: train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={100.0 * val_acc:.2f}%"
        )

        # Save checkpoint whenever validation accuracy improves.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    print(f"Training complete. Best validation accuracy: {100.0 * best_acc:.2f}%")


if __name__ == "__main__":
    main()
