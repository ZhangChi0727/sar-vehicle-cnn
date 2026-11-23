"""Inference script to predict class labels for unlabeled MSTAR SAR test images."""

import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.dataset import TestDataset
from src.model import ConvN


def predict_test_set(
    test_loader: DataLoader,
    test_dataset: TestDataset,
    model: torch.nn.Module,
    out_path: str,
    device: torch.device,
) -> None:
    """Runs forward inference and writes '<image_path> <predicted_class>' lines."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            log_probs = model(images)
            # Select the class index (0, 1, or 2) with the highest log-probability.
            predictions.extend(log_probs.argmax(dim=1).cpu().tolist())

    with open(out_path, "w", encoding="utf-8") as fh:
        for img_path, pred_label in zip(test_dataset.test_imgs, predictions):
            fh.write(f"{img_path} {pred_label}\n")

    print(f"Wrote {len(predictions)} predictions to {out_path}")


def main() -> None:
    """Parses CLI arguments, loads trained weights, and runs test inference."""
    parser = argparse.ArgumentParser(description="Run inference with trained ConvN.")
    parser.add_argument("--test-txt", type=str, default="data/test.txt")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--out", type=str, default="result.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    test_ds = TestDataset(args.test_txt, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ConvN(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    predict_test_set(test_loader, test_ds, model, args.out, device)


if __name__ == "__main__":
    main()
