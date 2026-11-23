"""Unit tests verifying ConvN tensor shapes, probability normalization, and dataset loading."""

import os
import tempfile
import unittest
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.dataset import SARTrainDataset, TestDataset
from src.model import ConvN


class TestSARClassifier(unittest.TestCase):
    """Smoke tests for the SAR CNN model and dataset classes."""

    def test_convn_output_shape_and_probabilities(self) -> None:
        """ConvN should map (B, 1, 32, 32) inputs to normalized log-probabilities of shape (B, 3)."""
        model = ConvN(num_classes=3)
        model.eval()
        dummy_batch = torch.randn(4, 1, 32, 32)
        with torch.no_grad():
            log_probs = model(dummy_batch)

        self.assertEqual(tuple(log_probs.shape), (4, 3))
        # Exponentiating log_softmax outputs must sum to 1.0 along the class dimension.
        probs_sum = torch.exp(log_probs).sum(dim=1)
        self.assertTrue(torch.allclose(probs_sum, torch.ones(4), atol=1e-5))

    def test_datasets_load_and_split_cleanly(self) -> None:
        """SARTrainDataset and TestDataset should load grayscale images and split deterministically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths = []
            for idx in range(10):
                p = os.path.join(tmpdir, f"sample_{idx}.png")
                Image.new("L", (40, 40), color=idx * 20).save(p)
                img_paths.append(p)

            train_txt = os.path.join(tmpdir, "train.txt")
            with open(train_txt, "w", encoding="utf-8") as fh:
                for idx, p in enumerate(img_paths):
                    fh.write(f"{p} {idx % 3}\n")

            test_txt = os.path.join(tmpdir, "test.txt")
            with open(test_txt, "w", encoding="utf-8") as fh:
                for p in img_paths[:3]:
                    fh.write(f"{p}\n")

            tf = transforms.Compose([transforms.CenterCrop(32), transforms.ToTensor()])
            train_ds = SARTrainDataset(train_txt, split="train", train_ratio=0.7, seed=42, transform=tf)
            val_ds = SARTrainDataset(train_txt, split="val", train_ratio=0.7, seed=42, transform=tf)
            test_ds = TestDataset(test_txt, transform=tf)

            self.assertEqual(len(train_ds), 7)
            self.assertEqual(len(val_ds), 3)
            self.assertEqual(len(test_ds), 3)
            img_tensor, label = train_ds[0]
            self.assertEqual(tuple(img_tensor.shape), (1, 32, 32))
            self.assertIn(label, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
