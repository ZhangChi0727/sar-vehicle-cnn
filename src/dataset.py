"""PyTorch Dataset definitions for 3-class MSTAR SAR vehicle recognition.

This module separates labeled training/validation loading (`SARTrainDataset`)
from unlabeled test-set loading (`TestDataset`).
"""

import random
from typing import List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset


class SARTrainDataset(Dataset):
    """Dataset for labeled MSTAR SAR images with a reproducible 70/30 split.

    Each line in `txt_path` must follow the format:
        <image_path> <class_id>
    where class_id is an integer in {0: BMP2, 1: BTR70, 2: T72}.
    """

    def __init__(
        self,
        txt_path: str,
        split: str = "all",
        train_ratio: float = 0.7,
        seed: int = 42,
        transform=None,
    ) -> None:
        """Initializes the dataset and selects the requested split.

        Args:
            txt_path: Path to the annotation file listing image paths and labels.
            split: One of {"all", "train", "val"}.
            train_ratio: Fraction of samples assigned to the training split.
            seed: Random seed ensuring deterministic train/val partitioning.
            transform: Optional torchvision transform applied to each PIL image.
        """
        if split not in {"all", "train", "val"}:
            raise ValueError(f"Unsupported split '{split}'; expected 'all', 'train', or 'val'.")

        # Parse (image_path, integer_label) pairs from the index file.
        samples: List[Tuple[str, int]] = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                img_path, label_str = line.split()
                samples.append((img_path, int(label_str)))

        # Shuffle deterministically using a local RNG so train/val splits never overlap.
        if split in {"train", "val"}:
            rng = random.Random(seed)
            rng.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            self.samples = samples[:split_idx] if split == "train" else samples[split_idx:]
        else:
            self.samples = samples

        self.transform = transform

    def __getitem__(self, index: int):
        """Loads and transforms the (image, label) pair at `index`."""
        img_path, label = self.samples[index]
        # Convert to grayscale ('L') so input consistently has 1 channel.
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        """Returns the total number of samples in this split."""
        return len(self.samples)


class TestDataset(Dataset):
    """Dataset for unlabeled MSTAR SAR test images (implemented by Chi Zhang).

    Each line in `txt_path` contains a single image path without a label.
    """

    def __init__(self, txt_path: str, transform=None) -> None:
        """Reads test image paths from `txt_path`."""
        with open(txt_path, "r", encoding="utf-8") as fh:
            self.test_imgs: List[str] = [line.strip() for line in fh if line.strip()]
        self.transform = transform

    def __getitem__(self, index: int):
        """Loads and transforms the unlabeled test image at `index`."""
        img_path = self.test_imgs[index]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        """Returns the total number of test images."""
        return len(self.test_imgs)
