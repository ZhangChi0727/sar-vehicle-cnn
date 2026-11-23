"""2D Convolutional Neural Network (`ConvN`) for SAR vehicle target recognition.

Designed and implemented by Chi Zhang for 3-class MSTAR classification
(BMP2, BTR70, T72) on 32x32 grayscale SAR image crops.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ConvN(nn.Module):
    """Lightweight 2D CNN with Batch Normalization and log-softmax output.

    Expected input tensor shape: (batch_size, 1, 32, 32).
    Output tensor shape: (batch_size, num_classes) containing log-probabilities.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        # Block 1: (B, 1, 32, 32) -> (B, 16, 16, 16) via stride-2 3x3 convolution.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Block 2: (B, 16, 16, 16) -> (B, 32, 8, 8) via stride-2 3x3 convolution.
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Spatial downsampling: (B, 32, 8, 8) -> (B, 32, 4, 4).
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Classification head: flatten 32*4*4 = 512 features -> 128 -> 64 -> 32 -> num_classes.
        self.fc1 = nn.Sequential(nn.Linear(32 * 4 * 4, 128))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass and returns class log-probabilities."""
        x = self.conv1(x)       # -> (B, 16, 16, 16)
        x = self.conv2(x)       # -> (B, 32, 8, 8)
        x = self.downsample(x)  # -> (B, 32, 4, 4)

        # Flatten spatial feature maps into a 1D vector per sample: (B, 512).
        x = x.view(x.size(0), -1)
        x = self.fc1(x)         # -> (B, 128)
        x = self.fc2(x)         # -> (B, num_classes)

        # Apply log_softmax along the class dimension for NLLLoss / CrossEntropy compatibility.
        return F.log_softmax(x, dim=1)
