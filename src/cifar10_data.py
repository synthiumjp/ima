"""
CIFAR-10 Data Loading (no torchvision dependency)
==================================================

Downloads CIFAR-10 from the official source and provides DataLoaders
with the pre-registered augmentation (random crop pad=4, hflip, normalise).

Replaces torchvision.datasets.CIFAR10 and torchvision.transforms.
"""

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-py"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def download_cifar10(data_dir: str) -> Path:
    """Download and extract CIFAR-10 if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted = data_dir / CIFAR10_FOLDER
    if extracted.exists() and (extracted / "data_batch_1").exists():
        return extracted

    tar_path = data_dir / CIFAR10_FILENAME
    if not tar_path.exists():
        print(f"Downloading CIFAR-10 to {tar_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        print("Download complete.")

    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")

    return extracted


def _load_batch(path: str) -> tuple:
    """Load a single CIFAR-10 batch file."""
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = d[b'data']       # (N, 3072) uint8
    labels = d[b'labels']   # list of ints
    return data, labels


def load_cifar10(data_dir: str) -> tuple:
    """Load all CIFAR-10 data.
    
    Returns:
        train_data: (50000, 3, 32, 32) float32 [0,1]
        train_labels: (50000,) int64
        test_data: (10000, 3, 32, 32) float32 [0,1]
        test_labels: (10000,) int64
    """
    folder = download_cifar10(data_dir)

    # Training batches
    train_data_list, train_label_list = [], []
    for i in range(1, 6):
        data, labels = _load_batch(str(folder / f"data_batch_{i}"))
        train_data_list.append(data)
        train_label_list.extend(labels)

    train_data = np.concatenate(train_data_list, axis=0)  # (50000, 3072)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    train_labels = np.array(train_label_list, dtype=np.int64)

    # Test batch
    test_data, test_labels = _load_batch(str(folder / "test_batch"))
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = np.array(test_labels, dtype=np.int64)

    return train_data, train_labels, test_data, test_labels


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with optional augmentation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        """
        Args:
            data: (N, 3, 32, 32) float32 in [0, 1]
            labels: (N,) int64
            augment: If True, apply random crop (pad=4) + horizontal flip
        """
        self.data = data
        self.labels = labels
        self.augment = augment

        # Pre-compute normalisation tensors
        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].copy())  # (3, 32, 32)
        label = int(self.labels[idx])

        if self.augment:
            # Random crop with padding 4 (pre-reg §3.4)
            img = torch.nn.functional.pad(img, (4, 4, 4, 4), mode='reflect')
            # Random crop back to 32x32
            i = torch.randint(0, 9, (1,)).item()  # 0 to 8 (40-32=8)
            j = torch.randint(0, 9, (1,)).item()
            img = img[:, i:i+32, j:j+32]

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)

        # Normalise
        img = (img - self.mean) / self.std

        return img, label


def get_data_loaders(data_dir: str, batch_size: int = 128,
                     num_workers: int = 2) -> tuple:
    """Create CIFAR-10 train and test DataLoaders.
    
    Pre-reg §3.4: random crop pad=4, horizontal flip, normalisation.
    """
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)

    train_set = CIFAR10Dataset(train_data, train_labels, augment=True)
    test_set = CIFAR10Dataset(test_data, test_labels, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == '__main__':
    print("Testing CIFAR-10 data loading...")
    train_loader, test_loader = get_data_loaders('./data', batch_size=128)
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, Labels: {y.shape}")
    print(f"Value range: [{x.min():.3f}, {x.max():.3f}]")
    print("✓ Data loading OK")
