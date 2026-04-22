"""
CIFAR-10 Data Loading for bPC condition ([-1, 1] normalisation)
================================================================

Same as cifar10_data.py but normalises to [-1, 1] range for bPC's
tanh-compatible generative pathway, following Oliviers et al. (2025).

Oliviers et al. use pixel values in [-1, 1] for bPC (SM §I, Table 21).
Standard discPC uses [0, 1] with channel-wise normalisation.

Author: JP Cacioli
Date: April 2026
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Reuse the download/load functions from the main data module
import sys
sys.path.insert(0, "src")
from cifar10_data import load_cifar10


class CIFAR10Dataset_BPC(Dataset):
    """CIFAR-10 dataset with [-1, 1] normalisation for bPC.
    
    Following Oliviers et al. (2025, SM Table 21):
    Images are rescaled to [-1, 1] (no channel-wise mean/std normalisation).
    This matches the tanh output activation range in bPC's generative pathway.
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        """
        Args:
            data: (N, 3, 32, 32) float32 in [0, 1] (raw from loader)
            labels: (N,) int64
            augment: If True, apply random crop (pad=4) + horizontal flip
        """
        self.data = data
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].copy())  # (3, 32, 32) in [0, 1]
        label = int(self.labels[idx])
        
        if self.augment:
            # Random crop with padding 4
            img = torch.nn.functional.pad(img, (4, 4, 4, 4), mode='reflect')
            i = torch.randint(0, 9, (1,)).item()
            j = torch.randint(0, 9, (1,)).item()
            img = img[:, i:i+32, j:j+32]
            
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
        
        # Rescale from [0, 1] to [-1, 1]
        img = img * 2.0 - 1.0
        
        return img, label


def get_data_loaders_bpc(data_dir: str, batch_size: int = 128,
                          num_workers: int = 2) -> tuple:
    """Create CIFAR-10 DataLoaders with [-1, 1] normalisation for bPC."""
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)
    
    train_set = CIFAR10Dataset_BPC(train_data, train_labels, augment=True)
    test_set = CIFAR10Dataset_BPC(test_data, test_labels, augment=False)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


if __name__ == '__main__':
    print("Testing bPC data loader ([-1, 1] normalisation)...")
    train_loader, test_loader = get_data_loaders_bpc('./data', batch_size=128,
                                                       num_workers=0)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    x, y = next(iter(test_loader))
    print(f"Batch shape: {x.shape}")
    print(f"Value range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Expected: [-1.000, 1.000]")
    print(f"Match: {'YES' if x.min() >= -1.01 and x.max() <= 1.01 else 'NO'}")
    print("Done.")
