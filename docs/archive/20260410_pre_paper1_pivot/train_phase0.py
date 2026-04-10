"""
IMA Phase 0: PCN Training on CIFAR-10
=======================================

Pre-registration v3.1 §3.4 training protocol:
  - Dataset: CIFAR-10, augmentation (random crop pad=4, hflip, normalise)
  - Optimiser: AdamW, lr=1e-3, weight_decay=1e-4
  - Scheduler: Cosine annealing
  - Epochs: 50-100 (until test acc plateaus for 10 consecutive epochs)
  - Batch size: 128
  - T_infer: 20, eta_infer: 0.1
  - Seeds: 42, 123, 456

Usage:
  python train_phase0.py --seed 42
  python train_phase0.py --seed 42 --device cpu  # for testing
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from conv_pcn import ConvPCN, train_pcn_epoch, evaluate_pcn
from cifar10_data import get_data_loaders

# =============================================================================
# Pre-registered constants (v3.1 §3.4)
# =============================================================================

VALID_SEEDS = [42, 123, 456]
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
T_INFER = 20
ETA_INFER = 0.1
MAX_EPOCHS = 100
PATIENCE = 10  # plateau = 10 consecutive epochs without improvement
NUM_CLASSES = 10


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(seed: int, device: str, data_dir: str, output_dir: str,
          max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE):
    """Full Phase 0 training for one seed.
    
    Implements the pre-registered stopping rule:
    "50-100 epochs (until test accuracy plateaus for 10 consecutive epochs)"
    """
    print(f"\n{'='*70}")
    print(f"IMA Phase 0 — PCN Training")
    print(f"Seed: {seed} | Device: {device}")
    print(f"{'='*70}\n")

    # Validate seed
    if seed not in VALID_SEEDS:
        print(f"WARNING: Seed {seed} is not pre-registered. "
              f"Valid seeds: {VALID_SEEDS}")
        print("Continuing for debugging, but results are not confirmatory.\n")

    set_seed(seed)

    # Create output directory
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_loader.dataset):,} samples")
    print(f"Test:  {len(test_loader.dataset):,} samples\n")

    # Model
    model = ConvPCN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimiser (pre-reg: AdamW, lr=1e-3, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler (pre-reg: cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop with early stopping
    history = []
    best_test_acc = 0.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        print(f"Epoch {epoch}/{max_epochs} (lr={scheduler.get_last_lr()[0]:.6f})")
        train_metrics = train_pcn_epoch(
            model, train_loader, optimizer,
            T_infer=T_INFER, eta_infer=ETA_INFER, device=device
        )

        # Evaluate
        test_metrics = evaluate_pcn(
            model, test_loader,
            T_infer=T_INFER, eta_infer=ETA_INFER, device=device
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        # Log
        entry = {
            'epoch': epoch,
            'seed': seed,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'test_acc': test_metrics['accuracy'],
            'lr': scheduler.get_last_lr()[0],
            'epoch_time_s': epoch_time,
            'total_time_s': total_time,
        }
        history.append(entry)

        print(f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  acc={test_metrics['accuracy']:.4f}")
        print(f"  Time:  {epoch_time:.1f}s (total: {total_time/60:.1f}m)")

        # Early stopping check
        if test_metrics['accuracy'] > best_test_acc + 0.001:
            best_test_acc = test_metrics['accuracy']
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc,
                'train_loss': train_metrics['loss'],
            }, seed_dir / 'best_model.pt')
            print(f"  ★ New best: {best_test_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

        # Save training log after each epoch (crash recovery)
        with open(seed_dir / 'training_log.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Minimum 50 epochs before early stopping (pre-reg: "50-100 epochs")
        if epoch >= 50 and epochs_without_improvement >= patience:
            print(f"\n✓ Stopping: test accuracy plateaued for {patience} epochs.")
            break

        print()

    # Save final model
    torch.save({
        'epoch': epoch,
        'seed': seed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_metrics['accuracy'],
        'best_test_acc': best_test_acc,
    }, seed_dir / 'final_model.pt')

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete — Seed {seed}")
    print(f"  Best test accuracy: {best_test_acc:.4f}")
    print(f"  Final epoch: {epoch}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Saved to: {seed_dir}")
    print(f"{'='*70}\n")

    return history


def main():
    parser = argparse.ArgumentParser(
        description='IMA Phase 0: Train PCN on CIFAR-10')
    parser.add_argument('--seed', type=int, required=True,
                        help=f'Random seed (pre-registered: {VALID_SEEDS})')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: 'cuda', 'cpu', or 'auto' (default)")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='CIFAR-10 data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/phase0',
                        help='Output directory for models and logs')
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS,
                        help=f'Maximum training epochs (default: {MAX_EPOCHS})')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            print(f"VRAM: {vram / 1e9:.1f} GB")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
    else:
        device = args.device

    print(f"Device: {device}")

    # Check HSA_OVERRIDE_GFX_VERSION for ROCm
    if device == 'cuda' and 'AMD' in torch.cuda.get_device_name(0):
        hsa = os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')
        print(f"HSA_OVERRIDE_GFX_VERSION: {hsa}")

    train(args.seed, device, args.data_dir, args.output_dir, args.max_epochs)


if __name__ == '__main__':
    main()
