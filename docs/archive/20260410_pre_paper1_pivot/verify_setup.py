"""
IMA Project Setup and GPU Verification
========================================

Run this FIRST after cloning to D:\\ima\\ to verify:
  1. Python environment
  2. PyTorch installation
  3. GPU detection (AMD RX 7900 GRE via ROCm)
  4. ConvPCN architecture (shapes, param count)
  5. Basic forward/backward pass on GPU
  6. CIFAR-10 download

Usage:
  set HSA_OVERRIDE_GFX_VERSION=11.0.0
  python scripts/verify_setup.py
"""

import sys
import time
import os
from pathlib import Path


def check_python():
    print(f"Python: {sys.version}")
    print(f"Path: {sys.executable}")
    assert sys.version_info >= (3, 9), "Need Python 3.9+"
    print("✓ Python OK\n")


def check_pytorch():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda or 'ROCm'}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            print(f"    VRAM: {vram / 1e9:.1f} GB")
    
    hsa = os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')
    print(f"HSA_OVERRIDE_GFX_VERSION: {hsa}")
    if not torch.cuda.is_available():
        print("\n⚠ No GPU detected. Check:")
        print("  1. ROCm is installed")
        print("  2. HSA_OVERRIDE_GFX_VERSION=11.0.0 is set")
        print("  3. PyTorch ROCm version is installed")
    print("✓ PyTorch OK\n")
    return torch.cuda.is_available()


def check_dependencies():
    deps = {
        'numpy': None,
        'sklearn': 'scikit-learn',
        'tqdm': None,
    }
    for module, pip_name in deps.items():
        try:
            __import__(module)
            print(f"  {module}: OK")
        except ImportError:
            print(f"  {module}: MISSING — pip install {pip_name or module}")

    # Check our custom CIFAR-10 loader
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
        from cifar10_data import get_data_loaders as _
        print(f"  cifar10_data: OK")
    except ImportError:
        print(f"  cifar10_data: MISSING — check src/cifar10_data.py")
    
    # Optional
    try:
        import metadpy
        print(f"  metadpy: OK")
    except ImportError:
        print(f"  metadpy: MISSING (optional, for M-ratio) — pip install metadpy")
    
    print("✓ Dependencies checked\n")


def check_architecture(device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from conv_pcn import ConvPCN, M_INPUT_DIM, LAYER_SHAPES

    model = ConvPCN(num_classes=10).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"ConvPCN parameters: {total:,}")
    assert 3_500_000 < total < 5_000_000, f"Expected ~4.2M, got {total:,}"

    # Forward pass
    x = torch.randn(4, 3, 32, 32, device=device)
    result = model.classify(x, T_infer=5, return_energy_trace=True,
                            return_errors=True)
    
    assert result.logits.shape == (4, 10), f"Logits: {result.logits.shape}"
    assert result.m_input.shape == (4, M_INPUT_DIM), f"M input: {result.m_input.shape}"
    assert len(result.energy_trace) == 6, f"Energy trace: {len(result.energy_trace)}"
    
    # Check energy is monotonically decreasing
    mono = all(result.energy_trace[i] >= result.energy_trace[i+1] 
               for i in range(len(result.energy_trace)-1))
    print(f"Energy monotonic (T=5): {mono}")
    
    print("✓ Architecture OK\n")
    return model


def check_gpu_training(model, device):
    """Quick training step to verify GPU compute works end-to-end."""
    import torch
    import torch.nn.functional as F
    from conv_pcn import train_pcn_epoch

    if device == 'cpu':
        print("Skipping GPU training test (CPU mode)\n")
        return

    print("Running GPU training test (1 batch)...")
    x = torch.randn(16, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (16,), device=device)
    
    # Manual single-batch training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    y_onehot = F.one_hot(y, 10).float()
    
    latents = model.init_latents_amortised(x)
    latents, _, _ = model.infer(x, y_onehot=y_onehot, latents=latents, T=5, eta=0.1)
    
    optimizer.zero_grad()
    errors, _ = model.compute_errors(x, [l.detach() for l in latents])
    logits = model.readout_layer(latents[3].detach())
    loss = 0.5 * sum(e.pow(2).sum() for e in errors) / 16 + F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    print("✓ GPU training OK\n")


def check_cifar10(data_dir: str = './data'):
    """Download CIFAR-10 if not present."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from cifar10_data import get_data_loaders

    print("Checking CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(data_dir)

    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Test: {len(test_loader.dataset):,} samples")

    # Verify batch shape
    x, y = next(iter(test_loader))
    assert x.shape[1:] == (3, 32, 32), f"Bad shape: {x.shape}"
    print(f"  Batch shape: {x.shape}")
    print("✓ CIFAR-10 OK\n")


def estimate_training_time(device):
    """Estimate Phase 0 training time from a timed batch."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    from conv_pcn import ConvPCN
    import torch.nn.functional as F
    
    model = ConvPCN(num_classes=10).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warm up
    x = torch.randn(128, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (128,), device=device)
    y_oh = F.one_hot(y, 10).float()
    
    latents = model.init_latents_amortised(x)
    latents, _, _ = model.infer(x, y_onehot=y_oh, latents=latents, T=20, eta=0.1)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time 5 batches
    times = []
    for _ in range(5):
        start = time.time()
        latents = model.init_latents_amortised(x)
        latents, _, _ = model.infer(x, y_onehot=y_oh, latents=latents, T=20, eta=0.1)
        
        optimizer.zero_grad()
        errors, _ = model.compute_errors(x, [l.detach() for l in latents])
        logits = model.readout_layer(latents[3].detach())
        loss = 0.5 * sum(e.pow(2).sum() for e in errors) / 128 + F.cross_entropy(logits, y)
        
        # Encoder loss — forward with gradients
        h = x
        enc_outputs = []
        for layer in model._encoder_layers[:7]:
            h = layer(h)
        enc_outputs.append(h)
        for layer in model._encoder_layers[7:14]:
            h = layer(h)
        enc_outputs.append(h)
        for layer in model._encoder_layers[14:21]:
            h = layer(h)
        enc_outputs.append(h)
        for layer in model._encoder_layers[21:]:
            h = layer(h)
        h = model.fc_encoder(h)
        enc_outputs.append(h)
        enc_loss = sum(F.mse_loss(enc_l, l.detach()) for enc_l, l in zip(enc_outputs, latents))
        
        total = loss + enc_loss
        total.backward()
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    mean_batch = sum(times) / len(times)
    batches_per_epoch = 50000 / 128  # ~391
    epoch_time = mean_batch * batches_per_epoch
    
    print(f"Estimated training time:")
    print(f"  Batch (128 samples, T=20): {mean_batch:.2f}s")
    print(f"  Epoch: {epoch_time/60:.1f} min")
    print(f"  50 epochs: {50*epoch_time/3600:.1f} hours")
    print(f"  100 epochs: {100*epoch_time/3600:.1f} hours")
    print(f"  Phase 0 (3 seeds × ~75 epochs): {3*75*epoch_time/3600:.1f} hours")
    if device == 'cuda':
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print()


def main():
    print("=" * 70)
    print("IMA Project Setup Verification")
    print("=" * 70)
    print()

    check_python()
    gpu_available = check_pytorch()
    check_dependencies()
    
    device = 'cuda' if gpu_available else 'cpu'
    print(f"Using device: {device}\n")
    
    model = check_architecture(device)
    check_gpu_training(model, device)
    check_cifar10()
    
    if gpu_available:
        estimate_training_time(device)
    
    print("=" * 70)
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    print(f"\nReady to train. Run:")
    print(f"  python scripts/train_phase0.py --seed 42")
    print(f"\nFor all seeds:")
    print(f"  python scripts/train_phase0.py --seed 42")
    print(f"  python scripts/train_phase0.py --seed 123")
    print(f"  python scripts/train_phase0.py --seed 456")


if __name__ == '__main__':
    main()
