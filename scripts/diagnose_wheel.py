"""
AMD ROCm wheel diagnostic — verify that the AMD ROCm PyTorch wheel
handles our class of workload (Conv2d + BatchNorm + forward + backward)
on native Windows.

IMPORTANT: This wheel (torch 2.8.0a0+gitfc14c65 / ROCm 6.4.4) has a
known MIOpen SQLite schema bug affecting BatchNorm training on RDNA3+
GPUs (ROCm/ROCm#5441, closed without fix). The workaround is to
disable cuDNN (MIOpen) globally before any BN ops run. This routes
BatchNorm through native HIP kernels. Slower than MIOpen would be, but
functional.

This workaround must be set at import time, before any modules that
use cudnn are touched.
"""
# --- CUDNN WORKAROUND: must be set before any BN ops ---
import torch
torch.backends.cudnn.enabled = False
# -------------------------------------------------------

import time
import torch.nn as nn
import torch.nn.functional as F


def banner(msg):
    print()
    print("=" * 60)
    print(msg)
    print("=" * 60)


def main():
    banner("1. Basic torch + GPU info")
    print(f"torch version:    {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "GPU not available — aborting"
    print(f"Device name:      {torch.cuda.get_device_name(0)}")
    print(f"Device count:     {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total memory:     {props.total_memory / 1e9:.2f} GB")
    print(f"cuDNN enabled:    {torch.backends.cudnn.enabled}  (disabled to work around MIOpen BN bug)")

    banner("2. Simple tensor ops on GPU")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"1000x1000 matmul: {tuple(y.shape)} on {y.device}  OK")

    banner("3. Conv2d forward pass")
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
    x = torch.randn(8, 3, 32, 32, device="cuda")
    out = conv(x)
    torch.cuda.synchronize()
    print(f"Conv2d(3,64,k3):  in {tuple(x.shape)} -> out {tuple(out.shape)}  OK")

    banner("4. BatchNorm2d forward pass (via HIP fallback)")
    bn = nn.BatchNorm2d(64).cuda()
    out_bn = bn(out)
    torch.cuda.synchronize()
    print(f"BatchNorm2d(64):  in {tuple(out.shape)} -> out {tuple(out_bn.shape)}  OK")

    banner("5. Conv + BN + GELU + backward pass")
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params:     {n_params:,}")

    x = torch.randn(128, 3, 32, 32, device="cuda")
    y_target = torch.randint(0, 10, (128,), device="cuda")
    logits = model(x)
    loss = F.cross_entropy(logits, y_target)
    print(f"Forward OK:       logits {tuple(logits.shape)}, loss {loss.item():.4f}")
    loss.backward()
    torch.cuda.synchronize()
    print("Backward OK:      gradients computed")
    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert any_grad, "No gradients computed"
    print("Gradient check:   non-zero gradients present  OK")

    banner("6. Timing: full conv forward + backward (VGG-ish)")
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(256),
        nn.GELU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(256),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(3):
        x = torch.randn(128, 3, 32, 32, device="cuda")
        y_target = torch.randint(0, 10, (128,), device="cuda")
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Time
    n_iters = 20
    start = time.time()
    for _ in range(n_iters):
        x = torch.randn(128, 3, 32, 32, device="cuda")
        y_target = torch.randint(0, 10, (128,), device="cuda")
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{n_iters} training steps: {elapsed:.2f}s  ({elapsed/n_iters*1000:.1f} ms/step)")

    banner("7. Memory usage")
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated:        {mem_alloc:.2f} GB")
    print(f"Reserved:         {mem_reserved:.2f} GB")

    banner("All checks passed")


if __name__ == "__main__":
    main()
