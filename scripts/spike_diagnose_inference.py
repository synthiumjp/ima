"""Diagnostic: is the inference loop actually moving latents?"""
import torch
torch.backends.cudnn.enabled = False

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

# Import TinyConvPCN from the spike script
import importlib.util
spec = importlib.util.spec_from_file_location("spike_dynamics", "scripts/spike_dynamics.py")
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)

import torch.nn.functional as F

device = "cuda"
torch.manual_seed(42)

model = spike.TinyConvPCN(num_classes=10).to(device)
model.train()

# Dummy batch
x = torch.randn(8, 3, 32, 32, device=device)
y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
y_onehot = F.one_hot(y, 10).float()

# Phase 1: init
with torch.no_grad():
    latents_init = model.forward_encoder(x)

h1_init = latents_init[0].clone()
h2_init = latents_init[1].clone()
h3_init = latents_init[2].clone()

# Phase 2: inference loop (copy of spike's logic)
h1 = latents_init[0].clone().detach().requires_grad_(True)
h2 = latents_init[1].clone().detach().requires_grad_(True)
h3 = latents_init[2].clone().detach().requires_grad_(True)
h4_clamped = y_onehot

T = 13
eta_h = 5e-2
momentum_h = 0.5
m1 = torch.zeros_like(h1)
m2 = torch.zeros_like(h2)
m3 = torch.zeros_like(h3)

for t in range(T):
    latents = [h1, h2, h3, h4_clamped]
    per_sample_total, _, _ = model.total_energy(latents, y_onehot=y_onehot)
    total_energy_scalar = per_sample_total.sum()

    grads = torch.autograd.grad(
        total_energy_scalar,
        [h1, h2, h3],
        create_graph=False,
        retain_graph=False,
    )

    # Report grad magnitudes at this step
    g1_norm = grads[0].abs().mean().item()
    g2_norm = grads[1].abs().mean().item()
    g3_norm = grads[2].abs().mean().item()

    with torch.no_grad():
        m1 = momentum_h * m1 + grads[0]
        m2 = momentum_h * m2 + grads[1]
        m3 = momentum_h * m3 + grads[2]
        h1 -= eta_h * m1
        h2 -= eta_h * m2
        h3 -= eta_h * m3
    h1.requires_grad_(True)
    h2.requires_grad_(True)
    h3.requires_grad_(True)

    # How far have latents moved from init?
    d1 = (h1.detach() - h1_init).abs().mean().item()
    d2 = (h2.detach() - h2_init).abs().mean().item()
    d3 = (h3.detach() - h3_init).abs().mean().item()

    if t in [0, 1, 5, 12]:
        print(f"  step {t:2d}: grad |L1|={g1_norm:.6f} |L2|={g2_norm:.6f} |L3|={g3_norm:.6f}")
        print(f"           move |dL1|={d1:.6f} |dL2|={d2:.6f} |dL3|={d3:.6f}")
        print(f"           energy={total_energy_scalar.item():.4f}")

# Final comparison
print()
print(f"Final MSE between settled and init:")
print(f"  L1: {F.mse_loss(h1.detach(), h1_init).item():.8f}")
print(f"  L2: {F.mse_loss(h2.detach(), h2_init).item():.8f}")
print(f"  L3: {F.mse_loss(h3.detach(), h3_init).item():.8f}")
