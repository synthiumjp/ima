"""
TinyConvPCN-MSE: Standard discriminative PC with MSE at output
===============================================================

Condition B in the bPC pre-registration.

Identical to TinyConvPCN (spike_dynamics.py) except:
  - Output energy uses (1/2) ‖h4 - y_onehot‖² instead of CE(softmax(h4), y)
  - Everything else unchanged: architecture, T=13, [0,1] normalisation,
    unidirectional inference dynamics

This isolates IMA assumption A1 (CE at output) while preserving A3
(effectively feedforward dynamics) and A2 (target clamping).

Author: JP Cacioli
Date: April 2026
Pre-registration: OSF [pending]
"""
# --- CUDNN WORKAROUND ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class TinyConvPCN_MSE(nn.Module):
    """Standard discriminative PC with MSE at output instead of CE.
    
    Architecture identical to TinyConvPCN. Only total_energy differs.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder (identical to TinyConvPCN)
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.enc_fc2 = nn.Linear(256, num_classes)
        
        # Generative weights (identical to TinyConvPCN)
        self.gen_fc3 = nn.Linear(num_classes, 256)
        self.gen_fc2 = nn.Linear(256, 64 * 8 * 8)
        self.gen_conv1 = nn.Conv2d(64, 32, 3, padding=1)
    
    def forward_encoder(self, x):
        """Amortised forward pass (identical to TinyConvPCN)."""
        h1 = F.gelu(self.enc_bn1(self.enc_conv1(x)))
        h1 = F.max_pool2d(h1, 2)
        h2 = F.gelu(self.enc_bn2(self.enc_conv2(h1)))
        h2 = F.max_pool2d(h2, 2)
        h2_flat = h2.view(x.size(0), -1)
        h3 = F.gelu(self.enc_fc1(h2_flat))
        h4 = self.enc_fc2(h3)
        return [h1, h2, h3, h4]
    
    def generative_predictions(self, latents):
        """Top-down predictions (identical to TinyConvPCN)."""
        h1, h2, h3, h4 = latents
        B = h1.size(0)
        mu_3 = self.gen_fc3(h4)
        mu_2_flat = self.gen_fc2(h3)
        mu_2 = mu_2_flat.view(B, 64, 8, 8)
        h2_up = F.interpolate(h2, scale_factor=2, mode='nearest')
        mu_1 = self.gen_conv1(h2_up)
        return [mu_1, mu_2, mu_3]
    
    def compute_errors(self, latents):
        """Prediction errors (identical to TinyConvPCN)."""
        mus = self.generative_predictions(latents)
        errors = [latents[l] - mus[l] for l in range(3)]
        return errors
    
    def total_energy(self, latents, y_onehot=None):
        """Total energy with MSE at output instead of CE.
        
        E = sum_l 0.5 * mean(eps_l^2)  (same as TinyConvPCN)
            + 0.5 * mean((h4 - y_onehot)^2)  (MSE replaces CE)
        
        This is the ONLY difference from TinyConvPCN.
        """
        B = latents[0].size(0)
        errors = self.compute_errors(latents)
        
        # Layer-wise prediction errors (identical to TinyConvPCN)
        per_sample_lat_e = torch.zeros(B, device=latents[0].device)
        for e in errors:
            e_flat = e.view(B, -1)
            per_sample_lat_e = per_sample_lat_e + 0.5 * e_flat.pow(2).mean(dim=1)
        
        # MSE at output (DIFFERENT from TinyConvPCN which uses CE)
        if y_onehot is not None:
            mse = 0.5 * (latents[3] - y_onehot).pow(2).mean(dim=1)  # (B,)
            per_sample_sup_e = mse
        else:
            per_sample_sup_e = torch.zeros(B, device=latents[0].device)
        
        per_sample_total = per_sample_lat_e + per_sample_sup_e
        return per_sample_total, per_sample_lat_e, per_sample_sup_e


def train_step_mse(model, x, y, T=13, eta_h=5e-2, momentum_h=0.5):
    """Training step for TinyConvPCN-MSE.
    
    Identical to train_step in spike_dynamics.py except:
      - Uses MSE energy at output (via model.total_energy)
      - Readout loss uses MSE instead of CE
    
    Everything else (inference dynamics, encoder alignment, T, momentum)
    is identical to standard discPC.
    """
    B = x.size(0)
    num_classes = model.num_classes
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    
    # Phase 1: amortised init
    with torch.no_grad():
        latents_init = model.forward_encoder(x)
    
    h1 = latents_init[0].clone().detach().requires_grad_(True)
    h2 = latents_init[1].clone().detach().requires_grad_(True)
    h3 = latents_init[2].clone().detach().requires_grad_(True)
    h4_clamped = y_onehot.clone().detach()
    
    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    m3 = torch.zeros_like(h3)
    
    # Phase 2: inference loop (identical dynamics to standard PC)
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
    
    # Phase 3: weight update
    latents_enc = model.forward_encoder(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc
    
    # Encoder alignment (identical to standard PC)
    enc_loss = (
        F.mse_loss(h1_enc, h1.detach())
        + F.mse_loss(h2_enc, h2.detach())
        + F.mse_loss(h3_enc, h3.detach())
    )
    
    # Energy at settled state
    latents_for_loss = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
    per_sample_total, per_sample_lat, per_sample_sup = model.total_energy(
        latents_for_loss, y_onehot=y_onehot
    )
    gen_loss = per_sample_total.mean()
    
    # Readout loss: MSE instead of CE (the key difference)
    readout_loss = 0.5 * F.mse_loss(h4_enc, y_onehot)
    
    total_loss = gen_loss + enc_loss + readout_loss
    return total_loss, gen_loss.item(), enc_loss.item(), readout_loss.item()


# =============================================================================
# Quick verification
# =============================================================================

if __name__ == '__main__':
    print("TinyConvPCN-MSE Architecture Test")
    print("=" * 60)
    
    model = TinyConvPCN_MSE(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected:         2,144,938")
    print(f"Match: {'YES' if total_params == 2_144_938 else 'NO'}")
    
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    y_onehot = F.one_hot(y, 10).float()
    
    # Test energy
    latents = model.forward_encoder(x)
    total_e, lat_e, sup_e = model.total_energy(latents, y_onehot)
    print(f"\nEnergy test:")
    print(f"  Total: {total_e.mean():.6f}")
    print(f"  Latent: {lat_e.mean():.6f}")
    print(f"  Sup (MSE): {sup_e.mean():.6f}")
    
    # Verify sup energy is MSE not CE
    manual_mse = 0.5 * (latents[3] - y_onehot).pow(2).mean(dim=1)
    print(f"  Manual MSE: {manual_mse.mean():.6f}")
    print(f"  Match: {'YES' if torch.allclose(sup_e, manual_mse) else 'NO'}")
    
    # Test training step
    print(f"\nTraining step test (T=3):")
    model.train()
    loss, gen_l, enc_l, read_l = train_step_mse(model, x, y, T=3)
    print(f"  Total: {loss.item():.6f}")
    print(f"  Gen: {gen_l:.6f}")
    print(f"  Enc: {enc_l:.6f}")
    print(f"  Readout (MSE): {read_l:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
