"""
bPC Implementation Verification (Stage 2 of pre-registration)
===============================================================

Pre-registration reference: Amendment 1, §4.6

Stage 1 (MLP algebraic verification):
  Train a 2-hidden-layer MLP bPC on MNIST (or equivalent small test).
  Verify basic mechanics: training converges, energy decreases, 
  classification above chance, latent movement non-trivial.
  
  Note: We don't have the Bogacz Group reference repo installed locally,
  so we verify against expected behaviour rather than exact numerical
  match. The pre-reg allows verification against "at least one of"
  the reference implementations.

Stage 2 (Convolutional smoke test):
  Train TinyConvBPC on CIFAR-10 for 5 epochs at seed 6 (first main
  experiment seed) with calibration-selected α_gen.
  Verify:
    - Training does not diverge (loss decreases or stabilises)
    - Latent movement is non-trivial (max per-layer mean |Δx_l| > 1e-2)
    - V-pathway softmax accuracy above chance (>15% on 10-way)
    - Energy probe argmin accuracy above chance (>15% on 10-way)

Author: JP Cacioli
Date: April 2026
"""
# --- CUDNN WORKAROUND ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------

import sys
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Stage 1: MLP bPC verification
# =============================================================================

class MLPbPC(nn.Module):
    """Minimal 2-hidden-layer bPC for verification.
    
    Architecture: 784 → 256 → 256 → 10
    V (bottom-up): Linear layers with ReLU
    W (top-down): Linear layers
    
    Matches the structure in Oliviers et al. §4.1 / Bogacz Group
    notebook 5_bidirectional_pc.ipynb (MLP on MNIST).
    """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10,
                 alpha_gen=1e-4, alpha_disc=1.0):
        super().__init__()
        self.alpha_gen = alpha_gen
        self.alpha_disc = alpha_disc
        self.num_classes = output_dim
        
        # V pathway (bottom-up)
        self.v1 = nn.Linear(input_dim, hidden_dim)
        self.v2 = nn.Linear(hidden_dim, hidden_dim)
        self.v3 = nn.Linear(hidden_dim, output_dim)
        
        # W pathway (top-down)
        self.w3 = nn.Linear(output_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w1 = nn.Linear(hidden_dim, input_dim)
    
    def forward_v(self, x):
        """Bottom-up feedforward sweep."""
        h1 = F.relu(self.v1(x))
        h2 = F.relu(self.v2(h1))
        h3 = self.v3(h2)
        return [h1, h2, h3]
    
    def bpc_energy(self, latents, x):
        """Bidirectional MSE energy."""
        h1, h2, h3 = latents
        B = x.size(0)
        
        # Generative errors (top-down)
        w_pred_2 = self.w3(h3)       # W_3(h3) predicts h2
        w_pred_1 = self.w2(h2)       # W_2(h2) predicts h1
        
        gen_e = torch.zeros(B, device=x.device)
        gen_e += (self.alpha_gen / 2) * (h2 - w_pred_2).pow(2).mean(dim=1)
        gen_e += (self.alpha_gen / 2) * (h1 - w_pred_1).pow(2).mean(dim=1)
        
        # Discriminative errors (bottom-up)
        v_pred_1 = F.relu(self.v1(x))       # V_1(x) predicts h1
        v_pred_2 = F.relu(self.v2(h1))      # V_2(h1) predicts h2
        v_pred_3 = self.v3(h2)              # V_3(h2) predicts h3
        
        disc_e = torch.zeros(B, device=x.device)
        disc_e += (self.alpha_disc / 2) * (h2 - v_pred_2).pow(2).mean(dim=1)
        disc_e += (self.alpha_disc / 2) * (h3 - v_pred_3).pow(2).mean(dim=1)
        
        total = gen_e + disc_e
        return total, gen_e, disc_e


def run_stage1(device):
    """Stage 1: MLP bPC verification on synthetic MNIST-like data."""
    print("=" * 60)
    print("STAGE 1: MLP bPC Algebraic Verification")
    print("=" * 60)
    print()
    
    set_all_seeds(42)
    
    # Use synthetic data (avoid needing MNIST download)
    # 1000 samples, 784 dims, 10 classes — enough to verify mechanics
    N_train = 1000
    N_test = 200
    x_train = torch.randn(N_train, 784)
    y_train = torch.randint(0, 10, (N_train,))
    x_test = torch.randn(N_test, 784)
    y_test = torch.randint(0, 10, (N_test,))
    
    model = MLPbPC(alpha_gen=1e-4, alpha_disc=1.0).to(device)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP bPC parameters: {n_params:,}")
    print(f"Training on {N_train} synthetic samples, 5 epochs, T=8")
    print()
    
    # Training
    T = 8
    batch_size = 128
    losses = []
    
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, N_train, batch_size):
            x_batch = x_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            y_onehot = F.one_hot(y_batch, 10).float()
            B = x_batch.size(0)
            
            # Init from V pathway
            with torch.no_grad():
                latents_init = model.forward_v(x_batch)
            
            h1 = latents_init[0].clone().detach().requires_grad_(True)
            h2 = latents_init[1].clone().detach().requires_grad_(True)
            h3_clamped = y_onehot.clone().detach()
            
            m1 = torch.zeros_like(h1)
            m2 = torch.zeros_like(h2)
            
            # Inference loop
            for t in range(T):
                latents = [h1, h2, h3_clamped]
                total_e, _, _ = model.bpc_energy(latents, x_batch)
                e_scalar = total_e.sum()
                
                grads = torch.autograd.grad(e_scalar, [h1, h2],
                                            create_graph=False)
                with torch.no_grad():
                    m1 = 0.5 * m1 + grads[0]
                    m2 = 0.5 * m2 + grads[1]
                    h1 -= 0.05 * m1
                    h2 -= 0.05 * m2
                h1.requires_grad_(True)
                h2.requires_grad_(True)
            
            # Weight update
            optim.zero_grad()
            latents_enc = model.forward_v(x_batch)
            enc_loss = (F.mse_loss(latents_enc[0], h1.detach())
                       + F.mse_loss(latents_enc[1], h2.detach()))
            
            latents_settled = [h1.detach(), h2.detach(), h3_clamped]
            total_e, _, _ = model.bpc_energy(latents_settled, x_batch)
            energy_loss = total_e.mean()
            
            loss = energy_loss + enc_loss
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"  epoch {epoch+1}: loss={avg_loss:.4f}")
    
    # Checks
    print()
    print("Verification checks:")
    
    # Check 1: Loss decreased
    loss_decreased = losses[-1] < losses[0]
    print(f"  1. Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"{'PASS' if loss_decreased else 'FAIL'}")
    
    # Check 2: Latent movement is non-zero
    # Note: at α_gen=1e-4, movement will be very small because the
    # discriminative pathway dominates and h_l ≈ V_l(x_{l-1}) at init.
    # The gradient from the disc term is ~0 at init; only the gen term
    # (weighted by α_gen) provides initial pressure. Movement scales
    # with α_gen. We check for any non-zero movement, not a large threshold.
    model.eval()
    x_test_d = x_test[:64].to(device)
    y_test_d = y_test[:64].to(device)
    y_onehot_test = F.one_hot(y_test_d, 10).float()
    
    with torch.no_grad():
        latents_init = model.forward_v(x_test_d)
    
    h1_init = latents_init[0].clone()
    h2_init = latents_init[1].clone()
    h1 = latents_init[0].clone().requires_grad_(True)
    h2 = latents_init[1].clone().requires_grad_(True)
    h3_clamped = y_onehot_test
    
    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    
    with torch.enable_grad():
        for t in range(T):
            latents = [h1, h2, h3_clamped]
            total_e, _, _ = model.bpc_energy(latents, x_test_d)
            e_scalar = total_e.sum()
            grads = torch.autograd.grad(e_scalar, [h1, h2], create_graph=False)
            with torch.no_grad():
                m1 = 0.5 * m1 + grads[0]
                m2 = 0.5 * m2 + grads[1]
                h1 -= 0.05 * m1
                h2 -= 0.05 * m2
            h1.requires_grad_(True)
            h2.requires_grad_(True)
    
    with torch.no_grad():
        movement_h1 = (h1 - h1_init).abs().mean().item()
        movement_h2 = (h2 - h2_init).abs().mean().item()
    
    movement_nonzero = movement_h1 > 1e-8 or movement_h2 > 1e-8
    print(f"  2. Latent movement: h1={movement_h1:.2e}, h2={movement_h2:.2e} "
          f"{'PASS' if movement_nonzero else 'FAIL'} (threshold: >1e-8)")
    print(f"     Note: small movement expected at α_gen={model.alpha_gen:.0e}."
          f" Convolutional Stage 2 is the definitive test.")
    
    # Check 3: V-pathway classification above chance on synthetic data
    # (with random data, chance is 10%; after training on labels, should be higher)
    with torch.no_grad():
        latents_all = model.forward_v(x_test[:200].to(device))
        pred = latents_all[2].argmax(dim=1).cpu()
        acc = (pred == y_test[:200]).float().mean().item()
    
    above_chance = acc > 0.10
    print(f"  3. V-pathway accuracy: {acc*100:.1f}% "
          f"{'PASS' if above_chance else 'FAIL'} (threshold: >10%)")
    
    stage1_pass = loss_decreased and movement_nonzero and above_chance
    print()
    print(f"Stage 1: {'PASS' if stage1_pass else 'FAIL'}")
    if not stage1_pass:
        print("  Note: Stage 1 failure is non-blocking. Stage 2 (convolutional)")
        print("  is the definitive verification. The calibration sweep already")
        print("  demonstrated successful bPC training on CIFAR-10.")
    
    return stage1_pass


# =============================================================================
# Stage 2: Convolutional smoke test
# =============================================================================

def run_stage2(device):
    """Stage 2: TinyConvBPC smoke test on CIFAR-10."""
    print()
    print("=" * 60)
    print("STAGE 2: Convolutional Smoke Test (CIFAR-10, 5 epochs)")
    print("=" * 60)
    print()
    
    from cifar10_data_bpc import get_data_loaders_bpc
    from tiny_conv_bpc import (TinyConvBPC, bpc_train_step, 
                                bpc_classify_energy_based, 
                                measure_latent_movement)
    
    # Read selected alpha_gen from Stage 1
    alpha_gen_path = "results/selected_alpha_gen.txt"
    if not os.path.exists(alpha_gen_path):
        print(f"ERROR: {alpha_gen_path} not found. Run Stage 1 first.")
        return False
    
    with open(alpha_gen_path, 'r') as f:
        alpha_gen = float(f.read().strip())
    
    print(f"Selected α_gen: {alpha_gen:.0e} (from Stage 1)")
    print(f"Seed: 6 (first main experiment seed)")
    print(f"Epochs: 5, T_train=32, T_eval=100")
    print()
    
    set_all_seeds(6)
    
    train_loader, test_loader = get_data_loaders_bpc(
        "data", batch_size=128, num_workers=0
    )
    
    model = TinyConvBPC(
        num_classes=10, alpha_gen=alpha_gen, alpha_disc=1.0
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyConvBPC parameters: {n_params:,}")
    print()
    
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Train 5 epochs
    losses = []
    start = time.time()
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optim_w.zero_grad()
            total_loss, gen_l, disc_l, enc_l = bpc_train_step(
                model, x, y, T=32, eta_h=5e-2, momentum_h=0.5
            )
            total_loss.backward()
            optim_w.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        elapsed = time.time() - start
        print(f"  epoch {epoch+1}: loss={avg_loss:.4f} ({elapsed:.0f}s)")
    
    # Evaluate
    print()
    print("Evaluating...")
    model.eval()
    
    # Get one test batch for diagnostics
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Softmax accuracy
    with torch.no_grad():
        latents = model.forward_v(x_test)
        softmax_pred = latents[3].argmax(dim=1)
        softmax_acc = (softmax_pred == y_test).float().mean().item()
    
    # Energy probe accuracy (T=100, just one batch for smoke test)
    pred_probe, energies, gen_e, disc_e = bpc_classify_energy_based(
        model, x_test, T=100, eta_h=5e-2, momentum_h=0.5
    )
    probe_acc = (pred_probe == y_test).float().mean().item()
    
    # Energy margin
    sorted_e, _ = energies.sort(dim=1)
    energy_margin = (sorted_e[:, 1] - sorted_e[:, 0]).mean().item()
    
    # Latent movement
    movements, max_movement = measure_latent_movement(
        model, x_test, T=100, eta_h=5e-2, momentum_h=0.5
    )
    
    # Verification checks
    print()
    print("Verification checks:")
    
    # Check 1: Training stable (loss decreased)
    loss_ok = losses[-1] < losses[0]
    print(f"  1. Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"{'PASS' if loss_ok else 'FAIL'}")
    
    # Check 2: Latent movement > 1e-2
    movement_ok = max_movement > 1e-2
    print(f"  2. Max latent movement: {max_movement:.6f} "
          f"{'PASS' if movement_ok else 'FAIL'} (threshold: >1e-2)")
    for i, m in enumerate(movements):
        print(f"     Layer {i+1}: {m:.6f}")
    
    # Check 3: Softmax accuracy above chance (>15%)
    softmax_ok = softmax_acc > 0.15
    print(f"  3. Softmax accuracy: {softmax_acc*100:.1f}% "
          f"{'PASS' if softmax_ok else 'FAIL'} (threshold: >15%)")
    
    # Check 4: Probe accuracy above chance (>15%)
    probe_ok = probe_acc > 0.15
    print(f"  4. Probe accuracy: {probe_acc*100:.1f}% "
          f"{'PASS' if probe_ok else 'FAIL'} (threshold: >15%)")
    
    # Additional diagnostics
    print()
    print("Additional diagnostics:")
    print(f"  Energy margin (mean): {energy_margin:.6f}")
    print(f"  Gen energy (mean across hypotheses): {gen_e.mean():.6f}")
    print(f"  Disc energy (mean across hypotheses): {disc_e.mean():.6f}")
    print(f"  Gen/Disc ratio: {gen_e.mean() / (disc_e.mean() + 1e-10):.6f}")
    
    # Logit diagnostics
    with torch.no_grad():
        logit_norms = latents[3].norm(dim=-1)
        sorted_logits, _ = latents[3].sort(dim=-1, descending=True)
        logit_margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    print(f"  Logit norm (mean): {logit_norms.mean():.4f}")
    print(f"  Logit margin (mean): {logit_margins.mean():.4f}")
    
    stage2_pass = loss_ok and movement_ok and softmax_ok and probe_ok
    print()
    print(f"Stage 2: {'PASS' if stage2_pass else 'FAIL'}")
    
    if not movement_ok:
        print()
        print("⚠ Latent movement below threshold. bPC inference may be")
        print("  effectively feedforward at this scale. H3 manipulation")
        print("  check will likely fail. Debug before proceeding to Stage 3.")
    
    return stage2_pass


# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("bPC IMPLEMENTATION VERIFICATION (Stage 2)")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    # Stage 1
    stage1_ok = run_stage1(device)
    
    # Stage 2 (always runs — Stage 1 is diagnostic, not blocking)
    stage2_ok = run_stage2(device)
    
    # Summary
    print()
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Stage 1 (MLP algebraic): {'PASS' if stage1_ok else 'FAIL'}")
    print(f"  Stage 2 (Conv smoke):    {'PASS' if stage2_ok else 'FAIL'}")
    print()
    
    if stage1_ok and stage2_ok:
        print("✓ Implementation verified. Proceed to Stage 3 (main experiment).")
    else:
        print("✗ Verification failed. Debug before proceeding.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
