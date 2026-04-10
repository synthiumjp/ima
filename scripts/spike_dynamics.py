"""
IMA dynamics mechanism spike (11 Apr 2026)

Goal: verify that energy-based K-way discriminative inference at test time
works on a trained convolutional PCN with Pinchetti-style CE energy at the
output layer, on CIFAR-10.

This is the "is the mechanism real?" test before committing to v6.

Weak success criteria:
  1. Per-hypothesis energies differ by more than measurement noise
     (relative spread > 1% per sample)
  2. Energy-based argmin classification beats chance (>10% on 10-way)
  3. Energy margin as confidence signal achieves AUROC2 > 0.55

If all three pass -> mechanism is viable, commit to v6.
If any fail -> report failure mode, reconsider.

Design choices:
  - Tiny ConvPCN (~150K params), not VGG5 — spike only
  - Separate optim_h (SGD momentum) and optim_w (AdamW), matching PCX
  - CE energy at output vode (the critical fix vs yesterday's attempts)
  - Target-clamped inference during training
  - Per-element mean energies (not sums) — matches CE scale
  - T=13 inference steps
  - 5 epochs, batch 128
  - Eval: K=10 inferences per test image, argmin residual energy
"""
# --- CUDNN WORKAROUND: required for AMD ROCm wheel BN bug (ROCm/ROCm#5441) ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------------------------------------------------------------

import sys
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, "src")
from cifar10_data import get_data_loaders


# =============================================================================
# Tiny ConvPCN
# =============================================================================
# Layer hierarchy:
#   L0: input (3, 32, 32)          — clamped to data
#   L1: after conv1+bn+gelu+pool   (32, 16, 16)  — latent vode
#   L2: after conv2+bn+gelu+pool   (64, 8, 8)    — latent vode
#   L3: after fc1+gelu             (256,)        — latent vode
#   L4: output logits              (10,)         — latent vode (ce_energy)
#
# Generative weights g_l map from latent above to latent below:
#   g_1 : L2 -> L1   (transposed conv + upsample)
#   g_2 : L3 -> L2   (linear + reshape)
#   g_3 : L4 -> L3   (linear)

class TinyConvPCN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Encoder (amortised init)
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.enc_fc2 = nn.Linear(256, num_classes)

        # Generative weights (for computing prediction errors)
        # g_3: L4 (10) -> L3 (256)
        self.gen_fc3 = nn.Linear(num_classes, 256)
        # g_2: L3 (256) -> L2 (64,8,8) flattened
        self.gen_fc2 = nn.Linear(256, 64 * 8 * 8)
        # g_1: L2 (64,16,16 after upsample) -> L1 (32,16,16)  [conv, no stride]
        self.gen_conv1 = nn.Conv2d(64, 32, 3, padding=1)

    def forward_encoder(self, x):
        """Amortised forward pass: produces initial values for latent vodes."""
        h1 = F.gelu(self.enc_bn1(self.enc_conv1(x)))  # (B, 32, 32, 32)
        h1 = F.max_pool2d(h1, 2)                      # (B, 32, 16, 16)
        h2 = F.gelu(self.enc_bn2(self.enc_conv2(h1))) # (B, 64, 16, 16)
        h2 = F.max_pool2d(h2, 2)                      # (B, 64, 8, 8)
        h2_flat = h2.view(x.size(0), -1)
        h3 = F.gelu(self.enc_fc1(h2_flat))            # (B, 256)
        h4 = self.enc_fc2(h3)                         # (B, 10) logits
        return [h1, h2, h3, h4]

    def generative_predictions(self, latents):
        """Compute g_l(h_{l+1}) for each layer l below L4.
        Returns predictions [mu_1, mu_2, mu_3] (no mu_0 because L0 is clamped).
        """
        h1, h2, h3, h4 = latents
        B = h1.size(0)

        # mu_3 = g_3(h4)  [L4 -> L3]
        mu_3 = self.gen_fc3(h4)  # (B, 256)

        # mu_2 = g_2(h3)  [L3 -> L2]
        mu_2_flat = self.gen_fc2(h3)  # (B, 64*8*8)
        mu_2 = mu_2_flat.view(B, 64, 8, 8)

        # mu_1 = g_1(h2)  [L2 -> L1]
        # L2 is (64, 8, 8), L1 is (32, 16, 16). Upsample L2 to (64, 16, 16), then conv -> (32, 16, 16)
        h2_up = F.interpolate(h2, scale_factor=2, mode='nearest')  # (B, 64, 16, 16)
        mu_1 = self.gen_conv1(h2_up)  # (B, 32, 16, 16)

        return [mu_1, mu_2, mu_3]

    def compute_errors(self, latents):
        """Compute prediction errors eps_l = h_l - g_l(h_{l+1}) for l in {1,2,3}."""
        mus = self.generative_predictions(latents)
        errors = [latents[l] - mus[l] for l in range(3)]  # eps_1, eps_2, eps_3
        return errors

    def total_energy(self, latents, y_onehot=None):
        """Compute total energy for a batch.
        
        E = sum_l 0.5 * mean(eps_l^2)  (per-element mean over latent dim)
            + CE(softmax(h4), y_onehot)   if y_onehot provided, else 0
        
        Returns per-sample energy (shape B) and scalar total.
        """
        B = latents[0].size(0)
        errors = self.compute_errors(latents)

        # Per-sample per-layer squared errors, averaged over non-batch dims
        per_sample_lat_e = torch.zeros(B, device=latents[0].device)
        for e in errors:
            e_flat = e.view(B, -1)
            per_sample_lat_e = per_sample_lat_e + 0.5 * e_flat.pow(2).mean(dim=1)

        # CE energy at output (if target provided)
        if y_onehot is not None:
            log_p = F.log_softmax(latents[3], dim=-1)
            ce = -(y_onehot * log_p).sum(dim=-1)  # (B,)
            per_sample_sup_e = ce
        else:
            per_sample_sup_e = torch.zeros(B, device=latents[0].device)

        per_sample_total = per_sample_lat_e + per_sample_sup_e
        return per_sample_total, per_sample_lat_e, per_sample_sup_e


# =============================================================================
# Training step (Pinchetti-style)
# =============================================================================

def train_step(model, x, y, T=13, eta_h=5e-2, momentum_h=0.5):
    """One training step with PCX-style separate inference + weight update.
    
    Pinchetti uses optim_h with SGD momentum nesterov. We implement manually
    because we need to manage the latents as non-leaf tensors.
    
    Phase 1: Amortised init (forward pass through encoder)
    Phase 2: T inference steps — update latents (except clamped output) by
             gradient of total energy. Output h4 is clamped to y_onehot.
    Phase 3: At settled configuration, compute total energy and backprop to
             weights (called by the caller's optim_w.step()).
    """
    B = x.size(0)
    num_classes = model.num_classes
    y_onehot = F.one_hot(y, num_classes=num_classes).float()

    # Phase 1: amortised init (no gradients through encoder for now;
    # we'll do encoder training via the final weight update backprop)
    with torch.no_grad():
        latents_init = model.forward_encoder(x)

    # Convert to leaf tensors for manual optimisation
    h1 = latents_init[0].clone().detach().requires_grad_(True)
    h2 = latents_init[1].clone().detach().requires_grad_(True)
    h3 = latents_init[2].clone().detach().requires_grad_(True)
    # h4 is clamped to y_onehot and frozen — this is the key fix
    h4_clamped = y_onehot.clone().detach()

    # Momentum buffers for SGD
    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    m3 = torch.zeros_like(h3)

    # Phase 2: inference loop
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
        # Nesterov-momentum-like update (simplified — no lookahead)
        with torch.no_grad():
            m1 = momentum_h * m1 + grads[0]
            m2 = momentum_h * m2 + grads[1]
            m3 = momentum_h * m3 + grads[2]
            h1 -= eta_h * m1
            h2 -= eta_h * m2
            h3 -= eta_h * m3
        # Re-enable grad tracking for next iteration
        h1.requires_grad_(True)
        h2.requires_grad_(True)
        h3.requires_grad_(True)

    # Phase 3: Compute loss for weight update.
    # At settled latent configuration, the total energy (with clamped target)
    # is the loss we want to minimize w.r.t. weights.
    # Re-run forward encoder WITH gradients so encoder weights get updated.
    latents_enc = model.forward_encoder(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc
    # Encoder alignment loss: encoder outputs should match settled latents
    enc_loss = (
        F.mse_loss(h1_enc, h1.detach())
        + F.mse_loss(h2_enc, h2.detach())
        + F.mse_loss(h3_enc, h3.detach())
    )

    # Energy at settled state, using current generative weights
    latents_for_loss = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
    per_sample_total, per_sample_lat, per_sample_sup = model.total_energy(
        latents_for_loss, y_onehot=y_onehot
    )
    gen_loss = per_sample_total.mean()

    # Train readout: the encoder's h4_enc should also match the clamped target
    # via CE (this trains the readout directly)
    readout_loss = F.cross_entropy(h4_enc, y)

    total_loss = gen_loss + enc_loss + readout_loss
    return total_loss, gen_loss.item(), enc_loss.item(), readout_loss.item()


# =============================================================================
# Energy-based K-way inference (test time, the Critical Thing)
# =============================================================================

@torch.no_grad()
def classify_energy_based(model, x, T=13, eta_h=5e-2, momentum_h=0.5):
    """For each test image, run K=num_classes inferences, one per candidate
    class (clamped as target). Return predicted class (argmin energy) and
    per-sample per-hypothesis energies for diagnostics.
    
    Returns:
      pred: (B,) predicted class indices
      all_energies: (B, K) per-sample per-hypothesis total energy
    """
    B = x.size(0)
    K = model.num_classes
    device = x.device

    all_energies = torch.zeros(B, K, device=device)

    # Shared amortised init across hypotheses
    with torch.enable_grad():
        pass  # dummy to make sure autograd is on for the inference loop below

    for k in range(K):
        # Re-init latents from encoder (shared init but we mutate per hypothesis)
        with torch.no_grad():
            latents_init = model.forward_encoder(x)

        h1 = latents_init[0].clone().detach().requires_grad_(True)
        h2 = latents_init[1].clone().detach().requires_grad_(True)
        h3 = latents_init[2].clone().detach().requires_grad_(True)

        # Clamp hypothesis class as target
        y_k = torch.zeros(B, K, device=device)
        y_k[:, k] = 1.0
        h4_clamped = y_k

        m1 = torch.zeros_like(h1)
        m2 = torch.zeros_like(h2)
        m3 = torch.zeros_like(h3)

        # Inference loop
        with torch.enable_grad():
            for t in range(T):
                latents = [h1, h2, h3, h4_clamped]
                per_sample_total, _, _ = model.total_energy(latents, y_onehot=y_k)
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

        # Compute final total energy at settled state (no grad needed)
        with torch.no_grad():
            latents_final = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
            per_sample_total, _, _ = model.total_energy(latents_final, y_onehot=y_k)
            all_energies[:, k] = per_sample_total

    pred = all_energies.argmin(dim=1)
    return pred, all_energies


# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda"
    torch.manual_seed(42)

    print("=" * 60)
    print("IMA DYNAMICS SPIKE")
    print("=" * 60)
    print("Goal: verify energy-based K-way discriminative inference works")
    print("Weak success criteria:")
    print("  1. Per-sample hypothesis energy spread > 1% relative")
    print("  2. Argmin accuracy > 10% (beats chance on 10-way)")
    print("  3. Energy margin AUROC2 > 0.55")
    print()

    # Data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # Model
    model = TinyConvPCN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # Weight optimizer (Pinchetti uses AdamW lr=1e-4)
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training
    print()
    print("Training (5 epochs, T=13)...")
    start = time.time()
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        epoch_gen = 0.0
        epoch_enc = 0.0
        epoch_read = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optim_w.zero_grad()
            total_loss, gen_l, enc_l, read_l = train_step(
                model, x, y, T=13, eta_h=5e-2, momentum_h=0.5
            )
            total_loss.backward()
            optim_w.step()

            epoch_loss += total_loss.item()
            epoch_gen += gen_l
            epoch_enc += enc_l
            epoch_read += read_l
            n_batches += 1

            if batch_idx % 50 == 0:
                print(f"  ep {epoch+1} batch {batch_idx}/{len(train_loader)}: "
                      f"loss={total_loss.item():.3f} "
                      f"gen={gen_l:.3f} enc={enc_l:.3f} read={read_l:.3f}")

        elapsed = time.time() - start
        print(f"  epoch {epoch+1}: avg_loss={epoch_loss/n_batches:.3f} "
              f"({elapsed:.1f}s total)")

    # Evaluation
    print()
    print("Evaluating with energy-based K-way inference...")
    model.eval()
    all_preds = []
    all_targets = []
    all_correct = []
    all_energies_list = []

    # Only use first N batches for spike (fast)
    N_eval_batches = 10  # ~1280 images

    eval_start = time.time()
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= N_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)

        pred, energies = classify_energy_based(
            model, x, T=13, eta_h=5e-2, momentum_h=0.5
        )

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_correct.append((pred == y).cpu())
        all_energies_list.append(energies.cpu())

        if batch_idx % 2 == 0:
            print(f"  eval batch {batch_idx}/{N_eval_batches}")

    eval_elapsed = time.time() - eval_start

    # Concatenate
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    correct = torch.cat(all_correct)
    energies = torch.cat(all_energies_list)  # (N, K)

    N = preds.size(0)
    accuracy = correct.float().mean().item()

    # Diagnostic 1: per-sample hypothesis energy spread
    e_min = energies.min(dim=1).values  # (N,)
    e_max = energies.max(dim=1).values  # (N,)
    e_mean = energies.mean(dim=1)       # (N,)
    relative_spread = (e_max - e_min) / (e_mean.abs() + 1e-8)
    mean_relative_spread = relative_spread.mean().item()

    # Diagnostic 2: argmin accuracy (already computed)

    # Diagnostic 3: AUROC2 using energy margin
    # Energy margin = (2nd lowest energy) - (lowest energy) per sample
    sorted_energies, _ = energies.sort(dim=1)
    energy_margin = sorted_energies[:, 1] - sorted_energies[:, 0]  # positive, higher = more confident

    if correct.sum() > 0 and (~correct).sum() > 0:
        auroc2 = roc_auc_score(correct.numpy(), energy_margin.numpy())
    else:
        auroc2 = float('nan')

    # Report
    print()
    print("=" * 60)
    print("SPIKE RESULTS")
    print("=" * 60)
    print(f"Evaluated on {N} test images ({eval_elapsed:.1f}s)")
    print()
    print(f"Diagnostic 1: per-sample hypothesis energy spread")
    print(f"  mean relative spread: {mean_relative_spread*100:.2f}%")
    print(f"  min/max relative spread: {relative_spread.min()*100:.2f}% / {relative_spread.max()*100:.2f}%")
    print(f"  PASS if > 1%: {'PASS' if mean_relative_spread > 0.01 else 'FAIL'}")
    print()
    print(f"Diagnostic 2: argmin accuracy")
    print(f"  accuracy: {accuracy*100:.2f}%")
    print(f"  PASS if > 10%: {'PASS' if accuracy > 0.10 else 'FAIL'}")
    print()
    print(f"Diagnostic 3: energy margin AUROC2")
    print(f"  AUROC2: {auroc2:.4f}")
    print(f"  PASS if > 0.55: {'PASS' if auroc2 > 0.55 else 'FAIL'}")
    print()

    # Inspection: show 5 example samples
    print("Example samples (first 5):")
    print(f"  {'idx':>4} {'true':>5} {'pred':>5} {'correct':>8} {'e_true':>10} {'e_pred':>10} {'margin':>10}")
    for i in range(min(5, N)):
        e_true = energies[i, targets[i]].item()
        e_pred = energies[i, preds[i]].item()
        margin = energy_margin[i].item()
        print(f"  {i:>4d} {targets[i].item():>5d} {preds[i].item():>5d} "
              f"{'Y' if correct[i] else 'N':>8} {e_true:>10.4f} {e_pred:>10.4f} {margin:>10.4f}")

    # Overall verdict
    print()
    print("=" * 60)
    passed = (
        (mean_relative_spread > 0.01)
        and (accuracy > 0.10)
        and (not math.isnan(auroc2) and auroc2 > 0.55)
    )
    if passed:
        print("SPIKE PASSED: mechanism is viable, proceed to v6 drafting")
    else:
        print("SPIKE FAILED: investigate failure mode before committing to v6")
    print("=" * 60)


if __name__ == "__main__":
    main()
