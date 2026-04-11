"""
spike_langevin_dynamics.py (Option 3A)
======================================

Langevin-noisy variant of spike_dynamics.py (11 Apr 2026).

This is a surgical modification of yesterday's spike 1: the TinyConvPCN
architecture, training-loss structure, and K-way eval protocol are all
preserved verbatim. The ONLY changes are:

  (1) Inference loop adds Langevin noise:   h_l <- h_l - eta_h * m_l + N(0, sigma^2)
  (2) T increased from 13 to 50 (Langevin needs more steps to mix)
  (3) eta_h reduced from 5e-2 to 1e-2 (stability under noise)
  (4) New criterion-1 diagnostic: measure per-element latent movement directly
  (5) New softmax baseline eval on the same network (from forward_encoder's h4)
  (6) Three weak success criteria auto-applied at end

Handover v2 §4.2 weak success criteria:
  C1. Max per-layer mean per-element latent movement > 1e-2
      (vs. spike 2's ~1e-4 deterministic baseline)
  C2. |argmin_acc - softmax_acc| > 5pp
  C3. struct_auroc2 - softmax_auroc2 >= 0.01

If C1 fails: sigma sweep {1e-3, 1e-2, 1e-1, 1.0}, then decide.
If C1 passes but C2 or C3 fails: informative null, fall through to Option 1.

Author: JP Cacioli + Claude (session 5, 12 Apr 2026)
"""

# --- CUDNN WORKAROUND: required for AMD ROCm wheel BN bug (ROCm/ROCm#5441) ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------------------------------------------------------------

import argparse
import math
import sys
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from cifar10_data import get_data_loaders


# =============================================================================
# Tiny ConvPCN — VERBATIM from spike_dynamics.py
# =============================================================================

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

        # Generative weights
        self.gen_fc3 = nn.Linear(num_classes, 256)
        self.gen_fc2 = nn.Linear(256, 64 * 8 * 8)
        self.gen_conv1 = nn.Conv2d(64, 32, 3, padding=1)

    def forward_encoder(self, x):
        h1 = F.gelu(self.enc_bn1(self.enc_conv1(x)))
        h1 = F.max_pool2d(h1, 2)
        h2 = F.gelu(self.enc_bn2(self.enc_conv2(h1)))
        h2 = F.max_pool2d(h2, 2)
        h2_flat = h2.view(x.size(0), -1)
        h3 = F.gelu(self.enc_fc1(h2_flat))
        h4 = self.enc_fc2(h3)
        return [h1, h2, h3, h4]

    def generative_predictions(self, latents):
        h1, h2, h3, h4 = latents
        B = h1.size(0)
        mu_3 = self.gen_fc3(h4)
        mu_2_flat = self.gen_fc2(h3)
        mu_2 = mu_2_flat.view(B, 64, 8, 8)
        h2_up = F.interpolate(h2, scale_factor=2, mode='nearest')
        mu_1 = self.gen_conv1(h2_up)
        return [mu_1, mu_2, mu_3]

    def compute_errors(self, latents):
        mus = self.generative_predictions(latents)
        errors = [latents[l] - mus[l] for l in range(3)]
        return errors

    def total_energy(self, latents, y_onehot=None):
        B = latents[0].size(0)
        errors = self.compute_errors(latents)
        per_sample_lat_e = torch.zeros(B, device=latents[0].device)
        for e in errors:
            e_flat = e.view(B, -1)
            per_sample_lat_e = per_sample_lat_e + 0.5 * e_flat.pow(2).mean(dim=1)
        if y_onehot is not None:
            log_p = F.log_softmax(latents[3], dim=-1)
            ce = -(y_onehot * log_p).sum(dim=-1)
            per_sample_sup_e = ce
        else:
            per_sample_sup_e = torch.zeros(B, device=latents[0].device)
        per_sample_total = per_sample_lat_e + per_sample_sup_e
        return per_sample_total, per_sample_lat_e, per_sample_sup_e


# =============================================================================
# Training step — same as spike_dynamics.train_step EXCEPT for Langevin noise
# =============================================================================

def train_step_langevin(model, x, y, T, eta_h, momentum_h, sigma):
    """Same 3-phase structure as spike 1's train_step:
      Phase 1: amortised init via forward_encoder
      Phase 2: T inference steps — NOW with Langevin noise per update
      Phase 3: weight update using gen_loss + enc_loss + readout_loss
    """
    B = x.size(0)
    num_classes = model.num_classes
    y_onehot = F.one_hot(y, num_classes=num_classes).float()

    with torch.no_grad():
        latents_init = model.forward_encoder(x)

    h1 = latents_init[0].clone().detach().requires_grad_(True)
    h2 = latents_init[1].clone().detach().requires_grad_(True)
    h3 = latents_init[2].clone().detach().requires_grad_(True)
    h4_clamped = y_onehot.clone().detach()

    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    m3 = torch.zeros_like(h3)

    # Phase 2: Langevin inference loop
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
            # --- LANGEVIN NOISE: h <- h - eta*m + sigma*N(0,I) ---
            h1 += -eta_h * m1 + torch.randn_like(h1) * sigma
            h2 += -eta_h * m2 + torch.randn_like(h2) * sigma
            h3 += -eta_h * m3 + torch.randn_like(h3) * sigma
        h1.requires_grad_(True)
        h2.requires_grad_(True)
        h3.requires_grad_(True)

    # Phase 3: weight update (identical to spike 1)
    latents_enc = model.forward_encoder(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc
    enc_loss = (
        F.mse_loss(h1_enc, h1.detach())
        + F.mse_loss(h2_enc, h2.detach())
        + F.mse_loss(h3_enc, h3.detach())
    )

    latents_for_loss = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
    per_sample_total, per_sample_lat, per_sample_sup = model.total_energy(
        latents_for_loss, y_onehot=y_onehot
    )
    gen_loss = per_sample_total.mean()

    readout_loss = F.cross_entropy(h4_enc, y)

    total_loss = gen_loss + enc_loss + readout_loss
    return total_loss, gen_loss.item(), enc_loss.item(), readout_loss.item()


# =============================================================================
# K-way energy-based eval with Langevin inference
# =============================================================================

def classify_langevin(model, x, T, eta_h, momentum_h, sigma):
    """Same K-way protocol as spike 1's classify_energy_based, with Langevin
    noise in the inference loop. Also returns movement snapshots from the
    last hypothesis for criterion-1 diagnostics.
    """
    B = x.size(0)
    K = model.num_classes
    device = x.device
    all_energies = torch.zeros(B, K, device=device)

    last_h1_init = last_h1_final = None
    last_h2_init = last_h2_final = None
    last_h3_init = last_h3_final = None

    for k in range(K):
        with torch.no_grad():
            latents_init = model.forward_encoder(x)

        h1 = latents_init[0].clone().detach().requires_grad_(True)
        h2 = latents_init[1].clone().detach().requires_grad_(True)
        h3 = latents_init[2].clone().detach().requires_grad_(True)

        h1_init_snap = latents_init[0].clone().detach()
        h2_init_snap = latents_init[1].clone().detach()
        h3_init_snap = latents_init[2].clone().detach()

        y_k = torch.zeros(B, K, device=device)
        y_k[:, k] = 1.0
        h4_clamped = y_k

        m1 = torch.zeros_like(h1)
        m2 = torch.zeros_like(h2)
        m3 = torch.zeros_like(h3)

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
                    h1 += -eta_h * m1 + torch.randn_like(h1) * sigma
                    h2 += -eta_h * m2 + torch.randn_like(h2) * sigma
                    h3 += -eta_h * m3 + torch.randn_like(h3) * sigma
                h1.requires_grad_(True)
                h2.requires_grad_(True)
                h3.requires_grad_(True)

        with torch.no_grad():
            latents_final = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
            per_sample_total, _, _ = model.total_energy(latents_final, y_onehot=y_k)
            all_energies[:, k] = per_sample_total

            last_h1_init, last_h1_final = h1_init_snap, h1.detach()
            last_h2_init, last_h2_final = h2_init_snap, h2.detach()
            last_h3_init, last_h3_final = h3_init_snap, h3.detach()

    pred = all_energies.argmin(dim=1)
    return pred, all_energies, (
        last_h1_init, last_h1_final,
        last_h2_init, last_h2_final,
        last_h3_init, last_h3_final,
    )


# =============================================================================
# Softmax baseline on the same network (new, for C2 and C3)
# =============================================================================

@torch.no_grad()
def softmax_eval_same_net(model, test_loader, device, N_batches):
    """Read softmax predictions from the encoder's readout (h4 = enc_fc2)."""
    model.eval()
    all_logits = []
    all_targets = []
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= N_batches:
            break
        x = x.to(device)
        y = y.to(device)
        latents = model.forward_encoder(x)
        all_logits.append(latents[3].cpu())
        all_targets.append(y.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    correct = (preds == targets)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])

    accuracy = correct.float().mean().item()
    if correct.sum() > 0 and (~correct).sum() > 0:
        auroc = roc_auc_score(correct.numpy(), margin.numpy())
    else:
        auroc = float('nan')
    return accuracy, auroc, correct, margin


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (handover recommends 10 for a spike)")
    parser.add_argument("--T", type=int, default=50,
                        help="Inference steps (handover recommends 50 for Langevin)")
    parser.add_argument("--eta_h", type=float, default=1e-2,
                        help="Latent step size (reduced from spike 1's 5e-2)")
    parser.add_argument("--momentum_h", type=float, default=0.5,
                        help="Matches spike 1's momentum_h")
    parser.add_argument("--sigma", type=float, default=1e-2,
                        help="Langevin noise std (sweep if C1 fails)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Use random init, criterion-1 sanity only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("IMA OPTION 3A: LANGEVIN DYNAMICS SPIKE")
    print("=" * 70)
    print(f"  sigma={args.sigma}  T={args.T}  eta_h={args.eta_h}  "
          f"momentum_h={args.momentum_h}")
    print(f"  epochs={args.epochs}  skip_train={args.skip_train}")
    print()
    print("Weak success criteria (handover v2 §4.2):")
    print("  C1. max per-layer mean |Δh| > 1e-2  (vs spike 2 ~1e-4)")
    print("  C2. |struct_acc - softmax_acc| > 5pp")
    print("  C3. struct_auroc2 - softmax_auroc2 >= 0.01")
    print()

    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    model = TinyConvPCN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ----- TRAINING -----
    if not args.skip_train:
        print()
        print(f"Training {args.epochs} epochs with Langevin inference...")
        start = time.time()
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_gen = 0.0
            epoch_enc = 0.0
            epoch_read = 0.0
            n_batches = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optim_w.zero_grad()
                total_loss, gen_l, enc_l, read_l = train_step_langevin(
                    model, x, y,
                    T=args.T, eta_h=args.eta_h,
                    momentum_h=args.momentum_h, sigma=args.sigma,
                )
                total_loss.backward()
                optim_w.step()

                epoch_loss += total_loss.item()
                epoch_gen += gen_l
                epoch_enc += enc_l
                epoch_read += read_l
                n_batches += 1

                if batch_idx % 100 == 0:
                    print(f"  ep {epoch+1} batch {batch_idx}/{len(train_loader)}: "
                          f"loss={total_loss.item():.3f} "
                          f"gen={gen_l:.3f} enc={enc_l:.3f} read={read_l:.3f}")

            elapsed = time.time() - start
            print(f"  epoch {epoch+1}: "
                  f"avg_loss={epoch_loss/n_batches:.3f} "
                  f"gen={epoch_gen/n_batches:.3f} "
                  f"enc={epoch_enc/n_batches:.3f} "
                  f"read={epoch_read/n_batches:.3f} "
                  f"({elapsed:.1f}s total)")
    else:
        print("Skipping training (random init)")

    # ----- EVAL -----
    print()
    print("Running K-way Langevin inference on test set...")
    model.eval()

    N_eval_batches = 10
    all_preds = []
    all_targets = []
    all_correct = []
    all_energies_list = []
    mv_h1_list, mv_h2_list, mv_h3_list = [], [], []
    std_h1_list, std_h2_list, std_h3_list = [], [], []

    eval_start = time.time()
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= N_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)

        pred, energies, movement = classify_langevin(
            model, x,
            T=args.T, eta_h=args.eta_h,
            momentum_h=args.momentum_h, sigma=args.sigma,
        )
        h1_i, h1_f, h2_i, h2_f, h3_i, h3_f = movement
        mv_h1_list.append((h1_f - h1_i).abs().mean().item())
        mv_h2_list.append((h2_f - h2_i).abs().mean().item())
        mv_h3_list.append((h3_f - h3_i).abs().mean().item())
        std_h1_list.append(h1_i.std().item())
        std_h2_list.append(h2_i.std().item())
        std_h3_list.append(h3_i.std().item())

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_correct.append((pred == y).cpu())
        all_energies_list.append(energies.cpu())

        if batch_idx % 2 == 0:
            print(f"  eval batch {batch_idx}/{N_eval_batches}")

    eval_elapsed = time.time() - eval_start

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    correct = torch.cat(all_correct)
    energies = torch.cat(all_energies_list)
    N = preds.size(0)

    struct_acc = correct.float().mean().item()
    sorted_energies, _ = energies.sort(dim=1)
    energy_margin = sorted_energies[:, 1] - sorted_energies[:, 0]
    if correct.sum() > 0 and (~correct).sum() > 0:
        struct_auroc2 = roc_auc_score(correct.numpy(), energy_margin.numpy())
    else:
        struct_auroc2 = float('nan')

    mv_h1 = float(np.mean(mv_h1_list))
    mv_h2 = float(np.mean(mv_h2_list))
    mv_h3 = float(np.mean(mv_h3_list))
    s_h1 = float(np.mean(std_h1_list))
    s_h2 = float(np.mean(std_h2_list))
    s_h3 = float(np.mean(std_h3_list))
    mv_max = max(mv_h1, mv_h2, mv_h3)

    # Softmax baseline
    print()
    print("Softmax baseline on the same network...")
    sm_acc, sm_auroc2, _, _ = softmax_eval_same_net(
        model, test_loader, device, N_eval_batches
    )

    # ===== REPORT =====
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Evaluated on {N} test images ({eval_elapsed:.1f}s)")
    print()
    print("--- Latent movement under Langevin inference (criterion 1) ---")
    print(f"  h1 mean |Δ| = {mv_h1:.3e}   (init std = {s_h1:.3e})")
    print(f"  h2 mean |Δ| = {mv_h2:.3e}   (init std = {s_h2:.3e})")
    print(f"  h3 mean |Δ| = {mv_h3:.3e}   (init std = {s_h3:.3e})")
    print(f"  max         = {mv_max:.3e}")
    print(f"  spike 2 reference: ~1e-4 (deterministic)")
    print()
    print("--- Structural probe (K-way energy argmin under Langevin) ---")
    print(f"  struct_acc     = {struct_acc*100:.2f}%")
    print(f"  struct_auroc2  = {struct_auroc2:.4f}")
    print()
    print("--- Softmax baseline on same network ---")
    print(f"  softmax_acc    = {sm_acc*100:.2f}%")
    print(f"  softmax_auroc2 = {sm_auroc2:.4f}")
    print()

    # Criteria
    delta_acc_pp = abs(struct_acc - sm_acc) * 100
    delta_auroc = struct_auroc2 - sm_auroc2

    c1 = mv_max > 1e-2
    c2 = delta_acc_pp > 5.0
    c3 = (not math.isnan(delta_auroc)) and (delta_auroc >= 0.01)

    print("=" * 70)
    print("WEAK SUCCESS CRITERIA (handover v2 §4.2)")
    print("=" * 70)
    print(f"  C1 (max mean |Δ| > 1e-2):   "
          f"{'PASS' if c1 else 'FAIL'}   (got {mv_max:.3e})")
    print(f"  C2 (|Δacc| > 5pp):           "
          f"{'PASS' if c2 else 'FAIL'}   (got {delta_acc_pp:.2f}pp)")
    print(f"  C3 (ΔAUROC2 >= 0.01):        "
          f"{'PASS' if c3 else 'FAIL'}   (got {delta_auroc:+.4f})")
    print()
    all_pass = c1 and c2 and c3
    if all_pass:
        print("  VERDICT: ALL PASS — Langevin dynamics produce a categorically")
        print("           different probe. Consider full MCPC implementation.")
    else:
        if not c1:
            print("  VERDICT: C1 FAIL — Langevin noise not moving latents enough.")
            print("           Next: sigma sweep {1e-3, 1e-2, 1e-1, 1.0}.")
            print("           If all fail, consider MCPC or fall through to Option 1.")
        elif c1 and not c2:
            print("  VERDICT: C1 pass, C2 fail — informative null. Latents move")
            print("           but probe still tracks softmax. Fall through to Option 1.")
        elif c1 and c2 and not c3:
            print("  VERDICT: C1+C2 pass, C3 fail — noise without signal.")
            print("           Fall through to Option 1 with this data point.")
    print("=" * 70)


if __name__ == "__main__":
    main()
