"""
spike_langevin_phase_b.py  (Option 3A, Phase B: MCPC trajectory-integrated training)
====================================================================================

Phase B of the MCPC investigation (session 5, 12 Apr 2026).

PURPOSE
-------
Test whether the MCPC recipe — averaging weight gradients over multiple
samples from the Langevin chain during training — produces a trained model
whose K-way structural probe meaningfully differs from Phase A's final-state-
trained model.

This is the real MCPC intervention. Phase A added Langevin noise to the
inference loop but still updated weights using the final settled state only.
Phase B additionally integrates the weight gradient over the post-burn-in
Langevin samples, which is the substantive algorithmic difference in
Oliviers et al. (2024) MCPC vs. standard PC.

ALGORITHMIC CHANGE FROM PHASE A
-------------------------------
During Phase 2 inference, save the last M latent states. In Phase 3 (weight
update), compute gen_loss AND enc_loss as averages over those M samples
rather than just using the final state.

    Phase A (final-state):
        gen_loss = E(x, z_T, θ).mean()
        enc_loss = MSE(enc(x), z_T).mean()

    Phase B (trajectory-integrated, MCPC-style):
        gen_loss = (1/M) Σ_{t=T-M..T} E(x, z_t, θ).mean()
        enc_loss = (1/M) Σ_{t=T-M..T} MSE(enc(x), z_t).mean()

M = 10 samples from the last 10 steps of a T=50 Langevin chain (20% post-burn-in).

EVAL
----
After training, evaluate the structural probe at TWO key sigmas:
  - sigma=0    (deterministic eval — the point that matters most for
                comparison against Phase A's 0.7531 and spike 4's 0.7661)
  - sigma=1e-2 (matched to training sigma — the "fair" MCPC eval)

WEAK SUCCESS CRITERIA
---------------------
C1 (latent movement): always passes at eval sigma > 0, not informative here
C2 (|Δacc| > 5pp): diagnostic
C3 (ΔAUROC2 >= 0.01): the binding criterion

FALL-THROUGH LOGIC
------------------
If Phase B's best struct_auroc2 (across sigma=0 and sigma=1e-2) is still
below softmax by more than 0.01, fall through to Option 1 per handover §3.
No further MCPC variants. No Phase C.
"""

# --- CUDNN WORKAROUND ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------

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
# TinyConvPCN — VERBATIM from spike_dynamics.py
# =============================================================================

class TinyConvPCN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.enc_fc2 = nn.Linear(256, num_classes)
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
        return [latents[l] - mus[l] for l in range(3)]

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
# MCPC training step: trajectory-integrated weight updates
# =============================================================================

def train_step_mcpc(model, x, y, T, eta_h, momentum_h, sigma, M):
    """MCPC-style training step.

    Phase 1: amortised init via forward_encoder
    Phase 2: T inference steps with Langevin noise, saving the last M latent
             states after burn-in
    Phase 3: weight update using gen_loss and enc_loss averaged over the M
             saved samples, plus the (unchanged) readout loss on h4_enc.
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

    # Buffer for post-burn-in samples. Collect the latents AFTER each of the
    # final M updates (so the last sample is the fully-settled state, same
    # as Phase A's single-sample target).
    sample_buffer = []  # list of (h1_t, h2_t, h3_t) tuples

    # Phase 2: Langevin inference, saving last M states
    for t in range(T):
        latents = [h1, h2, h3, h4_clamped]
        per_sample_total, _, _ = model.total_energy(latents, y_onehot=y_onehot)
        total_energy_scalar = per_sample_total.sum()

        grads = torch.autograd.grad(
            total_energy_scalar, [h1, h2, h3],
            create_graph=False, retain_graph=False,
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

        # Save sample if in the last M steps
        if t >= T - M:
            sample_buffer.append((
                h1.detach().clone(),
                h2.detach().clone(),
                h3.detach().clone(),
            ))

    # Phase 3: trajectory-integrated weight update
    latents_enc = model.forward_encoder(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc

    gen_loss_accum = 0.0
    enc_loss_accum = 0.0

    for (h1_t, h2_t, h3_t) in sample_buffer:
        # gen_loss term: energy at this sample
        latents_t = [h1_t, h2_t, h3_t, h4_clamped]
        per_sample_total_t, _, _ = model.total_energy(latents_t, y_onehot=y_onehot)
        gen_loss_accum = gen_loss_accum + per_sample_total_t.mean()

        # enc_loss term: encoder should predict this sample
        enc_loss_accum = enc_loss_accum + (
            F.mse_loss(h1_enc, h1_t)
            + F.mse_loss(h2_enc, h2_t)
            + F.mse_loss(h3_enc, h3_t)
        )

    gen_loss = gen_loss_accum / M
    enc_loss = enc_loss_accum / M

    readout_loss = F.cross_entropy(h4_enc, y)
    total_loss = gen_loss + enc_loss + readout_loss

    return total_loss, gen_loss.item(), enc_loss.item(), readout_loss.item()


# =============================================================================
# Eval (same as Phase A)
# =============================================================================

def classify_langevin(model, x, T, eta_h, momentum_h, sigma):
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
                    total_energy_scalar, [h1, h2, h3],
                    create_graph=False, retain_graph=False,
                )
                with torch.no_grad():
                    m1 = momentum_h * m1 + grads[0]
                    m2 = momentum_h * m2 + grads[1]
                    m3 = momentum_h * m3 + grads[2]
                    if sigma > 0:
                        h1 += -eta_h * m1 + torch.randn_like(h1) * sigma
                        h2 += -eta_h * m2 + torch.randn_like(h2) * sigma
                        h3 += -eta_h * m3 + torch.randn_like(h3) * sigma
                    else:
                        h1 -= eta_h * m1
                        h2 -= eta_h * m2
                        h3 -= eta_h * m3
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


@torch.no_grad()
def softmax_eval_same_net(model, test_loader, device, N_batches):
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
    return accuracy, auroc


def struct_eval_at_sigma(model, test_loader, device, T, eta_h, momentum_h,
                         sigma, N_eval_batches):
    model.eval()
    all_preds = []
    all_targets = []
    all_correct = []
    all_energies_list = []
    mv_h1_list, mv_h2_list, mv_h3_list = [], [], []

    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= N_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred, energies, movement = classify_langevin(
            model, x, T=T, eta_h=eta_h, momentum_h=momentum_h, sigma=sigma,
        )
        h1_i, h1_f, h2_i, h2_f, h3_i, h3_f = movement
        mv_h1_list.append((h1_f - h1_i).abs().mean().item())
        mv_h2_list.append((h2_f - h2_i).abs().mean().item())
        mv_h3_list.append((h3_f - h3_i).abs().mean().item())

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_correct.append((pred == y).cpu())
        all_energies_list.append(energies.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    correct = torch.cat(all_correct)
    energies = torch.cat(all_energies_list)

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
    mv_max = max(mv_h1, mv_h2, mv_h3)

    return {
        "sigma": sigma,
        "struct_acc": struct_acc,
        "struct_auroc2": struct_auroc2,
        "mv_max": mv_max,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--M", type=int, default=10,
                        help="Number of post-burn-in samples for MCPC averaging")
    parser.add_argument("--eta_h", type=float, default=1e-2)
    parser.add_argument("--momentum_h", type=float, default=0.5)
    parser.add_argument("--train_sigma", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("IMA OPTION 3A — PHASE B: MCPC TRAJECTORY-INTEGRATED TRAINING")
    print("=" * 72)
    print(f"  train_sigma = {args.train_sigma}")
    print(f"  T={args.T}  M={args.M}  eta_h={args.eta_h}  "
          f"momentum_h={args.momentum_h}  epochs={args.epochs}")
    print()
    print("Phase B change: gen_loss and enc_loss averaged over last M latent")
    print("samples from the Langevin chain (vs Phase A's final-state only).")
    print()
    print("Expected ~2x slower than Phase A due to M-fold forward passes in")
    print("Phase 3. Estimate: ~35 min training + ~1 min eval.")
    print()
    print("Key comparison points:")
    print("  Phase A sigma=0    struct_auroc2 = 0.7531")
    print("  Phase A sigma=1e-2 struct_auroc2 = 0.7360")
    print("  Softmax baseline (both phases): ~0.83")
    print()

    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    model = TinyConvPCN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ----- TRAINING -----
    print()
    print(f"Training {args.epochs} epochs with MCPC (M={args.M})...")
    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = epoch_gen = epoch_enc = epoch_read = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optim_w.zero_grad()
            total_loss, gen_l, enc_l, read_l = train_step_mcpc(
                model, x, y,
                T=args.T, eta_h=args.eta_h,
                momentum_h=args.momentum_h, sigma=args.train_sigma,
                M=args.M,
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
        print(f"  epoch {epoch+1:2d}: "
              f"loss={epoch_loss/n_batches:.3f} "
              f"gen={epoch_gen/n_batches:.3f} "
              f"enc={epoch_enc/n_batches:.3f} "
              f"read={epoch_read/n_batches:.3f} "
              f"({elapsed:.0f}s)")

    # ----- EVAL -----
    print()
    print("Computing softmax baseline on same network...")
    N_eval = 10
    sm_acc, sm_auroc2 = softmax_eval_same_net(model, test_loader, device, N_eval)
    print(f"  softmax_acc    = {sm_acc*100:.2f}%")
    print(f"  softmax_auroc2 = {sm_auroc2:.4f}")

    print()
    print("Evaluating structural probe at sigma=0 and sigma=1e-2...")
    results = []
    for sigma in [0.0, 1e-2]:
        t0 = time.time()
        print(f"  sigma={sigma:.0e}... ", end="", flush=True)
        res = struct_eval_at_sigma(
            model, test_loader, device,
            T=args.T, eta_h=args.eta_h, momentum_h=args.momentum_h,
            sigma=sigma, N_eval_batches=N_eval,
        )
        dt = time.time() - t0
        print(f"acc={res['struct_acc']*100:5.2f}%  "
              f"auroc2={res['struct_auroc2']:.4f}  "
              f"mv_max={res['mv_max']:.2e}  ({dt:.0f}s)")
        results.append(res)

    # ----- RESULTS TABLE -----
    print()
    print("=" * 72)
    print("PHASE B RESULTS")
    print("=" * 72)
    print(f"Softmax baseline (same net): "
          f"acc={sm_acc*100:.2f}%  auroc2={sm_auroc2:.4f}")
    print()
    print(f"  {'eval sigma':>10} {'mv_max':>10} {'struct_acc':>11} "
          f"{'struct_auroc2':>14} {'Δacc (pp)':>11} {'ΔAUROC2':>10} "
          f"{'C3':>4}")
    print("  " + "-" * 76)
    for res in results:
        sigma = res["sigma"]
        delta_acc_pp = (res["struct_acc"] - sm_acc) * 100
        delta_auroc = res["struct_auroc2"] - sm_auroc2
        c3 = (not math.isnan(delta_auroc)) and (delta_auroc >= 0.01)
        print(f"  {sigma:>10.0e} {res['mv_max']:>10.2e} "
              f"{res['struct_acc']*100:>10.2f}% "
              f"{res['struct_auroc2']:>14.4f} "
              f"{delta_acc_pp:>+10.2f}  "
              f"{delta_auroc:>+9.4f} "
              f"{'Y' if c3 else 'N':>4}")
    print("  " + "-" * 76)
    print()

    # Comparison against Phase A
    print("Phase B vs Phase A (same model architecture, same seed, same")
    print("training hyperparameters except MCPC trajectory integration):")
    print()
    print(f"  {'eval sigma':>10} {'Phase A auroc2':>16} {'Phase B auroc2':>16} "
          f"{'Δ (B-A)':>10}")
    print("  " + "-" * 56)
    phase_a_auroc = {0.0: 0.7531, 1e-2: 0.7360}
    for res in results:
        sigma = res["sigma"]
        a = phase_a_auroc.get(sigma, float('nan'))
        b = res["struct_auroc2"]
        delta_ba = b - a
        print(f"  {sigma:>10.0e} {a:>16.4f} {b:>16.4f} {delta_ba:>+10.4f}")
    print()

    # Verdict
    best_res = max(
        results,
        key=lambda r: -1 if math.isnan(r["struct_auroc2"]) else r["struct_auroc2"]
    )
    best_delta_softmax = best_res["struct_auroc2"] - sm_auroc2
    best_sigma_a = phase_a_auroc.get(best_res["sigma"], float('nan'))
    best_delta_phase_a = best_res["struct_auroc2"] - best_sigma_a

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"Best Phase B point: sigma={best_res['sigma']:.0e}  "
          f"auroc2={best_res['struct_auroc2']:.4f}")
    print(f"  vs softmax:  Δ = {best_delta_softmax:+.4f}")
    print(f"  vs Phase A:  Δ = {best_delta_phase_a:+.4f}")
    print()

    if best_delta_softmax >= 0.01:
        print("  PHASE B PASS: structural probe exceeds softmax at some sigma.")
        print("  This would be a POSITIVE result for MCPC. Proceed carefully:")
        print("  rerun with 3 seeds to verify the effect isn't seed artefact.")
    elif best_delta_softmax >= -0.02:
        print("  PHASE B AMBIGUOUS: best point within 0.02 AUROC of softmax.")
        print("  Not a clear pass or fail. Discuss next steps with JP.")
    else:
        print("  PHASE B NEGATIVE: best point still well below softmax.")
        print("  Per handover §3 stopping criterion: fall through to Option 1.")
        print("  The negative-result paper (§1.9 reduction + six spikes +")
        print("  MCPC characterization) is the publishable contribution.")
    print("=" * 72)


if __name__ == "__main__":
    main()
