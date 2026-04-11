"""
spike_langevin_phase_a.py  (Option 3A, Phase A: post-hoc sigma sweep)
====================================================================

Phase A of the MCPC investigation (session 5, 12 Apr 2026).

PURPOSE
-------
Train ONE model with Langevin inference (sigma=1e-2, as in session 5 morning),
then evaluate the K-way structural probe at MULTIPLE sigma values without
retraining. This characterizes the probe's signal across temperatures on a
single trained model.

IMPORTANT CAVEAT
----------------
This is a POST-HOC temperature sweep, not a train-eval matched sweep.
The model was trained at ONE sigma (1e-2). Evaluating at other sigmas
tests "how does this trained model's probe respond to different noise
levels at test time" — NOT "what happens if we train and evaluate at
sigma_k for each k." The latter would require retraining per sigma
(~17 min each), which is too expensive for a phase-A characterization.

A proper train-eval matched sweep would be Phase B extended, and requires
separate decision.

SWEEP VALUES
------------
sigma ∈ {0, 1e-3, 1e-2, 1e-1, 1.0}
  - 0:    deterministic baseline — compare to spike 4 directly
  - 1e-3: gradient-dominated regime
  - 1e-2: reproduction of session 5 morning result
  - 1e-1: hot — noise ≈ init std magnitude
  - 1.0:  very hot — effectively random walk

WEAK SUCCESS CRITERIA (applied per sigma)
-----------------------------------------
C1. max per-layer mean |Δh| > 1e-2
C2. |struct_acc - softmax_acc| > 5pp
C3. struct_auroc2 - softmax_auroc2 >= 0.01

If the whole sweep shows struct_auroc2 << softmax_auroc2 at every sigma,
that's a strong negative result and Phase B (trajectory-integrated training)
is unlikely to rescue it. If there's a sigma where the probe approaches
softmax, Phase B is motivated.
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
# Training step with Langevin inference (sigma fixed during training)
# =============================================================================

def train_step_langevin(model, x, y, T, eta_h, momentum_h, sigma):
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

    latents_enc = model.forward_encoder(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc
    enc_loss = (
        F.mse_loss(h1_enc, h1.detach())
        + F.mse_loss(h2_enc, h2.detach())
        + F.mse_loss(h3_enc, h3.detach())
    )

    latents_for_loss = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
    per_sample_total, _, _ = model.total_energy(latents_for_loss, y_onehot=y_onehot)
    gen_loss = per_sample_total.mean()
    readout_loss = F.cross_entropy(h4_enc, y)

    total_loss = gen_loss + enc_loss + readout_loss
    return total_loss, gen_loss.item(), enc_loss.item(), readout_loss.item()


# =============================================================================
# K-way energy eval — sigma is now a PARAMETER, not fixed
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


# =============================================================================
# Softmax baseline — sigma-independent, computed once
# =============================================================================

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


# =============================================================================
# Full structural eval at one sigma
# =============================================================================

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
        "mv_h1": mv_h1, "mv_h2": mv_h2, "mv_h3": mv_h3,
        "mv_max": mv_max,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--eta_h", type=float, default=1e-2)
    parser.add_argument("--momentum_h", type=float, default=0.5)
    parser.add_argument("--train_sigma", type=float, default=1e-2,
                        help="Sigma used during training (fixed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)

    # Sweep values
    sweep_sigmas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]

    print("=" * 72)
    print("IMA OPTION 3A — PHASE A: POST-HOC SIGMA SWEEP")
    print("=" * 72)
    print(f"  train_sigma = {args.train_sigma}  (fixed during training)")
    print(f"  eval sweep  = {sweep_sigmas}")
    print(f"  T={args.T}  eta_h={args.eta_h}  momentum_h={args.momentum_h}"
          f"  epochs={args.epochs}")
    print()
    print("Note: this is POST-HOC — one trained model, swept eval temperatures.")
    print("      Not a train-eval matched sweep (which would cost ~85 min).")
    print()

    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    model = TinyConvPCN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ----- TRAINING (single run at train_sigma) -----
    print()
    print(f"Training {args.epochs} epochs at sigma={args.train_sigma}...")
    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = epoch_gen = epoch_enc = epoch_read = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optim_w.zero_grad()
            total_loss, gen_l, enc_l, read_l = train_step_langevin(
                model, x, y,
                T=args.T, eta_h=args.eta_h,
                momentum_h=args.momentum_h, sigma=args.train_sigma,
            )
            total_loss.backward()
            optim_w.step()
            epoch_loss += total_loss.item()
            epoch_gen += gen_l
            epoch_enc += enc_l
            epoch_read += read_l
            n_batches += 1
        elapsed = time.time() - start
        print(f"  epoch {epoch+1:2d}: "
              f"loss={epoch_loss/n_batches:.3f} "
              f"gen={epoch_gen/n_batches:.3f} "
              f"enc={epoch_enc/n_batches:.3f} "
              f"read={epoch_read/n_batches:.3f} "
              f"({elapsed:.0f}s)")

    # ----- SOFTMAX BASELINE (sigma-independent, once) -----
    print()
    print("Computing softmax baseline on same network (sigma-independent)...")
    N_eval = 10
    sm_acc, sm_auroc2 = softmax_eval_same_net(model, test_loader, device, N_eval)
    print(f"  softmax_acc    = {sm_acc*100:.2f}%")
    print(f"  softmax_auroc2 = {sm_auroc2:.4f}")

    # ----- POST-HOC SIGMA SWEEP -----
    print()
    print("Post-hoc sigma sweep on trained model...")
    results = []
    for sigma in sweep_sigmas:
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
    print("PHASE A RESULTS: POST-HOC SIGMA SWEEP")
    print("=" * 72)
    print(f"Softmax baseline (same net): "
          f"acc={sm_acc*100:.2f}%  auroc2={sm_auroc2:.4f}")
    print()
    print(f"  {'sigma':>8} {'mv_max':>10} {'struct_acc':>11} {'struct_auroc2':>14} "
          f"{'Δacc (pp)':>11} {'ΔAUROC2':>10} {'C1':>4} {'C2':>4} {'C3':>4}")
    print("  " + "-" * 80)
    for res in results:
        sigma = res["sigma"]
        delta_acc_pp = (res["struct_acc"] - sm_acc) * 100
        delta_auroc = res["struct_auroc2"] - sm_auroc2
        c1 = res["mv_max"] > 1e-2
        c2 = abs(delta_acc_pp) > 5.0
        c3 = (not math.isnan(delta_auroc)) and (delta_auroc >= 0.01)
        print(f"  {sigma:>8.0e} {res['mv_max']:>10.2e} "
              f"{res['struct_acc']*100:>10.2f}% "
              f"{res['struct_auroc2']:>14.4f} "
              f"{delta_acc_pp:>+10.2f}  "
              f"{delta_auroc:>+9.4f} "
              f"{'Y' if c1 else 'N':>4} "
              f"{'Y' if c2 else 'N':>4} "
              f"{'Y' if c3 else 'N':>4}")
    print("  " + "-" * 80)
    print()
    print("Interpretation:")
    print("  sigma=0     : deterministic eval on Langevin-trained model")
    print("               (compare to spike 4 which was det-train + det-eval)")
    print("  sigma=1e-2  : reproduction of session 5 morning run")
    print("  Other sigmas: tests whether any temperature rescues the probe")
    print()
    best_auroc_idx = max(
        range(len(results)),
        key=lambda i: -1 if math.isnan(results[i]["struct_auroc2"]) else results[i]["struct_auroc2"]
    )
    best_res = results[best_auroc_idx]
    best_delta = best_res["struct_auroc2"] - sm_auroc2
    print(f"Best struct_auroc2 at sigma={best_res['sigma']:.0e}: "
          f"{best_res['struct_auroc2']:.4f}  (vs softmax {sm_auroc2:.4f}, "
          f"Δ={best_delta:+.4f})")
    if best_delta >= 0.01:
        print("  -> PHASE A PASS at some sigma. Phase B (trajectory-integrated")
        print("     training) is motivated. Consider proceeding.")
    elif best_delta >= -0.02:
        print("  -> PHASE A ambiguous: best point within 0.02 AUROC of softmax.")
        print("     Phase B may or may not help. Discuss.")
    else:
        print("  -> PHASE A STRONG NEGATIVE: even the best sigma is far below softmax.")
        print("     Phase B unlikely to rescue. Consider falling through to Option 1.")
    print("=" * 72)


if __name__ == "__main__":
    main()
