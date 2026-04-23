"""
Temperature Scaling Ablation (Post-hoc causal test)
=====================================================

Reviews 1-3 all flagged the same gap: the three-condition gradient
(CE → MSE → bPC) is correlational. The logit inflation mechanism
is plausible but not causally demonstrated. This script provides
the direct intervention.

Protocol:
  For each of the 10 stdPC-CE trained models (seeds 6-15):
  1. Load the raw output logits from the evaluation set (1280 images)
  2. Apply temperature scaling: logits_scaled = logits / T
     where T is chosen to match the mean logit norm of the MSE condition
  3. Recompute softmax AUROC2 on the rescaled logits
  4. Compare rescaled softmax AUROC2 to the original probe AUROC2

If the gap closes (rescaled softmax AUROC2 ≈ probe AUROC2), the
logit inflation mechanism is confirmed: CE's AUROC2 advantage over
the probe is driven by logit scale, not by superior ranking.

If the gap does NOT close, the mechanism is wrong: CE produces
genuinely better-ranked confidence signals, not just scaled-up ones.

This requires re-running evaluation on the stdPC-CE models to capture
raw logits. Since we didn't save model checkpoints, we retrain at
the same seeds and capture logits during eval.

Alternative: if model checkpoints exist, load directly.

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
import json
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


def train_and_collect_logits(seed, device, data_dir="data"):
    """Train stdPC-CE at a given seed, then collect raw logits on eval set.
    
    Returns:
        logits: (N, 10) raw output logits
        targets: (N,) true labels
        probe_auroc2: float, K-way energy probe AUROC2
        probe_correct: (N,) bool, probe's correct/incorrect
        probe_margins: (N,) energy margins
    """
    from spike_dynamics import TinyConvPCN, train_step, classify_energy_based
    from cifar10_data import get_data_loaders
    
    set_all_seeds(seed)
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=128, num_workers=0)
    model = TinyConvPCN(num_classes=10).to(device)
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Train 25 epochs (must match main experiment exactly)
    start = time.time()
    model.train()
    for epoch in range(25):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim_w.zero_grad()
            total_loss, _, _, _ = train_step(model, x, y, T=13, eta_h=5e-2, momentum_h=0.5)
            total_loss.backward()
            optim_w.step()
        elapsed = time.time() - start
        print(f"    epoch {epoch+1:2d} ({elapsed:.0f}s)", end="\r")
    print()
    
    # Collect raw logits and probe results on eval set
    model.eval()
    all_logits = []
    all_targets = []
    all_probe_correct = []
    all_probe_margins = []
    
    n_batches = 10
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        
        # Raw logits
        with torch.no_grad():
            latents = model.forward_encoder(x)
            logits = latents[3]  # (B, 10)
        
        # K-way energy probe
        pred_probe, energies = classify_energy_based(
            model, x, T=13, eta_h=5e-2, momentum_h=0.5
        )
        sorted_e, _ = energies.sort(dim=1)
        energy_margin = sorted_e[:, 1] - sorted_e[:, 0]
        
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        all_probe_correct.append((pred_probe == y).cpu())
        all_probe_margins.append(energy_margin.cpu())
    
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probe_correct = torch.cat(all_probe_correct)
    probe_margins = torch.cat(all_probe_margins)
    
    # Probe AUROC2
    if probe_correct.sum() > 0 and (~probe_correct).sum() > 0:
        probe_auroc2 = roc_auc_score(probe_correct.numpy(), probe_margins.numpy())
    else:
        probe_auroc2 = float('nan')
    
    return logits, targets, probe_auroc2, probe_correct, probe_margins


def compute_softmax_auroc2(logits, targets, temperature=1.0):
    """Compute softmax AUROC2 at a given temperature.
    
    logits_scaled = logits / temperature
    """
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    pred = scaled.argmax(dim=-1)
    correct = (pred == targets)
    conf = probs.max(dim=-1).values
    
    if correct.sum() > 0 and (~correct).sum() > 0:
        auroc2 = roc_auc_score(correct.numpy(), conf.numpy())
    else:
        auroc2 = float('nan')
    
    acc = correct.float().mean().item()
    logit_norm = scaled.norm(dim=-1).mean().item()
    
    return auroc2, acc, logit_norm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("TEMPERATURE SCALING ABLATION (Causal test of logit inflation)")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    
    # Load main experiment results for reference values
    with open("results/main_experiment.json", 'r') as f:
        main_data = json.load(f)
    
    mse_runs = [r for r in main_data['per_run_results'] if r['condition'] == 'MSE']
    target_logit_norm = np.mean([r['logit_norm_mean'] for r in mse_runs])
    
    print(f"Target logit norm (from MSE condition): {target_logit_norm:.4f}")
    print()
    print("Protocol:")
    print("  1. Retrain stdPC-CE at each seed (deterministic reproduction)")
    print("  2. Collect raw logits on eval set")
    print("  3. Apply temperature scaling to match MSE logit norm")
    print("  4. Recompute softmax AUROC2 at original and rescaled temperatures")
    print("  5. Compare rescaled softmax AUROC2 to probe AUROC2")
    print()
    print("Prediction if logit inflation mechanism is correct:")
    print("  rescaled softmax AUROC2 ≈ probe AUROC2")
    print("  (gap closes)")
    print()
    print("Prediction if mechanism is wrong:")
    print("  rescaled softmax AUROC2 still >> probe AUROC2")
    print("  (gap persists)")
    print()
    
    seeds = list(range(6, 16))
    results = []
    
    # Also test a sweep of temperatures for the figure
    temp_sweep = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    sweep_results = {t: [] for t in temp_sweep}
    
    for seed in seeds:
        print(f"--- Seed {seed} ---")
        print(f"  Training stdPC-CE (25 epochs)...")
        
        logits, targets, probe_auroc2, probe_correct, probe_margins = \
            train_and_collect_logits(seed, device)
        
        # Original (T=1)
        orig_auroc2, orig_acc, orig_norm = compute_softmax_auroc2(logits, targets, temperature=1.0)
        
        # Find temperature that matches MSE logit norm
        # logit_norm_scaled = logit_norm_orig / T
        # Want logit_norm_scaled = target_logit_norm
        # So T = logit_norm_orig / target_logit_norm
        T_match = orig_norm / target_logit_norm
        
        # Rescaled
        rescaled_auroc2, rescaled_acc, rescaled_norm = compute_softmax_auroc2(
            logits, targets, temperature=T_match
        )
        
        # Probe AUROC2 (from the same model)
        delta_orig = probe_auroc2 - orig_auroc2
        delta_rescaled = probe_auroc2 - rescaled_auroc2
        
        result = {
            'seed': seed,
            'orig_logit_norm': orig_norm,
            'orig_softmax_auroc2': orig_auroc2,
            'orig_softmax_acc': orig_acc,
            'T_match': T_match,
            'rescaled_logit_norm': rescaled_norm,
            'rescaled_softmax_auroc2': rescaled_auroc2,
            'rescaled_softmax_acc': rescaled_acc,
            'probe_auroc2': probe_auroc2,
            'delta_orig': delta_orig,
            'delta_rescaled': delta_rescaled,
        }
        results.append(result)
        
        print(f"  Orig:     logit_norm={orig_norm:.2f}, softmax_AUROC2={orig_auroc2:.4f}, "
              f"Delta={delta_orig:+.4f}")
        print(f"  Rescaled: T={T_match:.2f}, logit_norm={rescaled_norm:.2f}, "
              f"softmax_AUROC2={rescaled_auroc2:.4f}, Delta={delta_rescaled:+.4f}")
        print(f"  Probe:    AUROC2={probe_auroc2:.4f}")
        print()
        
        # Temperature sweep for figure
        for t in temp_sweep:
            t_auroc2, t_acc, t_norm = compute_softmax_auroc2(logits, targets, temperature=t)
            sweep_results[t].append({
                'seed': seed,
                'temperature': t,
                'softmax_auroc2': t_auroc2,
                'softmax_acc': t_acc,
                'logit_norm': t_norm,
                'probe_auroc2': probe_auroc2,
            })
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print()
    
    print(f"{'Seed':>4}  {'Orig AUROC2':>12}  {'Rescaled':>10}  {'Probe':>8}  "
          f"{'Δ_orig':>8}  {'Δ_rescaled':>11}  {'T':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['seed']:>4}  {r['orig_softmax_auroc2']:>12.4f}  "
              f"{r['rescaled_softmax_auroc2']:>10.4f}  "
              f"{r['probe_auroc2']:>8.4f}  "
              f"{r['delta_orig']:>+8.4f}  "
              f"{r['delta_rescaled']:>+11.4f}  "
              f"{r['T_match']:>6.1f}")
    
    # Means
    mean_delta_orig = np.mean([r['delta_orig'] for r in results])
    mean_delta_rescaled = np.mean([r['delta_rescaled'] for r in results])
    mean_orig_auroc = np.mean([r['orig_softmax_auroc2'] for r in results])
    mean_rescaled_auroc = np.mean([r['rescaled_softmax_auroc2'] for r in results])
    mean_probe_auroc = np.mean([r['probe_auroc2'] for r in results])
    
    print()
    print(f"Mean original softmax AUROC2:  {mean_orig_auroc:.4f}")
    print(f"Mean rescaled softmax AUROC2:  {mean_rescaled_auroc:.4f}")
    print(f"Mean probe AUROC2:             {mean_probe_auroc:.4f}")
    print(f"Mean Delta (orig):             {mean_delta_orig:+.4f}")
    print(f"Mean Delta (rescaled):         {mean_delta_rescaled:+.4f}")
    print()
    
    # Interpretation
    gap_reduction = abs(mean_delta_orig) - abs(mean_delta_rescaled)
    gap_reduction_pct = gap_reduction / abs(mean_delta_orig) * 100 if abs(mean_delta_orig) > 0 else 0
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print(f"Temperature scaling reduced the probe-softmax gap by "
          f"{gap_reduction:.4f} ({gap_reduction_pct:.0f}%).")
    print()
    
    if abs(mean_delta_rescaled) < 0.01:
        print("MECHANISM CONFIRMED: Temperature scaling closes the gap.")
        print("The original softmax AUROC2 advantage was driven by logit scale,")
        print("not by superior ranking. CE-induced logit inflation is the")
        print("primary mechanism behind the IMA negative result.")
    elif gap_reduction_pct > 50:
        print("MECHANISM PARTIALLY CONFIRMED: Temperature scaling reduces the")
        print("gap substantially but does not fully close it. Logit inflation")
        print("is a major contributor but not the sole mechanism.")
    else:
        print("MECHANISM NOT CONFIRMED: Temperature scaling does not substantially")
        print("reduce the gap. CE produces genuinely better-ranked confidence")
        print("signals, not just scaled-up logits. The logit inflation account")
        print("is insufficient.")
    
    # Temperature sweep summary for figure
    print()
    print("Temperature sweep (means across seeds):")
    print(f"{'T':>6}  {'Softmax AUROC2':>15}  {'Logit Norm':>11}  {'Probe AUROC2':>13}")
    print("-" * 50)
    for t in temp_sweep:
        mean_auroc = np.mean([r['softmax_auroc2'] for r in sweep_results[t]])
        mean_norm = np.mean([r['logit_norm'] for r in sweep_results[t]])
        mean_probe = np.mean([r['probe_auroc2'] for r in sweep_results[t]])
        print(f"{t:>6.1f}  {mean_auroc:>15.4f}  {mean_norm:>11.2f}  {mean_probe:>13.4f}")
    
    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        'per_seed_results': results,
        'temperature_sweep': {str(t): sweep_results[t] for t in temp_sweep},
        'target_logit_norm': target_logit_norm,
    }
    with open("results/temperature_ablation.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to results/temperature_ablation.json")


if __name__ == "__main__":
    main()
