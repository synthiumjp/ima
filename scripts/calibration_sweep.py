"""
bPC α_gen Calibration Sweep (Stage 1 of pre-registration)
==========================================================

Pre-registration reference: Amendment 1, §6.1

Protocol:
  1. Train TinyConvBPC at 4 α_gen values {1e-3, 1e-4, 1e-5, 1e-6}
     across 5 seeds (1-5). Total: 20 training runs.
  2. For each α_gen, record:
     - Mean Type-1 accuracy at epoch 25 (softmax argmax on V-pathway)
     - Mean K-way energy probe AUROC2 at epoch 25
     - Mean energy margin E_(2) - E_(1) across test images
  3. Select α_gen with highest mean Type-1 accuracy.
  4. Tie-break: if multiple values within 1pp, take closest to 1e-5.
  5. Lock selected α_gen before Stage 2.

This is hyperparameter selection, not hypothesis testing.
All results reported in supplementary material.

Hardware: AMD RX 7900 GRE, ROCm 6.4.4, Windows 11
Estimated time: ~15 hours (20 runs × ~45 min each)

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
from cifar10_data_bpc import get_data_loaders_bpc
from tiny_conv_bpc import TinyConvBPC, bpc_train_step, bpc_classify_energy_based


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_bpc(model, test_loader, device, n_batches=10, T_eval=100,
                 eta_h=5e-2, momentum_h=0.5):
    """Evaluate a trained bPC network.
    
    Returns dict with:
      - softmax_acc: V-pathway argmax accuracy
      - probe_acc: energy argmin accuracy
      - softmax_auroc2: AUROC2 using softmax confidence margin
      - probe_auroc2: AUROC2 using energy margin
      - mean_energy_margin: mean E_(2) - E_(1) across test images
      - logit_norm_mean: mean L2 norm of output logits
      - logit_margin_mean: mean (top1 - top2) raw logit
    """
    model.eval()
    
    all_correct_softmax = []
    all_correct_probe = []
    all_softmax_conf = []
    all_energy_margins = []
    all_logit_norms = []
    all_logit_margins = []
    
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)
        
        # K-way energy probe
        pred_probe, energies, gen_e, disc_e = bpc_classify_energy_based(
            model, x, T=T_eval, eta_h=eta_h, momentum_h=momentum_h
        )
        
        # Energy margin
        sorted_e, _ = energies.sort(dim=1)
        energy_margin = sorted_e[:, 1] - sorted_e[:, 0]
        
        # Softmax from V pathway
        with torch.no_grad():
            latents = model.forward_v(x)
            logits = latents[3]
            probs = F.softmax(logits, dim=-1)
            pred_softmax = logits.argmax(dim=-1)
            softmax_conf = probs.max(dim=-1).values
            
            # Logit diagnostics
            logit_norms = logits.norm(dim=-1)
            sorted_logits, _ = logits.sort(dim=-1, descending=True)
            logit_margins_batch = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        all_correct_softmax.append((pred_softmax == y).cpu())
        all_correct_probe.append((pred_probe == y).cpu())
        all_softmax_conf.append(softmax_conf.cpu())
        all_energy_margins.append(energy_margin.cpu())
        all_logit_norms.append(logit_norms.cpu())
        all_logit_margins.append(logit_margins_batch.cpu())
        
        print(f"    eval batch {batch_idx+1}/{n_batches}", end="\r")
    
    print()  # clear the \r
    
    correct_softmax = torch.cat(all_correct_softmax)
    correct_probe = torch.cat(all_correct_probe)
    softmax_conf = torch.cat(all_softmax_conf)
    energy_margins = torch.cat(all_energy_margins)
    logit_norms = torch.cat(all_logit_norms)
    logit_margins = torch.cat(all_logit_margins)
    
    softmax_acc = correct_softmax.float().mean().item()
    probe_acc = correct_probe.float().mean().item()
    
    # AUROC2 for softmax
    if correct_softmax.sum() > 0 and (~correct_softmax).sum() > 0:
        softmax_auroc2 = roc_auc_score(correct_softmax.numpy(), softmax_conf.numpy())
    else:
        softmax_auroc2 = float('nan')
    
    # AUROC2 for probe
    if correct_probe.sum() > 0 and (~correct_probe).sum() > 0:
        probe_auroc2 = roc_auc_score(correct_probe.numpy(), energy_margins.numpy())
    else:
        probe_auroc2 = float('nan')
    
    return {
        'softmax_acc': softmax_acc,
        'probe_acc': probe_acc,
        'softmax_auroc2': softmax_auroc2,
        'probe_auroc2': probe_auroc2,
        'mean_energy_margin': energy_margins.mean().item(),
        'logit_norm_mean': logit_norms.mean().item(),
        'logit_margin_mean': logit_margins.mean().item(),
    }


def train_and_evaluate(seed, alpha_gen, device, data_dir="data",
                       epochs=25, T_train=32, T_eval=100,
                       eta_h=5e-2, momentum_h=0.5, batch_size=128):
    """Train one TinyConvBPC network and evaluate it.
    
    Returns evaluation results dict.
    """
    set_all_seeds(seed)
    
    train_loader, test_loader = get_data_loaders_bpc(
        data_dir, batch_size=batch_size, num_workers=0
    )
    
    model = TinyConvBPC(
        num_classes=10, alpha_gen=alpha_gen, alpha_disc=1.0
    ).to(device)
    
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    start = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optim_w.zero_grad()
            total_loss, gen_l, disc_l, enc_l = bpc_train_step(
                model, x, y, T=T_train, eta_h=eta_h, momentum_h=momentum_h
            )
            total_loss.backward()
            optim_w.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        elapsed = time.time() - start
        avg_loss = epoch_loss / n_batches
        print(f"  epoch {epoch+1:2d}: loss={avg_loss:.4f} ({elapsed:.0f}s)")
    
    # Evaluate
    print(f"  Evaluating (T_eval={T_eval})...")
    results = evaluate_bpc(
        model, test_loader, device, n_batches=10,
        T_eval=T_eval, eta_h=eta_h, momentum_h=momentum_h
    )
    
    train_time = time.time() - start
    results['train_time_s'] = train_time
    results['seed'] = seed
    results['alpha_gen'] = alpha_gen
    
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("bPC α_gen CALIBRATION SWEEP (Stage 1)")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    print("Protocol (Amendment 1, §6.1):")
    print("  Seeds: 1-5")
    print("  α_gen values: 1e-3, 1e-4, 1e-5, 1e-6")
    print("  α_disc: 1.0 (fixed)")
    print("  Epochs: 25")
    print("  T_train: 32, T_eval: 100")
    print("  Selection: highest mean Type-1 accuracy (softmax argmax)")
    print("  Tie-break: closest to 1e-5")
    print()
    
    alpha_gen_values = [1e-3, 1e-4, 1e-5, 1e-6]
    seeds = [1, 2, 3, 4, 5]
    
    # Resume support: load any previously completed runs
    os.makedirs("results", exist_ok=True)
    checkpoint_path = "results/calibration_sweep_checkpoint.json"
    all_results = []
    completed = set()
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            all_results = json.load(f)
        for r in all_results:
            completed.add((r['alpha_gen'], r['seed']))
        print(f"Resumed from checkpoint: {len(completed)} runs already completed.")
    
    total_runs = len(alpha_gen_values) * len(seeds)
    run_count = len(completed)
    
    sweep_start = time.time()
    
    for alpha_gen in alpha_gen_values:
        print(f"\n{'='*70}")
        print(f"α_gen = {alpha_gen:.0e}")
        print(f"{'='*70}")
        
        for seed in seeds:
            # Skip if already completed
            if (alpha_gen, seed) in completed:
                run_count_display = sum(1 for ag in alpha_gen_values for s in seeds 
                                        if (ag, s) in completed 
                                        or (ag == alpha_gen and s == seed))
                print(f"\n--- Run {run_count_display}/{total_runs}: "
                      f"α_gen={alpha_gen:.0e}, seed={seed} — SKIPPED (already done) ---")
                continue
            
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs}: "
                  f"α_gen={alpha_gen:.0e}, seed={seed} ---")
            
            results = train_and_evaluate(
                seed=seed, alpha_gen=alpha_gen, device=device
            )
            all_results.append(results)
            
            print(f"  Results: softmax_acc={results['softmax_acc']*100:.1f}% "
                  f"probe_acc={results['probe_acc']*100:.1f}% "
                  f"probe_AUROC2={results['probe_auroc2']:.4f} "
                  f"margin={results['mean_energy_margin']:.6f}")
            
            # Save checkpoint after each completed run
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  [checkpoint saved: {len(all_results)}/{total_runs} runs]")
    
    sweep_time = time.time() - sweep_start
    
    # =========================================================================
    # Summary and selection
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION SWEEP SUMMARY")
    print("=" * 70)
    
    # Aggregate by alpha_gen
    summary = {}
    for alpha_gen in alpha_gen_values:
        runs = [r for r in all_results if r['alpha_gen'] == alpha_gen]
        accs = [r['softmax_acc'] for r in runs]
        aurocs = [r['probe_auroc2'] for r in runs]
        margins = [r['mean_energy_margin'] for r in runs]
        probe_accs = [r['probe_acc'] for r in runs]
        
        summary[alpha_gen] = {
            'mean_softmax_acc': np.mean(accs),
            'std_softmax_acc': np.std(accs),
            'mean_probe_acc': np.mean(probe_accs),
            'mean_probe_auroc2': np.mean(aurocs),
            'std_probe_auroc2': np.std(aurocs),
            'mean_margin': np.mean(margins),
            'std_margin': np.std(margins),
        }
    
    print(f"\n{'α_gen':>10} {'Softmax Acc':>12} {'Probe Acc':>10} "
          f"{'Probe AUROC2':>13} {'Energy Margin':>14}")
    print("-" * 70)
    for ag in alpha_gen_values:
        s = summary[ag]
        print(f"{ag:>10.0e} "
              f"{s['mean_softmax_acc']*100:>9.2f}±{s['std_softmax_acc']*100:.2f}% "
              f"{s['mean_probe_acc']*100:>8.2f}% "
              f"{s['mean_probe_auroc2']:>10.4f}±{s['std_probe_auroc2']:.4f} "
              f"{s['mean_margin']:>11.6f}±{s['std_margin']:.6f}")
    
    # Selection
    best_acc = max(s['mean_softmax_acc'] for s in summary.values())
    candidates = [ag for ag in alpha_gen_values 
                  if summary[ag]['mean_softmax_acc'] >= best_acc - 0.01]
    
    if len(candidates) == 1:
        selected = candidates[0]
        reason = f"Highest mean accuracy ({summary[selected]['mean_softmax_acc']*100:.2f}%)"
    else:
        # Tie-break: closest to 1e-5
        selected = min(candidates, key=lambda ag: abs(np.log10(ag) - np.log10(1e-5)))
        reason = (f"Tie-break among {[f'{c:.0e}' for c in candidates]} "
                  f"(all within 1pp of best). Selected closest to 1e-5.")
    
    print(f"\n{'='*70}")
    print(f"SELECTED: α_gen = {selected:.0e}")
    print(f"Reason: {reason}")
    print(f"{'='*70}")
    
    # Check energy margin at selected α_gen
    sel_margin = summary[selected]['mean_margin']
    if sel_margin < 1e-6:
        print("\n⚠ WARNING: Energy margin is near zero at selected α_gen.")
        print("  The generative chain may not be contributing to hypothesis")
        print("  discrimination. Consider using a higher α_gen.")
    
    # Check if accuracy-optimal differs from AUROC2-optimal
    auroc_best_ag = max(alpha_gen_values, 
                        key=lambda ag: summary[ag]['mean_probe_auroc2'])
    if auroc_best_ag != selected:
        print(f"\nNote: AUROC2-optimal α_gen ({auroc_best_ag:.0e}, "
              f"AUROC2={summary[auroc_best_ag]['mean_probe_auroc2']:.4f}) "
              f"differs from accuracy-optimal ({selected:.0e}). "
              f"Accuracy-based selection is pre-registered; AUROC2 reported "
              f"for transparency.")
    
    print(f"\nTotal sweep time: {sweep_time/3600:.1f} hours")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        'sweep_config': {
            'alpha_gen_values': alpha_gen_values,
            'seeds': seeds,
            'alpha_disc': 1.0,
            'epochs': 25,
            'T_train': 32,
            'T_eval': 100,
        },
        'per_run_results': all_results,
        'summary': {str(k): v for k, v in summary.items()},
        'selected_alpha_gen': selected,
        'selection_reason': reason,
        'total_time_s': sweep_time,
    }
    
    results_path = "results/calibration_sweep.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Write the selected value to a simple file for Stage 2/3 to read
    with open("results/selected_alpha_gen.txt", 'w') as f:
        f.write(f"{selected}")
    print(f"Selected α_gen written to results/selected_alpha_gen.txt")


if __name__ == "__main__":
    main()
