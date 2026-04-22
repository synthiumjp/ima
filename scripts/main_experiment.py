"""
bPC Main Experiment (Stage 3 of pre-registration)
====================================================

Pre-registration reference: Amendment 1, §3.1, §4.4, §4.5

Design: 3 conditions × 10 seeds (6-15) = 30 training runs.

Condition A (TinyConvPCN): Standard discriminative PC with CE.
  T=13 train/eval, [0,1] normalisation.

Condition B (TinyConvPCN-MSE): Standard discriminative PC with MSE.
  T=13 train/eval, [0,1] normalisation.

Condition C (TinyConvBPC): Bidirectional PC.
  T=32 train, T=100 eval, [-1,1] normalisation.
  α_gen from Stage 1 calibration, α_disc=1.0.

All conditions: AdamW lr=1e-4 wd=1e-4, SGD momentum=0.5 lr=5e-2,
batch_size=128, 25 epochs, eval on first 1280 test images.

Measured variables per network:
  - Softmax accuracy, probe accuracy
  - Softmax AUROC2, probe AUROC2
  - Delta = probe_AUROC2 - softmax_AUROC2
  - Energy margin (mean)
  - Per-layer latent movement
  - Logit norm, logit margin distributions
  - Per-layer energy decomposition (Condition C only)

No interim analyses. All 30 runs complete before any statistics.

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


# =============================================================================
# Evaluation functions
# =============================================================================

def evaluate_standard_pc(model, test_loader, device, n_batches=10,
                         T_eval=13, eta_h=5e-2, momentum_h=0.5,
                         condition_label="stdPC"):
    """Evaluate a standard PC or PC-MSE network.
    
    Uses classify_energy_based from spike_dynamics.py for the K-way probe.
    """
    from spike_dynamics import classify_energy_based
    
    model.eval()
    
    all_correct_softmax = []
    all_correct_probe = []
    all_softmax_conf = []
    all_energy_margins = []
    all_logit_norms = []
    all_logit_margins = []
    all_movements_per_layer = [[], [], []]  # 3 latent layers
    
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)
        
        # K-way energy probe
        pred_probe, energies = classify_energy_based(
            model, x, T=T_eval, eta_h=eta_h, momentum_h=momentum_h
        )
        
        # Energy margin
        sorted_e, _ = energies.sort(dim=1)
        energy_margin = sorted_e[:, 1] - sorted_e[:, 0]
        
        # Softmax from encoder
        with torch.no_grad():
            latents = model.forward_encoder(x)
            logits = latents[3]
            probs = F.softmax(logits, dim=-1)
            pred_softmax = logits.argmax(dim=-1)
            softmax_conf = probs.max(dim=-1).values
            
            logit_norms = logits.norm(dim=-1)
            sorted_logits, _ = logits.sort(dim=-1, descending=True)
            logit_margins_batch = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        # Latent movement (using true class for clamping)
        with torch.no_grad():
            latents_init = model.forward_encoder(x)
        
        h1_init = latents_init[0].clone()
        h2_init = latents_init[1].clone()
        h3_init = latents_init[2].clone()
        
        # Run inference with true label clamped
        y_onehot = F.one_hot(y, model.num_classes).float()
        h1 = h1_init.clone().requires_grad_(True)
        h2 = h2_init.clone().requires_grad_(True)
        h3 = h3_init.clone().requires_grad_(True)
        h4_clamped = y_onehot
        
        m1 = torch.zeros_like(h1)
        m2 = torch.zeros_like(h2)
        m3 = torch.zeros_like(h3)
        
        with torch.enable_grad():
            for t in range(T_eval):
                lat = [h1, h2, h3, h4_clamped]
                per_e, _, _ = model.total_energy(lat, y_onehot=y_onehot)
                e_scalar = per_e.sum()
                grads = torch.autograd.grad(e_scalar, [h1, h2, h3],
                                            create_graph=False)
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
        
        with torch.no_grad():
            all_movements_per_layer[0].append((h1 - h1_init).abs().mean().item())
            all_movements_per_layer[1].append((h2 - h2_init).abs().mean().item())
            all_movements_per_layer[2].append((h3 - h3_init).abs().mean().item())
        
        all_correct_softmax.append((pred_softmax == y).cpu())
        all_correct_probe.append((pred_probe == y).cpu())
        all_softmax_conf.append(softmax_conf.cpu())
        all_energy_margins.append(energy_margin.cpu())
        all_logit_norms.append(logit_norms.cpu())
        all_logit_margins.append(logit_margins_batch.cpu())
        
        print(f"    eval batch {batch_idx+1}/{n_batches}", end="\r")
    
    print()
    
    correct_softmax = torch.cat(all_correct_softmax)
    correct_probe = torch.cat(all_correct_probe)
    softmax_conf = torch.cat(all_softmax_conf)
    energy_margins = torch.cat(all_energy_margins)
    logit_norms = torch.cat(all_logit_norms)
    logit_margins = torch.cat(all_logit_margins)
    
    softmax_acc = correct_softmax.float().mean().item()
    probe_acc = correct_probe.float().mean().item()
    
    if correct_softmax.sum() > 0 and (~correct_softmax).sum() > 0:
        softmax_auroc2 = roc_auc_score(correct_softmax.numpy(), softmax_conf.numpy())
    else:
        softmax_auroc2 = float('nan')
    
    if correct_probe.sum() > 0 and (~correct_probe).sum() > 0:
        probe_auroc2 = roc_auc_score(correct_probe.numpy(), energy_margins.numpy())
    else:
        probe_auroc2 = float('nan')
    
    delta = probe_auroc2 - softmax_auroc2
    
    # Per-layer movement (mean across batches)
    movements = [np.mean(m) for m in all_movements_per_layer]
    max_movement = max(movements)
    
    return {
        'condition': condition_label,
        'softmax_acc': softmax_acc,
        'probe_acc': probe_acc,
        'softmax_auroc2': softmax_auroc2,
        'probe_auroc2': probe_auroc2,
        'delta': delta,
        'mean_energy_margin': energy_margins.mean().item(),
        'movements': movements,
        'max_movement': max_movement,
        'logit_norm_mean': logit_norms.mean().item(),
        'logit_norm_std': logit_norms.std().item(),
        'logit_margin_mean': logit_margins.mean().item(),
        'logit_margin_std': logit_margins.std().item(),
    }


def evaluate_bpc(model, test_loader, device, n_batches=10,
                 T_eval=100, eta_h=5e-2, momentum_h=0.5):
    """Evaluate a trained bPC network with full diagnostics."""
    from tiny_conv_bpc import bpc_classify_energy_based, measure_latent_movement
    
    model.eval()
    
    all_correct_softmax = []
    all_correct_probe = []
    all_softmax_conf = []
    all_energy_margins = []
    all_logit_norms = []
    all_logit_margins = []
    all_gen_energies = []
    all_disc_energies = []
    all_movements_per_layer = [[], [], []]
    
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)
        
        # K-way energy probe with decomposition
        pred_probe, energies, gen_e, disc_e = bpc_classify_energy_based(
            model, x, T=T_eval, eta_h=eta_h, momentum_h=momentum_h
        )
        
        sorted_e, _ = energies.sort(dim=1)
        energy_margin = sorted_e[:, 1] - sorted_e[:, 0]
        
        # Softmax from V pathway
        with torch.no_grad():
            latents = model.forward_v(x)
            logits = latents[3]
            probs = F.softmax(logits, dim=-1)
            pred_softmax = logits.argmax(dim=-1)
            softmax_conf = probs.max(dim=-1).values
            
            logit_norms = logits.norm(dim=-1)
            sorted_logits, _ = logits.sort(dim=-1, descending=True)
            logit_margins_batch = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        # Latent movement (using V-pathway argmax for clamping)
        movements, max_mov = measure_latent_movement(
            model, x, T=T_eval, eta_h=eta_h, momentum_h=momentum_h
        )
        for l in range(3):
            all_movements_per_layer[l].append(movements[l])
        
        all_correct_softmax.append((pred_softmax == y).cpu())
        all_correct_probe.append((pred_probe == y).cpu())
        all_softmax_conf.append(softmax_conf.cpu())
        all_energy_margins.append(energy_margin.cpu())
        all_logit_norms.append(logit_norms.cpu())
        all_logit_margins.append(logit_margins_batch.cpu())
        all_gen_energies.append(gen_e.cpu())
        all_disc_energies.append(disc_e.cpu())
        
        print(f"    eval batch {batch_idx+1}/{n_batches}", end="\r")
    
    print()
    
    correct_softmax = torch.cat(all_correct_softmax)
    correct_probe = torch.cat(all_correct_probe)
    softmax_conf = torch.cat(all_softmax_conf)
    energy_margins = torch.cat(all_energy_margins)
    logit_norms = torch.cat(all_logit_norms)
    logit_margins = torch.cat(all_logit_margins)
    gen_energies = torch.cat(all_gen_energies)
    disc_energies = torch.cat(all_disc_energies)
    
    softmax_acc = correct_softmax.float().mean().item()
    probe_acc = correct_probe.float().mean().item()
    
    if correct_softmax.sum() > 0 and (~correct_softmax).sum() > 0:
        softmax_auroc2 = roc_auc_score(correct_softmax.numpy(), softmax_conf.numpy())
    else:
        softmax_auroc2 = float('nan')
    
    if correct_probe.sum() > 0 and (~correct_probe).sum() > 0:
        probe_auroc2 = roc_auc_score(correct_probe.numpy(), energy_margins.numpy())
    else:
        probe_auroc2 = float('nan')
    
    delta = probe_auroc2 - softmax_auroc2
    
    movements = [np.mean(m) for m in all_movements_per_layer]
    max_movement = max(movements)
    
    return {
        'condition': 'bPC',
        'softmax_acc': softmax_acc,
        'probe_acc': probe_acc,
        'softmax_auroc2': softmax_auroc2,
        'probe_auroc2': probe_auroc2,
        'delta': delta,
        'mean_energy_margin': energy_margins.mean().item(),
        'movements': movements,
        'max_movement': max_movement,
        'logit_norm_mean': logit_norms.mean().item(),
        'logit_norm_std': logit_norms.std().item(),
        'logit_margin_mean': logit_margins.mean().item(),
        'logit_margin_std': logit_margins.std().item(),
        'gen_energy_mean': gen_energies.mean().item(),
        'disc_energy_mean': disc_energies.mean().item(),
        'gen_disc_ratio': (gen_energies.mean() / (disc_energies.mean() + 1e-10)).item(),
    }


# =============================================================================
# Training functions
# =============================================================================

def train_condition_a(seed, device, data_dir="data"):
    """Condition A: Standard PC with CE."""
    from spike_dynamics import TinyConvPCN, train_step
    from cifar10_data import get_data_loaders
    
    set_all_seeds(seed)
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=128, num_workers=0)
    model = TinyConvPCN(num_classes=10).to(device)
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    start = time.time()
    model.train()
    for epoch in range(25):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim_w.zero_grad()
            total_loss, gen_l, enc_l, read_l = train_step(
                model, x, y, T=13, eta_h=5e-2, momentum_h=0.5
            )
            total_loss.backward()
            optim_w.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        elapsed = time.time() - start
        print(f"  epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} ({elapsed:.0f}s)")
    
    print("  Evaluating...")
    results = evaluate_standard_pc(
        model, test_loader, device, n_batches=10,
        T_eval=13, condition_label="stdPC"
    )
    results['seed'] = seed
    results['train_time_s'] = time.time() - start
    return results


def train_condition_b(seed, device, data_dir="data"):
    """Condition B: Standard PC with MSE."""
    from tiny_conv_pcn_mse import TinyConvPCN_MSE, train_step_mse
    from cifar10_data import get_data_loaders
    
    set_all_seeds(seed)
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=128, num_workers=0)
    model = TinyConvPCN_MSE(num_classes=10).to(device)
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    start = time.time()
    model.train()
    for epoch in range(25):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim_w.zero_grad()
            total_loss, gen_l, enc_l, read_l = train_step_mse(
                model, x, y, T=13, eta_h=5e-2, momentum_h=0.5
            )
            total_loss.backward()
            optim_w.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        elapsed = time.time() - start
        print(f"  epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} ({elapsed:.0f}s)")
    
    print("  Evaluating...")
    # MSE model has same interface as standard PC for evaluation
    results = evaluate_standard_pc(
        model, test_loader, device, n_batches=10,
        T_eval=13, condition_label="MSE"
    )
    results['seed'] = seed
    results['train_time_s'] = time.time() - start
    return results


def train_condition_c(seed, alpha_gen, device, data_dir="data"):
    """Condition C: Bidirectional PC."""
    from tiny_conv_bpc import TinyConvBPC, bpc_train_step
    from cifar10_data_bpc import get_data_loaders_bpc
    
    set_all_seeds(seed)
    train_loader, test_loader = get_data_loaders_bpc(data_dir, batch_size=128, num_workers=0)
    model = TinyConvBPC(num_classes=10, alpha_gen=alpha_gen, alpha_disc=1.0).to(device)
    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    start = time.time()
    model.train()
    for epoch in range(25):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim_w.zero_grad()
            total_loss, gen_l, disc_l, enc_l = bpc_train_step(
                model, x, y, T=32, eta_h=5e-2, momentum_h=0.5
            )
            total_loss.backward()
            optim_w.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        elapsed = time.time() - start
        print(f"  epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} ({elapsed:.0f}s)")
    
    print("  Evaluating (T_eval=100)...")
    results = evaluate_bpc(
        model, test_loader, device, n_batches=10,
        T_eval=100
    )
    results['seed'] = seed
    results['alpha_gen'] = alpha_gen
    results['train_time_s'] = time.time() - start
    return results


# =============================================================================
# Main experiment
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("bPC MAIN EXPERIMENT (Stage 3)")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    
    # Read selected alpha_gen
    alpha_gen_path = "results/selected_alpha_gen.txt"
    if not os.path.exists(alpha_gen_path):
        print(f"ERROR: {alpha_gen_path} not found. Run Stage 1 first.")
        return
    with open(alpha_gen_path, 'r') as f:
        alpha_gen = float(f.read().strip())
    
    print(f"Selected α_gen: {alpha_gen:.0e}")
    print(f"Seeds: 6-15")
    print(f"Conditions: A (stdPC-CE), B (stdPC-MSE), C (bPC)")
    print(f"Epochs: 25 per run")
    print(f"Total runs: 30")
    print()
    
    seeds = list(range(6, 16))
    
    # Resume support
    os.makedirs("results", exist_ok=True)
    checkpoint_path = "results/main_experiment_checkpoint.json"
    all_results = []
    completed = set()
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            all_results = json.load(f)
        for r in all_results:
            completed.add((r['condition'], r['seed']))
        print(f"Resumed from checkpoint: {len(completed)} runs already completed.")
    
    total_runs = 30
    run_count = len(completed)
    
    experiment_start = time.time()
    
    # Run all conditions for each seed (seed-major ordering)
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        
        # Condition A: Standard PC
        if ('stdPC', seed) not in completed:
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs}: Condition A (stdPC-CE), seed={seed} ---")
            results = train_condition_a(seed, device)
            all_results.append(results)
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Delta={results['delta']:+.4f} "
                  f"softmax_acc={results['softmax_acc']*100:.1f}% "
                  f"probe_AUROC2={results['probe_auroc2']:.4f} "
                  f"softmax_AUROC2={results['softmax_auroc2']:.4f}")
            print(f"  [checkpoint saved: {len(all_results)}/{total_runs}]")
        else:
            print(f"\n  Condition A, seed={seed} — SKIPPED (already done)")
        
        # Condition B: MSE
        if ('MSE', seed) not in completed:
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs}: Condition B (stdPC-MSE), seed={seed} ---")
            results = train_condition_b(seed, device)
            all_results.append(results)
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Delta={results['delta']:+.4f} "
                  f"softmax_acc={results['softmax_acc']*100:.1f}% "
                  f"probe_AUROC2={results['probe_auroc2']:.4f} "
                  f"softmax_AUROC2={results['softmax_auroc2']:.4f}")
            print(f"  [checkpoint saved: {len(all_results)}/{total_runs}]")
        else:
            print(f"\n  Condition B, seed={seed} — SKIPPED (already done)")
        
        # Condition C: bPC
        if ('bPC', seed) not in completed:
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs}: Condition C (bPC), seed={seed} ---")
            results = train_condition_c(seed, alpha_gen, device)
            all_results.append(results)
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Delta={results['delta']:+.4f} "
                  f"softmax_acc={results['softmax_acc']*100:.1f}% "
                  f"probe_AUROC2={results['probe_auroc2']:.4f} "
                  f"softmax_AUROC2={results['softmax_auroc2']:.4f}")
            print(f"  [checkpoint saved: {len(all_results)}/{total_runs}]")
        else:
            print(f"\n  Condition C, seed={seed} — SKIPPED (already done)")
    
    total_time = time.time() - experiment_start
    
    # =========================================================================
    # Raw results summary (NO statistical tests — those go in analysis script)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RAW RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("NOTE: No statistical tests computed here. All hypothesis tests")
    print("are in the pre-registered analysis script (Stage 4).")
    print()
    
    # Per-seed table
    print(f"{'Seed':>4} {'Cond':>5} {'SoftAcc':>8} {'ProbeAcc':>9} "
          f"{'SoftAUROC':>10} {'ProbeAUROC':>11} {'Delta':>8} "
          f"{'MaxMov':>8} {'LogitNorm':>10}")
    print("-" * 85)
    
    for seed in seeds:
        for cond in ['stdPC', 'MSE', 'bPC']:
            runs = [r for r in all_results 
                    if r['condition'] == cond and r['seed'] == seed]
            if runs:
                r = runs[0]
                print(f"{seed:>4} {cond:>5} "
                      f"{r['softmax_acc']*100:>7.1f}% "
                      f"{r['probe_acc']*100:>8.1f}% "
                      f"{r['softmax_auroc2']:>10.4f} "
                      f"{r['probe_auroc2']:>11.4f} "
                      f"{r['delta']:>+8.4f} "
                      f"{r['max_movement']:>8.4f} "
                      f"{r.get('logit_norm_mean', 0):>10.4f}")
        print()
    
    # Condition means
    print()
    print("Condition means:")
    print(f"{'Cond':>5} {'SoftAcc':>8} {'ProbeAcc':>9} "
          f"{'SoftAUROC':>10} {'ProbeAUROC':>11} {'Delta':>8}")
    print("-" * 55)
    for cond in ['stdPC', 'MSE', 'bPC']:
        runs = [r for r in all_results if r['condition'] == cond]
        if runs:
            print(f"{cond:>5} "
                  f"{np.mean([r['softmax_acc'] for r in runs])*100:>7.1f}% "
                  f"{np.mean([r['probe_acc'] for r in runs])*100:>8.1f}% "
                  f"{np.mean([r['softmax_auroc2'] for r in runs]):>10.4f} "
                  f"{np.mean([r['probe_auroc2'] for r in runs]):>11.4f} "
                  f"{np.mean([r['delta'] for r in runs]):>+8.4f}")
    
    print(f"\nTotal experiment time: {total_time/3600:.1f} hours")
    
    # Save final results
    output = {
        'experiment_config': {
            'seeds': seeds,
            'alpha_gen': alpha_gen,
            'alpha_disc': 1.0,
            'epochs': 25,
            'conditions': {
                'A': {'name': 'stdPC', 'T_train': 13, 'T_eval': 13, 'norm': '[0,1]'},
                'B': {'name': 'MSE', 'T_train': 13, 'T_eval': 13, 'norm': '[0,1]'},
                'C': {'name': 'bPC', 'T_train': 32, 'T_eval': 100, 'norm': '[-1,1]'},
            },
        },
        'per_run_results': all_results,
        'total_time_s': total_time,
    }
    
    results_path = "results/main_experiment.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print("Proceed to Stage 4 (analysis) with the pre-registered script.")


if __name__ == "__main__":
    main()
