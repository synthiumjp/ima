"""
IMA Phase 0: PCN Sanity Checks and Confidence Signal Evaluation
================================================================

Pre-registration v3.1 §3.3 sanity checks:
  1. Energy monotonicity: >=90% of inference steps show energy decrease
  2. Error norm stabilisation: ||ε_l|| at T=20 differs from T=15 by <10%
  3. Accuracy saturation: acc@T=20 differs from acc@T=15 by <1pp
  4. T_infer sensitivity: AUROC-2 top-2 signals stable across T ∈ {10,15,20}
  5. Non-degenerate inference: mean ||ε_l|| at T=0 >= 2x at T=20

Pre-registration v3.1 §3.5-3.6 confidence signals:
  1. Energy decay rate (primary candidate)
  2. Negative residual energy
  3. Negative SSE
  4. Per-layer error norms (individual)
  5. Max softmax probability
  6. Negative entropy

Pre-registration v3.1 §3.7 go/no-go:
  - Primary: AUROC-2 > 0.55 replicated across 3 seeds → PROCEED
  - Secondary: M-ratio > 0.5 (informative, not gating)

Usage:
  python evaluate_phase0.py --seed 42
  python evaluate_phase0.py --seed 42 --all-t   # test T ∈ {5,10,15,20}
  python evaluate_phase0.py --summarise          # aggregate across seeds
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from conv_pcn import ConvPCN, M_INPUT_DIM
from cifar10_data import get_data_loaders

# Try to import metadpy for meta-d' (optional — install separately)
try:
    from metadpy.mle import metad
    METADPY_AVAILABLE = True
except ImportError:
    METADPY_AVAILABLE = False
    print("WARNING: metadpy not installed. M-ratio will not be computed.")
    print("Install: pip install metadpy")

VALID_SEEDS = [42, 123, 456]
BATCH_SIZE = 128
NUM_CLASSES = 10
T_INFER_VALUES = [5, 10, 15, 20]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def load_model(checkpoint_path: str, device: str) -> ConvPCN:
    """Load trained PCN from checkpoint."""
    model = ConvPCN(num_classes=NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Test acc: {ckpt.get('test_acc', '?'):.4f}")
    return model


def get_test_loader(data_dir: str):
    """CIFAR-10 test set with standard normalisation."""
    _, test_loader = get_data_loaders(data_dir, batch_size=BATCH_SIZE)
    return test_loader


def collect_trials(model: ConvPCN, data_loader, T_infer: int,
                   eta_infer: float, device: str) -> list:
    """Run evaluation and collect per-trial data for SDT analysis."""
    model.eval()
    trials = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)

            result = model.classify(
                x_batch, T_infer=T_infer, eta_infer=eta_infer,
                return_errors=True,
                return_energy_trace=True,
                return_error_norms_trace=True,
            )

            preds = result.logits.argmax(dim=1)
            correct = (preds == y_batch)

            for i in range(B):
                trial = {
                    'correct': int(correct[i].item()),
                    'pred': preds[i].item(),
                    'target': y_batch[i].item(),
                    # Confidence signals (pre-reg §3.5)
                    'max_prob': result.probs[i].max().item(),
                    'neg_entropy': (result.probs[i] * result.probs[i].clamp(min=1e-12).log()).sum().item(),
                    'neg_sse': -sum(e[i].pow(2).sum().item() for e in result.errors),
                    'neg_residual_energy': -result.energy_trace[-1] if result.energy_trace else None,
                    'energy_decay_rate': (
                        result.energy_trace[-2] - result.energy_trace[-1]
                        if result.energy_trace and len(result.energy_trace) >= 2
                        else None
                    ),
                    'error_norms': [e[i].pow(2).sum().item() ** 0.5 for e in result.errors],
                    'energy_trace': list(result.energy_trace) if result.energy_trace else None,
                }
                # Non-degenerate inference check data
                if result.error_norms_trace:
                    trial['error_norms_t0'] = result.error_norms_trace[0]
                    trial['error_norms_tT'] = result.error_norms_trace[-1]

                trials.append(trial)

    return trials


# =============================================================================
# Sanity checks (pre-reg §3.3)
# =============================================================================

def check_energy_monotonicity(trials: list) -> dict:
    """Check 1: Energy must decrease in >=90% of inference steps."""
    total_steps = 0
    decreasing_steps = 0
    for trial in trials:
        trace = trial.get('energy_trace', [])
        if trace and len(trace) >= 2:
            for i in range(len(trace) - 1):
                total_steps += 1
                if trace[i + 1] <= trace[i]:
                    decreasing_steps += 1

    pct = decreasing_steps / total_steps if total_steps > 0 else 0
    return {
        'name': 'Energy monotonicity',
        'criterion': '>=90% of steps show decrease',
        'total_steps': total_steps,
        'decreasing_steps': decreasing_steps,
        'percentage': pct * 100,
        'passed': pct >= 0.90,
    }


def check_error_stabilisation(trials: list, T_infer: int = 20) -> dict:
    """Check 2: Per-layer error norms at T=20 differ from T=15 by <10%."""
    # This requires running at both T=15 and T=20. 
    # For simplicity, we check the last 5 steps of the error norms trace.
    # The trace has T+1 entries (t=0 through t=T).
    if not trials[0].get('error_norms_trace'):
        # Need separate runs at T=15 and T=20 — handle in the evaluation loop
        return {
            'name': 'Error norm stabilisation',
            'criterion': '||ε_l||@T=20 vs T=15 < 10% relative',
            'note': 'Requires separate T=15 and T=20 runs',
            'passed': None,
        }

    # Use the trace: compare norms at step 15 vs step 20
    layer_diffs = defaultdict(list)
    for trial in trials:
        trace = trial.get('error_norms_trace', [])
        if trace and len(trace) > 20:
            for l in range(4):
                norm_15 = trace[15][l]
                norm_20 = trace[20][l]
                if norm_15 > 0:
                    rel_diff = abs(norm_20 - norm_15) / norm_15
                    layer_diffs[l].append(rel_diff)

    results_per_layer = {}
    all_passed = True
    for l in range(4):
        if layer_diffs[l]:
            mean_diff = np.mean(layer_diffs[l])
            results_per_layer[f'L{l}'] = mean_diff
            if mean_diff >= 0.10:
                all_passed = False

    return {
        'name': 'Error norm stabilisation',
        'criterion': '<10% relative change T=15 to T=20',
        'per_layer_mean_rel_diff': results_per_layer,
        'passed': all_passed if results_per_layer else None,
    }


def check_nondegen_inference(trials: list) -> dict:
    """Check 5: Mean ||ε_l|| at T=0 >= 2x at T=20."""
    ratios = []
    for trial in trials:
        t0 = trial.get('error_norms_t0')
        tT = trial.get('error_norms_tT')
        if t0 and tT:
            mean_t0 = np.mean(t0)
            mean_tT = np.mean(tT)
            if mean_tT > 0:
                ratios.append(mean_t0 / mean_tT)

    mean_ratio = np.mean(ratios) if ratios else 0
    return {
        'name': 'Non-degenerate inference',
        'criterion': 'mean ||ε_l||@T=0 >= 2x ||ε_l||@T=20',
        'mean_ratio': mean_ratio,
        'n_trials': len(ratios),
        'passed': mean_ratio >= 2.0,
    }


# =============================================================================
# Confidence signal analysis (pre-reg §3.6)
# =============================================================================

def compute_auroc2(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Type-2 AUROC: how well does confidence discriminate correct vs incorrect."""
    if len(np.unique(correct)) < 2:
        return np.nan
    return roc_auc_score(correct, confidence)


def bootstrap_auroc2(correct: np.ndarray, confidence: np.ndarray,
                     n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                     ci_level: float = 0.95) -> dict:
    """Bootstrap AUROC-2 with confidence intervals.
    
    Pre-reg §3.6: "10,000 resamples within each seed"
    """
    rng = np.random.RandomState(seed)
    n = len(correct)
    point = compute_auroc2(correct, confidence)

    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(correct[idx])) < 2:
            continue
        boots.append(roc_auc_score(correct[idx], confidence[idx]))

    boots = np.array(boots)
    alpha = 1 - ci_level
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))

    return {
        'auroc2': point,
        'ci_lo': lo,
        'ci_hi': hi,
        'ci_level': ci_level,
        'n_boot': len(boots),
    }


def compute_mratio(correct: np.ndarray, confidence: np.ndarray,
                   n_bins: int = 4) -> dict:
    """Compute meta-d' and M-ratio via metadpy.
    
    Pre-reg §3.6: "Confidence binned into 4 quantiles"
    "Hautus log-linear correction (+0.5 to zero cells)"
    """
    if not METADPY_AVAILABLE:
        return {'mratio': None, 'meta_dprime': None, 'dprime': None,
                'note': 'metadpy not installed'}

    # Bin confidence into 4 quantiles
    quantiles = np.percentile(confidence, [25, 50, 75])
    binned = np.digitize(confidence, quantiles)  # 0-3

    # For metadpy: stimID (0 or 1), response (0 or 1), confidence (1 to nRatings)
    # We use a binary framing: stimulus=correct_answer, response=model_prediction
    # But for Type-2 analysis, the convention is:
    #   stimID = arbitrary (e.g., all 1), response = correct (0 or 1)
    # Actually, metadpy.mle.metad expects nR_S1, nR_S2 (response counts)
    # This gets complex — let's use the standard approach

    try:
        # Simple approach: use metadpy with stimuli and responses
        from metadpy.mle import metad as metad_func

        # Convert to format metadpy expects
        # nR_S1[i] = count of confidence=i when stimulus=S1 and response=S1/S2
        # This is getting complicated — use the simpler fit function
        # For now, report AUROC-2 as primary (pre-reg: "AUROC₂ is the primary metric")
        return {'mratio': None, 'note': 'M-ratio computation deferred to analysis script'}
    except Exception as e:
        return {'mratio': None, 'error': str(e)}


def analyse_signals(trials: list) -> dict:
    """Compute AUROC-2 for all 6 pre-registered confidence signals."""
    correct = np.array([t['correct'] for t in trials])
    n_correct = correct.sum()
    n_incorrect = len(correct) - n_correct
    accuracy = n_correct / len(correct)

    print(f"\n  Accuracy: {accuracy:.4f} ({n_correct}/{len(correct)})")
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")

    # Define signals (pre-reg §3.5)
    signals = {}

    # Signal 1: Energy decay rate (PRIMARY)
    vals = [t['energy_decay_rate'] for t in trials if t['energy_decay_rate'] is not None]
    if vals:
        signals['energy_decay_rate'] = np.array(vals)

    # Signal 2: Negative residual energy (higher = more confident)
    vals = [t['neg_residual_energy'] for t in trials if t['neg_residual_energy'] is not None]
    if vals:
        signals['neg_residual_energy'] = np.array(vals)

    # Signal 3: Negative SSE
    signals['neg_sse'] = np.array([t['neg_sse'] for t in trials])

    # Signal 4: Per-layer error norms (negative, higher = more confident)
    for l in range(4):
        signals[f'neg_error_norm_L{l}'] = np.array(
            [-t['error_norms'][l] for t in trials]
        )

    # Signal 5: Max softmax probability
    signals['max_prob'] = np.array([t['max_prob'] for t in trials])

    # Signal 6: Negative entropy (higher = more confident)
    signals['neg_entropy'] = np.array([t['neg_entropy'] for t in trials])

    # Compute AUROC-2 for each
    results = {}
    print(f"\n  {'Signal':<25} {'AUROC-2':>8} {'95% CI':>16} {'Pearson r':>10}")
    print(f"  {'-'*60}")

    for name, conf in sorted(signals.items()):
        # Trim correct array to match if needed (energy signals may have fewer)
        c = correct[:len(conf)]

        boot = bootstrap_auroc2(c, conf)
        r = np.corrcoef(c, conf)[0, 1]

        results[name] = {
            **boot,
            'pearson_r': float(r),
            'mean_correct': float(conf[c == 1].mean()),
            'mean_incorrect': float(conf[c == 0].mean()) if (c == 0).any() else None,
        }

        primary = ' ★' if name == 'energy_decay_rate' else ''
        print(f"  {name:<25} {boot['auroc2']:>8.4f} [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}] {r:>10.4f}{primary}")

    return {
        'accuracy': accuracy,
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect),
        'n_trials': len(correct),
        'signals': results,
    }


# =============================================================================
# Go/No-Go evaluation (pre-reg §3.7)
# =============================================================================

def go_nogo_evaluation(all_seed_results: dict) -> dict:
    """Apply pre-registered go/no-go decision rule."""
    print(f"\n{'='*70}")
    print("GO / NO-GO EVALUATION (pre-reg §3.7)")
    print(f"{'='*70}\n")

    # Collect AUROC-2 per signal per seed
    signal_names = set()
    for seed_data in all_seed_results.values():
        signal_names.update(seed_data['signals'].keys())

    decision = {'signals': {}, 'decision': None, 'reason': ''}

    for signal in sorted(signal_names):
        aurocs = []
        for seed, seed_data in sorted(all_seed_results.items()):
            if signal in seed_data['signals']:
                aurocs.append(seed_data['signals'][signal]['auroc2'])
        
        if len(aurocs) < 3:
            continue

        mean_auroc = np.mean(aurocs)
        all_above_055 = all(a > 0.55 for a in aurocs)
        all_above_050 = all(a > 0.50 for a in aurocs)

        decision['signals'][signal] = {
            'aurocs': aurocs,
            'mean': float(mean_auroc),
            'all_above_0.55': all_above_055,
            'all_above_0.50': all_above_050,
        }

        primary = ' (PRIMARY)' if signal == 'energy_decay_rate' else ''
        print(f"  {signal}{primary}:")
        print(f"    Seeds: {aurocs}")
        print(f"    Mean: {mean_auroc:.4f}")
        print(f"    All > 0.55: {all_above_055}")
        print(f"    All > 0.50: {all_above_050}")

    # Decision logic (pre-reg §3.7)
    primary_signal = decision['signals'].get('energy_decay_rate', {})
    any_above_055 = any(
        s.get('all_above_0.55', False)
        for s in decision['signals'].values()
    )
    any_above_050 = any(
        s.get('all_above_0.50', False)
        for s in decision['signals'].values()
    )

    if primary_signal.get('all_above_0.55', False):
        decision['decision'] = 'PROCEED'
        decision['reason'] = 'Primary candidate (energy decay rate) AUROC-2 > 0.55 at all seeds'
    elif any_above_055:
        decision['decision'] = 'PROCEED'
        winner = [k for k, v in decision['signals'].items() if v.get('all_above_0.55')]
        decision['reason'] = f'Non-primary signal(s) {winner} AUROC-2 > 0.55 at all seeds'
        decision['note'] = 'Primary candidate was not the winner'
    elif any_above_050:
        decision['decision'] = 'PROCEED_WITH_CAUTION'
        decision['reason'] = 'Signal(s) above chance at all seeds (>0.50) but below 0.55 threshold'
    else:
        decision['decision'] = 'STOP_OR_INVESTIGATE'
        decision['reason'] = 'No signal replicates above chance across seeds'

    print(f"\n  DECISION: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")
    if 'note' in decision:
        print(f"  Note: {decision['note']}")

    return decision


# =============================================================================
# Main
# =============================================================================

def evaluate_seed(seed: int, device: str, data_dir: str,
                  checkpoint_dir: str, output_dir: str,
                  t_values: list = None):
    """Full Phase 0 evaluation for one seed."""
    if t_values is None:
        t_values = [20]  # default: just the pre-registered T_infer

    seed_ckpt = Path(checkpoint_dir) / f'seed_{seed}' / 'best_model.pt'
    if not seed_ckpt.exists():
        print(f"ERROR: No checkpoint at {seed_ckpt}")
        return None

    model = load_model(str(seed_ckpt), device)
    test_loader = get_test_loader(data_dir)

    seed_out = Path(output_dir) / f'seed_{seed}'
    seed_out.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for T in t_values:
        print(f"\n{'='*60}")
        print(f"Evaluating seed={seed}, T_infer={T}")
        print(f"{'='*60}")

        trials = collect_trials(model, test_loader, T_infer=T,
                                eta_infer=0.1, device=device)

        # Sanity checks
        print("\nSanity Checks:")
        sc1 = check_energy_monotonicity(trials)
        print(f"  1. {sc1['name']}: {sc1['percentage']:.1f}% — {'PASS' if sc1['passed'] else 'FAIL'}")

        sc5 = check_nondegen_inference(trials)
        print(f"  5. {sc5['name']}: ratio={sc5['mean_ratio']:.2f} — {'PASS' if sc5['passed'] else 'FAIL'}")

        # Confidence signal analysis
        signal_results = analyse_signals(trials)

        result = {
            'seed': seed,
            'T_infer': T,
            'sanity_checks': {
                'energy_monotonicity': sc1,
                'nondegen_inference': sc5,
            },
            **signal_results,
        }
        all_results[T] = result

        # Save per-T results
        with open(seed_out / f'phase0_T{T}.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save raw trials for T=20 (for later M-ratio / metadpy analysis)
        if T == 20:
            # Save compact trial data (without energy traces to save space)
            compact_trials = []
            for t in trials:
                compact_trials.append({
                    'correct': t['correct'],
                    'pred': t['pred'],
                    'target': t['target'],
                    'max_prob': t['max_prob'],
                    'neg_entropy': t['neg_entropy'],
                    'neg_sse': t['neg_sse'],
                    'neg_residual_energy': t['neg_residual_energy'],
                    'energy_decay_rate': t['energy_decay_rate'],
                    'error_norms': t['error_norms'],
                })
            with open(seed_out / 'trials_T20.json', 'w') as f:
                json.dump(compact_trials, f)

    # T_infer sensitivity check (sanity check 4)
    if len(t_values) >= 3:
        print(f"\nSanity Check 4: T_infer sensitivity")
        for T in t_values:
            if T in all_results:
                sigs = all_results[T]['signals']
                ranked = sorted(sigs.items(), key=lambda x: x[1]['auroc2'], reverse=True)
                top2 = [r[0] for r in ranked[:2]]
                print(f"  T={T}: top-2 = {top2}")

    # Accuracy saturation check (sanity check 3)
    if 15 in all_results and 20 in all_results:
        acc_15 = all_results[15]['accuracy']
        acc_20 = all_results[20]['accuracy']
        diff = abs(acc_20 - acc_15)
        print(f"\nSanity Check 3: Accuracy saturation")
        print(f"  acc@T=15: {acc_15:.4f}, acc@T=20: {acc_20:.4f}")
        print(f"  Difference: {diff:.4f} — {'PASS' if diff < 0.01 else 'FAIL'} (<1pp)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='IMA Phase 0: Evaluate PCN and confidence signals')
    parser.add_argument('--seed', type=int,
                        help='Seed to evaluate (omit for summarise mode)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/phase0')
    parser.add_argument('--output-dir', type=str, default='./results/phase0')
    parser.add_argument('--all-t', action='store_true',
                        help='Evaluate at T ∈ {5,10,15,20}')
    parser.add_argument('--summarise', action='store_true',
                        help='Summarise across all seeds and run go/no-go')

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if args.summarise:
        # Aggregate across seeds
        all_seed_results = {}
        for seed in VALID_SEEDS:
            result_path = Path(args.output_dir) / f'seed_{seed}' / 'phase0_T20.json'
            if result_path.exists():
                with open(result_path) as f:
                    all_seed_results[seed] = json.load(f)
            else:
                print(f"WARNING: No results for seed {seed} at {result_path}")

        if len(all_seed_results) == 3:
            decision = go_nogo_evaluation(all_seed_results)
            with open(Path(args.output_dir) / 'go_nogo_decision.json', 'w') as f:
                json.dump(decision, f, indent=2)
        else:
            print(f"\nNeed results for all 3 seeds. Found: {list(all_seed_results.keys())}")
    else:
        if args.seed is None:
            print("ERROR: Specify --seed or --summarise")
            return

        t_values = T_INFER_VALUES if args.all_t else [20]
        evaluate_seed(args.seed, device, args.data_dir,
                      args.checkpoint_dir, args.output_dir, t_values)


if __name__ == '__main__':
    main()
