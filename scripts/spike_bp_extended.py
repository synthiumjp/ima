"""
IMA matched-budget BP training spike (11 Apr 2026, session 4)

Goal: determine whether BP-trained TinyFFN at 25 epochs (matching the
extended PC spike training budget) produces softmax AUROC2 comparable to
PC's 0.8712 at 25 epochs, or whether BP plateaus meaningfully below.

This is the critical control for Finding A:
  "PC softmax AUROC2 at 25 epochs (0.8712) is substantially above
   BP softmax AUROC2 at 5 epochs (0.7765)"

The question: is this gap real (PC training produces better-calibrated
classifiers) or an artefact of the training budget mismatch?

Three outcomes:

OUTCOME MATCH: BP 25-epoch softmax AUROC2 ~ PC 25-epoch softmax AUROC2
  (i.e., both around 0.87). The 0.1 gap at 5 epochs was a training
  budget artefact. PC and BP produce equally well-calibrated softmax
  readouts at matched budget. Finding A evaporates.

OUTCOME PC-WINS: BP 25-epoch softmax AUROC2 plateaus meaningfully below
  PC (e.g., BP at 0.80-0.83 vs PC at 0.87). PC training produces
  better-calibrated classifiers at matched budget. Finding A is real.
  This becomes the core of Paper A.

OUTCOME BP-WINS: BP 25-epoch softmax AUROC2 actually exceeds PC.
  BP is simply better at this scale. PC has no advantage. Null
  programme.

Uses same TinyFFN architecture, same optimizer, same eval set, same
checkpoints (5, 10, 15, 20, 25) as the extended PC spike.
"""
# --- CUDNN WORKAROUND ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------

import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from cifar10_data import get_data_loaders

# Import TinyFFN from the BP+decoder spike
import importlib.util
spec = importlib.util.spec_from_file_location("spike_bp_decoder", "scripts/spike_bp_decoder.py")
bp_spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp_spike)

TinyFFN = bp_spike.TinyFFN


@torch.no_grad()
def evaluate_bp(model, test_loader, device, n_batches=10):
    """Standard softmax eval on a BP-trained encoder."""
    model.eval()
    correct_list = []
    conf_list = []
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        max_p = probs.max(dim=-1).values
        correct_list.append((pred == y).cpu())
        conf_list.append(max_p.cpu())

    correct = torch.cat(correct_list)
    conf = torch.cat(conf_list)
    acc = correct.float().mean().item()
    if correct.sum() > 0 and (~correct).sum() > 0:
        auroc = roc_auc_score(correct.numpy(), conf.numpy())
    else:
        auroc = float('nan')
    return acc, auroc


def main():
    device = "cuda"
    torch.manual_seed(42)

    print("=" * 60)
    print("MATCHED-BUDGET BP TRAINING SPIKE (25 epochs)")
    print("=" * 60)
    print("Goal: determine whether PC's apparent softmax AUROC2 advantage")
    print("      (0.8712 at ep25) survives matched training budget vs BP.")
    print()
    print("PC reference (from extended PC spike):")
    print("  ep5:  acc=69.8% softmax_AUROC2=0.8020")
    print("  ep10: acc=73.9% softmax_AUROC2=0.8150")
    print("  ep15: acc=77.0% softmax_AUROC2=0.8319")
    print("  ep20: acc=77.7% softmax_AUROC2=0.8513")
    print("  ep25: acc=78.2% softmax_AUROC2=0.8712")
    print()

    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # Model: same architecture as TinyFFN in BP+decoder spike
    model = TinyFFN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # Same optimizer as PC's weight optimizer (AdamW, lr=1e-4, wd=1e-4)
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    checkpoints = [5, 10, 15, 20, 25]
    results = {}

    print()
    print("Training BP TinyFFN for 25 epochs...")
    start = time.time()
    for epoch in range(25):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        train_acc = total_correct / total
        elapsed = time.time() - start

        if (epoch + 1) in checkpoints:
            test_acc, test_auroc = evaluate_bp(model, test_loader, device, n_batches=10)
            results[epoch + 1] = {'test_acc': test_acc, 'test_auroc': test_auroc}
            print(f"  epoch {epoch+1:2d}: train_loss={total_loss/len(train_loader):.3f} "
                  f"train_acc={train_acc*100:.1f}% "
                  f"test_acc={test_acc*100:.1f}% "
                  f"softmax_AUROC2={test_auroc:.4f} "
                  f"({elapsed:.0f}s)")
        else:
            print(f"  epoch {epoch+1:2d}: train_loss={total_loss/len(train_loader):.3f} "
                  f"train_acc={train_acc*100:.1f}% "
                  f"({elapsed:.0f}s)")

    # Side-by-side comparison
    print()
    print("=" * 60)
    print("SIDE-BY-SIDE: PC vs BP at matched training budget")
    print("=" * 60)
    print(f"{'Epoch':>6} {'PC_Acc':>8} {'PC_AUROC':>10} {'BP_Acc':>8} {'BP_AUROC':>10} "
          f"{'Δ_Acc':>8} {'Δ_AUROC':>10}")

    pc_results = {
        5:  {'acc': 0.6977, 'auroc': 0.8020},
        10: {'acc': 0.7391, 'auroc': 0.8150},
        15: {'acc': 0.7703, 'auroc': 0.8319},
        20: {'acc': 0.7773, 'auroc': 0.8513},
        25: {'acc': 0.7820, 'auroc': 0.8712},
    }

    for ep in checkpoints:
        pc = pc_results[ep]
        bp = results[ep]
        d_acc = pc['acc'] - bp['test_acc']
        d_auroc = pc['auroc'] - bp['test_auroc']
        print(f"{ep:>6d} {pc['acc']*100:>7.2f}% {pc['auroc']:>10.4f} "
              f"{bp['test_acc']*100:>7.2f}% {bp['test_auroc']:>10.4f} "
              f"{d_acc*100:>+7.2f}pp {d_auroc:>+10.4f}")

    # Interpretation
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    final_pc = pc_results[25]
    final_bp = results[25]
    auroc_gap = final_pc['auroc'] - final_bp['test_auroc']
    acc_gap = final_pc['acc'] - final_bp['test_acc']

    print(f"At 25 epochs (matched training budget):")
    print(f"  PC Type-1 accuracy:   {final_pc['acc']*100:.2f}%")
    print(f"  BP Type-1 accuracy:   {final_bp['test_acc']*100:.2f}%")
    print(f"  PC softmax AUROC2:    {final_pc['auroc']:.4f}")
    print(f"  BP softmax AUROC2:    {final_bp['test_auroc']:.4f}")
    print(f"  AUROC2 gap (PC - BP): {auroc_gap:+.4f}")
    print(f"  Acc gap (PC - BP):    {acc_gap*100:+.2f}pp")
    print()

    if abs(auroc_gap) < 0.015:
        outcome = "MATCH"
        interp = (
            "OUTCOME MATCH: PC and BP produce nearly identical softmax AUROC2 "
            "at matched training budget. The apparent gap at 5 epochs was "
            "a training budget artefact. Finding A evaporates. "
            "No PC advantage in softmax calibration at this scale."
        )
    elif auroc_gap > 0.03:
        outcome = "PC-WINS"
        interp = (
            f"OUTCOME PC-WINS: PC softmax AUROC2 exceeds BP by {auroc_gap:.4f} "
            f"at matched training budget. At comparable Type-1 accuracies "
            f"(PC {final_pc['acc']*100:.1f}%, BP {final_bp['test_acc']*100:.1f}%), "
            f"PC training produces meaningfully better-calibrated softmax "
            f"readouts. Finding A is real. This becomes the core of Paper A."
        )
    elif auroc_gap > 0.015:
        outcome = "PC-EDGE"
        interp = (
            f"OUTCOME PC-EDGE: PC softmax AUROC2 exceeds BP by {auroc_gap:.4f}, "
            f"a small but detectable advantage. Whether this is a real finding "
            f"worth building a paper around depends on whether it persists "
            f"across seeds, architectures, and datasets. Single-seed single-"
            f"architecture evidence is weak. Need more runs before committing."
        )
    else:
        outcome = "BP-WINS"
        interp = (
            f"OUTCOME BP-WINS: BP softmax AUROC2 exceeds PC by {-auroc_gap:.4f}. "
            f"BP is simply a better classifier at this scale. PC has no "
            f"calibration advantage. This is a null programme for IMA as "
            f"currently framed."
        )

    print(interp)
    print("=" * 60)


if __name__ == "__main__":
    main()
