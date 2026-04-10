"""
IMA extended PC training spike (11 Apr 2026, session 4)

Goal: determine whether PC-trained TinyConvPCN at proper training budget
(25 epochs vs the 5-epoch spike this morning) produces:
  (a) Type-1 accuracy competitive with BP (BP got 71% at 5 epochs)
  (b) Structural probe AUROC2 at or above BP softmax (BP got 0.7765)
  (c) A meaningful signal beyond what is already in the output logits
      (softmax on PC's trained classifier head)

This settles whether Paper A's primary hypothesis is testable at this
scale. Three outcomes:

OUTCOME X: PC AUROC2 climbs to 0.77+ while matching BP Type-1 at ~65-70%.
  -> The structural probe can compete with BP softmax. H2 as
     "PC structural probe > BP softmax at matched Type-1" is testable.
     Proceed to draft v6 with this as the primary hypothesis.

OUTCOME Y: PC AUROC2 climbs but plateaus meaningfully below BP softmax
  (e.g., PC at 0.68-0.74, BP at 0.78).
  -> The primary hypothesis fails at this scale. Either (a) the advantage
     doesn't exist or is small, or (b) VGG5 at full scale is needed to
     reveal it. v6 becomes a methodology-characterization paper not a
     superiority-claim paper, or we pause to reconsider.

OUTCOME Z: PC fails to train, Type-1 stalls below 50%, AUROC2 stays low.
  -> Our PC implementation may be broken for extended training. Debug.

Reports at 5, 10, 15, 20, 25 epochs so we can see the trajectory.
"""
# --- CUDNN WORKAROUND ---
import torch
torch.backends.cudnn.enabled = False
# ------------------------

import sys
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from cifar10_data import get_data_loaders

# Import TinyConvPCN and training functions from the earlier spike
import importlib.util
spec = importlib.util.spec_from_file_location("spike_dynamics", "scripts/spike_dynamics.py")
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)

TinyConvPCN = spike.TinyConvPCN
train_step = spike.train_step
classify_energy_based = spike.classify_energy_based


def evaluate_structural_probe(model, test_loader, device, n_batches=10):
    """Return (argmin_acc, softmax_acc, structural_auroc, softmax_auroc)."""
    model.eval()
    all_preds_struct = []
    all_preds_softmax = []
    all_targets = []
    all_correct_struct = []
    all_correct_softmax = []
    all_energies_list = []
    all_softmax_conf = []

    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)

        # Structural probe: K-way energy-based
        pred_struct, energies = classify_energy_based(
            model, x, T=13, eta_h=5e-2, momentum_h=0.5
        )

        # Softmax reference (use the trained readout head)
        with torch.no_grad():
            latents = model.forward_encoder(x)
            logits = latents[3]
            probs = F.softmax(logits, dim=-1)
            pred_softmax = logits.argmax(dim=-1)
            max_p = probs.max(dim=-1).values

        all_preds_struct.append(pred_struct.cpu())
        all_preds_softmax.append(pred_softmax.cpu())
        all_targets.append(y.cpu())
        all_correct_struct.append((pred_struct == y).cpu())
        all_correct_softmax.append((pred_softmax == y).cpu())
        all_energies_list.append(energies.cpu())
        all_softmax_conf.append(max_p.cpu())

    targets = torch.cat(all_targets)
    correct_struct = torch.cat(all_correct_struct)
    correct_softmax = torch.cat(all_correct_softmax)
    energies = torch.cat(all_energies_list)
    softmax_conf = torch.cat(all_softmax_conf)

    argmin_acc = correct_struct.float().mean().item()
    softmax_acc = correct_softmax.float().mean().item()

    # Structural probe AUROC using energy margin
    sorted_e, _ = energies.sort(dim=1)
    energy_margin = sorted_e[:, 1] - sorted_e[:, 0]
    if correct_struct.sum() > 0 and (~correct_struct).sum() > 0:
        structural_auroc = roc_auc_score(
            correct_struct.numpy(), energy_margin.numpy()
        )
    else:
        structural_auroc = float('nan')

    # Softmax AUROC (for direct comparison to BP+decoder spike)
    if correct_softmax.sum() > 0 and (~correct_softmax).sum() > 0:
        softmax_auroc = roc_auc_score(
            correct_softmax.numpy(), softmax_conf.numpy()
        )
    else:
        softmax_auroc = float('nan')

    return argmin_acc, softmax_acc, structural_auroc, softmax_auroc


def main():
    device = "cuda"
    torch.manual_seed(42)

    print("=" * 60)
    print("EXTENDED PC TRAINING SPIKE (25 epochs)")
    print("=" * 60)
    print("Goal: determine if PC at proper training budget produces")
    print("competitive Type-1 and structural probe AUROC2.")
    print()
    print("Reference points (from BP+decoder spike earlier this session):")
    print("  BP encoder Type-1 accuracy:       71.17%")
    print("  BP encoder softmax AUROC2:        0.7765")
    print("  BP+decoder structural AUROC2:     0.7674 (= softmax)")
    print()
    print("Reference points (from PC spike at 5 epochs this morning):")
    print("  PC Type-1 accuracy:               37.27%")
    print("  PC structural AUROC2:             0.6514")
    print()

    # Data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # Model
    model = TinyConvPCN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optim_w = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Checkpoints at 5, 10, 15, 20, 25 epochs
    checkpoints = [5, 10, 15, 20, 25]
    results = {}

    print()
    print("Training and evaluating at checkpoints...")
    start = time.time()
    model.train()
    for epoch in range(25):
        epoch_loss = 0.0
        epoch_gen = 0.0
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
            epoch_read += read_l
            n_batches += 1

        elapsed = time.time() - start
        avg_loss = epoch_loss / n_batches
        print(f"  epoch {epoch+1:2d}: loss={avg_loss:.3f} "
              f"gen={epoch_gen/n_batches:.3f} "
              f"read={epoch_read/n_batches:.3f} "
              f"({elapsed:.0f}s)")

        # Evaluate at checkpoints
        if (epoch + 1) in checkpoints:
            argmin_acc, softmax_acc, struct_auroc, softmax_auroc = \
                evaluate_structural_probe(model, test_loader, device, n_batches=10)
            results[epoch + 1] = {
                'argmin_acc': argmin_acc,
                'softmax_acc': softmax_acc,
                'struct_auroc': struct_auroc,
                'softmax_auroc': softmax_auroc,
            }
            print(f"    @ep{epoch+1}: argmin_acc={argmin_acc*100:.1f}% "
                  f"softmax_acc={softmax_acc*100:.1f}% "
                  f"struct_AUROC2={struct_auroc:.4f} "
                  f"softmax_AUROC2={softmax_auroc:.4f}")
            model.train()

    # Final report
    print()
    print("=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"{'Epoch':>6} {'ArgminAcc':>10} {'SoftmaxAcc':>11} "
          f"{'StructAUROC':>12} {'SoftmaxAUROC':>13}")
    for ep in checkpoints:
        r = results[ep]
        print(f"{ep:>6d} {r['argmin_acc']*100:>9.2f}% "
              f"{r['softmax_acc']*100:>10.2f}% "
              f"{r['struct_auroc']:>12.4f} {r['softmax_auroc']:>13.4f}")

    # Reference table
    print()
    print(f"{'Reference':>25} {'Acc':>6} {'AUROC2':>9}")
    print(f"{'BP encoder softmax':>25} {71.17:>5.1f}% {0.7765:>9.4f}")
    print(f"{'BP+decoder structural':>25} {71.09:>5.1f}% {0.7674:>9.4f}")
    print(f"{'PC ep5 (morning spike)':>25} {37.27:>5.1f}% {0.6514:>9.4f}")

    # Interpretation
    final = results[25]
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    pc_final_acc = final['softmax_acc']
    pc_final_struct_auroc = final['struct_auroc']
    pc_final_softmax_auroc = final['softmax_auroc']
    bp_reference_acc = 0.7117
    bp_reference_auroc = 0.7765

    print(f"At 25 epochs, PC network:")
    print(f"  Type-1 accuracy (softmax arg): {pc_final_acc*100:.2f}%")
    print(f"  Type-1 accuracy (argmin energy): {final['argmin_acc']*100:.2f}%")
    print(f"  Structural probe AUROC2: {pc_final_struct_auroc:.4f}")
    print(f"  Softmax on PC head AUROC2: {pc_final_softmax_auroc:.4f}")
    print()
    print(f"Compared to BP reference (71.17% acc, 0.7765 softmax AUROC2):")
    print(f"  Accuracy gap: {(pc_final_acc - bp_reference_acc)*100:+.2f} pp")
    print(f"  Structural AUROC2 gap: {pc_final_struct_auroc - bp_reference_auroc:+.4f}")
    print()

    # Outcome classification
    if pc_final_acc < 0.50:
        outcome = "Z"
        interp = ("OUTCOME Z: PC failed to train to reasonable accuracy. "
                  "Implementation may be broken for extended training. Debug needed.")
    elif pc_final_struct_auroc > bp_reference_auroc - 0.01:
        if pc_final_acc > bp_reference_acc - 0.03:
            outcome = "X"
            interp = ("OUTCOME X: PC structural probe competitive with BP softmax "
                      "at comparable Type-1 accuracy. Primary hypothesis testable. "
                      "Proceed with v6 drafting.")
        else:
            outcome = "X'"
            interp = ("OUTCOME X': PC structural probe matches BP softmax AUROC2 "
                      "but Type-1 accuracy is still meaningfully behind. "
                      "Need to think about whether 'at matched Type-1' is the "
                      "right framing or whether M-ratio-style normalization is "
                      "the honest comparison.")
    else:
        outcome = "Y"
        gap = bp_reference_auroc - pc_final_struct_auroc
        interp = (f"OUTCOME Y: PC structural probe plateaus {gap:.4f} below "
                  f"BP softmax ({pc_final_struct_auroc:.4f} vs {bp_reference_auroc:.4f}). "
                  f"Primary hypothesis fails at this scale. v6 needs substantial "
                  f"reframing: methodology characterization, not superiority claim.")

    print(interp)

    # Also check whether the structural probe adds anything beyond softmax on PC
    struct_vs_softmax_pc = pc_final_struct_auroc - pc_final_softmax_auroc
    print()
    print(f"Structural probe vs softmax on same PC network:")
    print(f"  struct AUROC2 - softmax AUROC2 = {struct_vs_softmax_pc:+.4f}")
    if abs(struct_vs_softmax_pc) < 0.01:
        print(f"  Near-identical: structural probe is NOT a different readout")
        print(f"  from softmax on the same network at this scale.")
    elif struct_vs_softmax_pc > 0.01:
        print(f"  Structural probe adds signal beyond softmax on same network.")
    else:
        print(f"  Structural probe is WORSE than softmax on same network.")
        print(f"  This would mean the probe is losing information.")

    print("=" * 60)


if __name__ == "__main__":
    main()
