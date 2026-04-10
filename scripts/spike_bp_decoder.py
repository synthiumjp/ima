"""
IMA BP+decoder fairness control spike (11 Apr 2026, session 4)

Goal: verify that post-hoc training a generative mirror decoder on a frozen
BP-trained encoder produces a residual-based structural probe with
above-chance Type-2 AUROC2, and to measure how the result compares to the
PC spike's 0.6514.

This is the gating check for IMA Paper A v6 hypothesis H2:
  H2: structural probe on PC > structural probe on BP + trained decoder
If BP+decoder structural probe AUROC2 >= PC structural probe AUROC2,
the fairness contrast (H2) is dead before we register.

Three outcomes:
  (a) BP+decoder AUROC2 ~ 0.50 (chance): structural probing on BP+decoder
      doesn't work at all. Serious rethink before v6.
  (b) BP+decoder AUROC2 substantially below PC (~0.55-0.60): ideal. H2 is
      credible, PC has room to win.
  (c) BP+decoder AUROC2 matches or beats PC (~0.65+): Paper A's central
      claim is dead. Full reframe needed.

Design:
  - TinyFFN encoder: mirrors TinyConvPCN's forward_encoder exactly
    (conv->bn->gelu->pool x2 + fc1+gelu + fc2), ~1.07M params, trained 5 epochs BP
  - TinyDecoder: mirrors TinyConvPCN's gen_fc3, gen_fc2, gen_conv1
    (~1.07M params), trained 5 epochs post-hoc on frozen encoder, MSE loss
  - Evaluation: same 10 batches of 1280 test images as PC spike
  - Probe: residual-based structural probe with CE energy at output
    (clamping each candidate class, computing total energy, energy margin)
  - Directly comparable to spike_dynamics.py's 0.6514 AUROC2

This spike is for v6 gating only. Not a full experiment. Weak success
criteria only: does BP+decoder produce a non-noise AUROC2?
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
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, "src")
from cifar10_data import get_data_loaders


# =============================================================================
# TinyFFN encoder (mirrors TinyConvPCN.forward_encoder exactly)
# =============================================================================

class TinyFFN(nn.Module):
    """Standard BP-trained encoder with the same architecture as
    TinyConvPCN.forward_encoder. No generative weights here."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # Matches TinyConvPCN encoder layout
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_latents=False):
        h1 = F.gelu(self.bn1(self.conv1(x)))      # (B, 32, 32, 32)
        h1 = F.max_pool2d(h1, 2)                   # (B, 32, 16, 16)
        h2 = F.gelu(self.bn2(self.conv2(h1)))      # (B, 64, 16, 16)
        h2 = F.max_pool2d(h2, 2)                   # (B, 64, 8, 8)
        h2_flat = h2.view(x.size(0), -1)
        h3 = F.gelu(self.fc1(h2_flat))             # (B, 256)
        h4 = self.fc2(h3)                          # (B, 10) logits
        if return_latents:
            return [h1, h2, h3, h4]
        return h4


# =============================================================================
# TinyDecoder (mirrors TinyConvPCN's gen_* weights)
# =============================================================================

class TinyDecoder(nn.Module):
    """Post-hoc generative decoder that mirrors TinyConvPCN's generative
    weights (gen_fc3, gen_fc2, gen_conv1).
    
    Maps latents from layer above to predictions for layer below:
      g_3: h4 (10) -> pred for h3 (256)
      g_2: h3 (256) -> pred for h2 (64, 8, 8)
      g_1: h2 (64, 8, 8) -> pred for h1 (32, 16, 16)
           (via upsample to 16x16 then conv)
    
    Trained post-hoc via MSE reconstruction on frozen encoder activations.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.gen_fc3 = nn.Linear(num_classes, 256)
        self.gen_fc2 = nn.Linear(256, 64 * 8 * 8)
        self.gen_conv1 = nn.Conv2d(64, 32, 3, padding=1)

    def predict(self, latents):
        """Given [h1, h2, h3, h4], return predictions [mu_1, mu_2, mu_3]
        where mu_l is the decoder's prediction of h_l from h_{l+1}.
        """
        h1, h2, h3, h4 = latents
        B = h1.size(0)

        # mu_3 = g_3(h4)
        mu_3 = self.gen_fc3(h4)  # (B, 256)

        # mu_2 = g_2(h3) reshaped to (B, 64, 8, 8)
        mu_2_flat = self.gen_fc2(h3)
        mu_2 = mu_2_flat.view(B, 64, 8, 8)

        # mu_1: upsample h2 to (B, 64, 16, 16) then conv down to (B, 32, 16, 16)
        h2_up = F.interpolate(h2, scale_factor=2, mode='nearest')
        mu_1 = self.gen_conv1(h2_up)

        return [mu_1, mu_2, mu_3]

    def compute_residuals(self, latents):
        """eps_l = h_l - g_l(h_{l+1}) for l in {1, 2, 3}."""
        mus = self.predict(latents)
        residuals = [latents[l] - mus[l] for l in range(3)]
        return residuals


# =============================================================================
# Training: BP encoder + post-hoc decoder
# =============================================================================

def train_encoder(model, train_loader, device, epochs=5):
    """Standard BP training with CE loss."""
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    start = time.time()
    for epoch in range(epochs):
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
        acc = total_correct / total
        elapsed = time.time() - start
        print(f"  enc epoch {epoch+1}: loss={total_loss/len(train_loader):.3f} "
              f"acc={acc*100:.1f}% ({elapsed:.1f}s)")


def train_decoder(encoder, decoder, train_loader, device, epochs=5):
    """Train decoder to reconstruct encoder activations via MSE.
    Encoder is frozen throughout."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    optim = AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    decoder.train()
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            with torch.no_grad():
                latents = encoder(x, return_latents=True)
            optim.zero_grad()
            residuals = decoder.compute_residuals(latents)
            # MSE loss: minimize per-layer residual norm
            loss = sum(r.pow(2).mean() for r in residuals)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        elapsed = time.time() - start
        print(f"  dec epoch {epoch+1}: recon_loss={total_loss/len(train_loader):.4f} "
              f"({elapsed:.1f}s)")


# =============================================================================
# Structural probe evaluation (K-way energy-based)
# =============================================================================

@torch.no_grad()
def structural_probe_bp_decoder(encoder, decoder, x):
    """For each test image, compute per-hypothesis total energy:
      E(x, y_k) = sum_l 0.5 * mean(residual_l^2) + CE(logits, y_k)
    
    where residuals are computed with h_output clamped to y_k.
    
    Note: since encoder is frozen and only fc2 feeds into g_3, clamping
    y_k only changes the top residual via g_3. The lower residuals are
    recomputed from the frozen forward-pass h1, h2, h3 values.
    """
    B = x.size(0)
    K = 10  # num_classes
    device = x.device

    # Forward pass through encoder (frozen)
    latents_init = encoder(x, return_latents=True)
    h1, h2, h3, h4 = latents_init

    # For each hypothesis k, clamp h4 -> y_k and compute total energy
    all_energies = torch.zeros(B, K, device=device)
    for k in range(K):
        y_k = torch.zeros(B, K, device=device)
        y_k[:, k] = 1.0

        # Build latents with clamped top
        # h1, h2, h3 stay frozen (forward-pass values)
        # h4 is clamped to y_k
        clamped_latents = [h1, h2, h3, y_k]

        # Compute residuals with clamped top
        residuals = decoder.compute_residuals(clamped_latents)

        # Per-sample latent energy (per-element mean, matching PC spike)
        lat_energy = torch.zeros(B, device=device)
        for r in residuals:
            r_flat = r.view(B, -1)
            lat_energy = lat_energy + 0.5 * r_flat.pow(2).mean(dim=1)

        # CE energy at output: -sum_k y_k * log_softmax(h4_logits)_k
        log_p = F.log_softmax(h4, dim=-1)
        ce_energy = -(y_k * log_p).sum(dim=-1)  # (B,)

        all_energies[:, k] = lat_energy + ce_energy

    pred = all_energies.argmin(dim=1)
    return pred, all_energies


# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda"
    torch.manual_seed(42)

    print("=" * 60)
    print("BP+DECODER FAIRNESS CONTROL SPIKE")
    print("=" * 60)
    print("Goal: measure AUROC2 of structural probe on BP+decoder")
    print("Direct comparison point: PC spike got AUROC2 = 0.6514")
    print()

    # Data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_data_loaders("data", batch_size=128, num_workers=0)
    print(f"  train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    # Encoder: BP training
    print()
    print("Training encoder (BP, 5 epochs)...")
    encoder = TinyFFN(num_classes=10).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder params: {enc_params:,}")
    train_encoder(encoder, train_loader, device, epochs=5)

    # Decoder: post-hoc MSE training
    print()
    print("Training decoder on frozen encoder (MSE recon, 5 epochs)...")
    decoder = TinyDecoder(num_classes=10).to(device)
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  decoder params: {dec_params:,}")
    train_decoder(encoder, decoder, train_loader, device, epochs=5)

    # Evaluation
    print()
    print("Evaluating structural probe...")
    encoder.eval()
    decoder.eval()
    all_preds = []
    all_targets = []
    all_correct = []
    all_energies_list = []

    # Use same 10 batches as PC spike (~1280 images)
    N_eval_batches = 10

    eval_start = time.time()
    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx >= N_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)

        pred, energies = structural_probe_bp_decoder(encoder, decoder, x)

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_correct.append((pred == y).cpu())
        all_energies_list.append(energies.cpu())

    eval_elapsed = time.time() - eval_start

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    correct = torch.cat(all_correct)
    energies = torch.cat(all_energies_list)

    N = preds.size(0)
    accuracy = correct.float().mean().item()

    # Also report the softmax-based accuracy and AUROC for reference
    print()
    print("Reference: standard softmax on same data...")
    with torch.no_grad():
        softmax_correct_list = []
        softmax_conf_list = []
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= N_eval_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = encoder(x)
            probs = F.softmax(logits, dim=-1)
            pred_sm = logits.argmax(dim=-1)
            max_p = probs.max(dim=-1).values
            softmax_correct_list.append((pred_sm == y).cpu())
            softmax_conf_list.append(max_p.cpu())
        softmax_correct = torch.cat(softmax_correct_list)
        softmax_conf = torch.cat(softmax_conf_list)
        softmax_acc = softmax_correct.float().mean().item()
        if softmax_correct.sum() > 0 and (~softmax_correct).sum() > 0:
            softmax_auroc = roc_auc_score(softmax_correct.numpy(), softmax_conf.numpy())
        else:
            softmax_auroc = float('nan')

    # Structural probe diagnostics
    e_min = energies.min(dim=1).values
    e_max = energies.max(dim=1).values
    e_mean = energies.mean(dim=1)
    relative_spread = (e_max - e_min) / (e_mean.abs() + 1e-8)
    mean_relative_spread = relative_spread.mean().item()

    sorted_energies, _ = energies.sort(dim=1)
    energy_margin = sorted_energies[:, 1] - sorted_energies[:, 0]

    if correct.sum() > 0 and (~correct).sum() > 0:
        structural_auroc = roc_auc_score(correct.numpy(), energy_margin.numpy())
    else:
        structural_auroc = float('nan')

    # Report
    print()
    print("=" * 60)
    print("SPIKE RESULTS — BP + Decoder Structural Probe")
    print("=" * 60)
    print(f"Evaluated on {N} test images ({eval_elapsed:.1f}s)")
    print()
    print(f"Reference (standard softmax on same BP encoder):")
    print(f"  accuracy:        {softmax_acc*100:.2f}%")
    print(f"  softmax AUROC2:  {softmax_auroc:.4f}")
    print()
    print(f"Structural probe (BP + trained decoder, energy-based K-way):")
    print(f"  argmin accuracy: {accuracy*100:.2f}%")
    print(f"  energy spread:   {mean_relative_spread*100:.2f}% mean relative")
    print(f"  spread range:    {relative_spread.min()*100:.2f}% / {relative_spread.max()*100:.2f}%")
    print(f"  energy margin AUROC2: {structural_auroc:.4f}")
    print()
    print(f"Comparison reference:")
    print(f"  PC spike (same architecture, same eval): AUROC2 = 0.6514")
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if not math.isnan(structural_auroc):
        if structural_auroc < 0.53:
            print("OUTCOME (a): BP+decoder produces chance-level AUROC2 (~0.50)")
            print("  -> Structural probing on BP+decoder doesn't work at all.")
            print("  -> RETHINK REQUIRED: the residual-based approach may not")
            print("     transfer to post-hoc trained decoders. Consider whether")
            print("     v6 should drop H2 entirely and focus on 'PC structural")
            print("     probe vs post-hoc single-point probes on BP'.")
        elif structural_auroc < 0.6514 - 0.02:
            print("OUTCOME (b): BP+decoder produces real but weaker AUROC2")
            print(f"  -> Difference to PC: {0.6514 - structural_auroc:.4f} ({(0.6514 - structural_auroc)/0.6514*100:.1f}%)")
            print("  -> IDEAL outcome: H2 is credible, PC has demonstrable room")
            print("     to win. Proceed to register v6 with H2 as framed.")
        elif structural_auroc <= 0.6514 + 0.02:
            print("OUTCOME (b'): BP+decoder produces AUROC2 very close to PC")
            print(f"  -> Difference to PC: {structural_auroc - 0.6514:.4f}")
            print("  -> BORDERLINE: the PC advantage at this scale may not be")
            print("     large enough to survive 6-seed sign consistency.")
            print("     Need to think about whether H2 is registerable.")
        else:
            print("OUTCOME (c): BP+decoder MATCHES OR BEATS PC structural probe")
            print(f"  -> Difference: BP+decoder is {structural_auroc - 0.6514:.4f} ABOVE PC")
            print("  -> CENTRAL CLAIM COLLAPSE: Paper A's H2 as framed is dead.")
            print("     The structural probe advantage is not about PC training,")
            print("     it's about having a residual-based energy function.")
            print("     Full reframe needed before v6 drafting.")
    print("=" * 60)


if __name__ == "__main__":
    main()
