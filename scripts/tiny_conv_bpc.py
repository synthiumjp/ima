"""
TinyConvBPC: Bidirectional Predictive Coding on the TinyConv backbone
=====================================================================

Implements Oliviers, Tang & Bogacz (2025; arXiv:2505.23415) bPC on
the same structural backbone as TinyConvPCN from the IMA paper
(Cacioli, 2026; arXiv:2604.11011).

Key differences from TinyConvPCN (standard discriminative PC):
  - Energy: bidirectional MSE (α_gen * top-down + α_disc * bottom-up), NO CE
  - Inference: both generative and discriminative errors drive latent updates
  - V pathway (encoder) weights participate in inference, not just init
  - Training: x_1 clamped to input, x_L clamped to one-hot label (same as discPC)

Architecture (identical backbone):
  L0: input (3, 32, 32) — clamped
  L1: (32, 16, 16) — latent
  L2: (64, 8, 8) — latent
  L3: (256,) — latent
  L4: (10,) — output, clamped during training

  V (bottom-up / discriminative):
    V_0: Conv(3→32, 3×3, pad=1) → BN → GELU → MaxPool(2)
    V_1: Conv(32→64, 3×3, pad=1) → BN → GELU → MaxPool(2)
    V_2: Linear(64·8·8 → 256) → GELU
    V_3: Linear(256 → 10)

  W (top-down / generative):
    W_3: Linear(10 → 256)
    W_2: Linear(256 → 64·8·8), reshaped to (64, 8, 8)
    W_1: Interpolate(scale=2) + Conv(64 → 32, 3×3, pad=1)

  Total parameters: 2,144,938 (identical to TinyConvPCN)

Author: JP Cacioli
Date: April 2026
Pre-registration: OSF [pending]
"""
# --- CUDNN WORKAROUND: required for AMD ROCm wheel BN bug ---
import torch
torch.backends.cudnn.enabled = False
# -------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Needed for measure_latent_movement forward reference



class TinyConvBPC(nn.Module):
    """Bidirectional Predictive Coding network on TinyConv backbone.
    
    Same V (encoder) and W (generative) weight shapes as TinyConvPCN.
    Energy is bidirectional MSE with α_gen and α_disc weighting.
    No cross-entropy term anywhere.
    """
    
    def __init__(self, num_classes: int = 10, 
                 alpha_gen: float = 1e-5, 
                 alpha_disc: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_gen = alpha_gen
        self.alpha_disc = alpha_disc
        
        # =============================================================
        # V pathway (bottom-up / discriminative predictions)
        # These are the same weights as TinyConvPCN's encoder.
        # In bPC, they also participate in inference (not just init).
        # =============================================================
        self.v_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.v_bn1 = nn.BatchNorm2d(32)
        self.v_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.v_bn2 = nn.BatchNorm2d(64)
        self.v_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, num_classes)
        
        # =============================================================
        # W pathway (top-down / generative predictions)
        # Same weights as TinyConvPCN's generative chain.
        # =============================================================
        # W_3: L4 (10) → L3 (256)
        self.w_fc3 = nn.Linear(num_classes, 256)
        # W_2: L3 (256) → L2 (64, 8, 8)
        self.w_fc2 = nn.Linear(256, 64 * 8 * 8)
        # W_1: L2 (64, 8, 8) → L1 (32, 16, 16) via upsample + conv
        self.w_conv1 = nn.Conv2d(64, 32, 3, padding=1)
    
    def forward_v(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Bottom-up feedforward sweep through V weights.
        
        Used for initialisation (same as TinyConvPCN.forward_encoder).
        Returns [h1, h2, h3, h4] — initial latent values.
        """
        h1 = F.gelu(self.v_bn1(self.v_conv1(x)))   # (B, 32, 32, 32)
        h1 = F.max_pool2d(h1, 2)                     # (B, 32, 16, 16)
        h2 = F.gelu(self.v_bn2(self.v_conv2(h1)))   # (B, 64, 16, 16)
        h2 = F.max_pool2d(h2, 2)                     # (B, 64, 8, 8)
        h2_flat = h2.view(x.size(0), -1)
        h3 = F.gelu(self.v_fc1(h2_flat))             # (B, 256)
        h4 = self.v_fc2(h3)                          # (B, 10)
        return [h1, h2, h3, h4]
    
    def compute_v_predictions(self, latents: List[torch.Tensor], 
                               x: torch.Tensor) -> List[torch.Tensor]:
        """Compute bottom-up (discriminative) predictions V_l(f(x_l)).
        
        These predict each layer FROM the layer below.
        Returns [v_pred_1, v_pred_2, v_pred_3, v_pred_4] for layers 1-4.
        
        v_pred_1 = V_0(f(x_0)) — prediction of L1 from input
        v_pred_2 = V_1(f(x_1)) — prediction of L2 from L1
        v_pred_3 = V_2(f(x_2)) — prediction of L3 from L2
        v_pred_4 = V_3(f(x_3)) — prediction of L4 from L3
        """
        h1, h2, h3, h4 = latents
        B = x.size(0)
        
        # v_pred_1: V_0(x_0) = Conv1 + BN + GELU + Pool applied to input
        v_pred_1 = F.gelu(self.v_bn1(self.v_conv1(x)))
        v_pred_1 = F.max_pool2d(v_pred_1, 2)  # (B, 32, 16, 16)
        
        # v_pred_2: V_1(h1) = Conv2 + BN + GELU + Pool applied to h1
        v_pred_2 = F.gelu(self.v_bn2(self.v_conv2(h1)))
        v_pred_2 = F.max_pool2d(v_pred_2, 2)  # (B, 64, 8, 8)
        
        # v_pred_3: V_2(h2) = FC1 + GELU applied to flattened h2
        h2_flat = h2.view(B, -1)
        v_pred_3 = F.gelu(self.v_fc1(h2_flat))  # (B, 256)
        
        # v_pred_4: V_3(h3) = FC2 applied to h3 (no activation at output)
        v_pred_4 = self.v_fc2(h3)  # (B, 10)
        
        return [v_pred_1, v_pred_2, v_pred_3, v_pred_4]
    
    def compute_w_predictions(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute top-down (generative) predictions W_l(f(x_{l+1})).
        
        These predict each layer FROM the layer above.
        Returns [w_pred_1, w_pred_2, w_pred_3] for layers 1-3.
        
        w_pred_3 = W_3(h4) — prediction of L3 from L4
        w_pred_2 = W_2(h3) — prediction of L2 from L3
        w_pred_1 = W_1(h2) — prediction of L1 from L2
        """
        h1, h2, h3, h4 = latents
        B = h1.size(0)
        
        # w_pred_3: W_3(h4)
        w_pred_3 = self.w_fc3(h4)  # (B, 256)
        
        # w_pred_2: W_2(h3) reshaped
        w_pred_2_flat = self.w_fc2(h3)  # (B, 64*8*8)
        w_pred_2 = w_pred_2_flat.view(B, 64, 8, 8)
        
        # w_pred_1: upsample h2 then conv
        h2_up = F.interpolate(h2, scale_factor=2, mode='nearest')  # (B, 64, 16, 16)
        w_pred_1 = self.w_conv1(h2_up)  # (B, 32, 16, 16)
        
        return [w_pred_1, w_pred_2, w_pred_3]
    
    def bpc_energy(self, latents: List[torch.Tensor], 
                   x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bPC energy (Oliviers et al. Eq. 3).
        
        E = Σ_{l=1}^{L-1} (α_gen/2) ‖x_l - W_{l+1}(f(x_{l+1}))‖²
          + Σ_{l=2}^{L}   (α_disc/2) ‖x_l - V_{l-1}(f(x_{l-1}))‖²
        
        Per-element mean reduction (matching TinyConvPCN convention).
        
        Returns:
            per_sample_total: (B,) total energy per sample
            per_sample_gen: (B,) generative component
            per_sample_disc: (B,) discriminative component
        """
        h1, h2, h3, h4 = latents
        B = h1.size(0)
        
        # Generative errors: ε^gen_l = x_l - W_{l+1}(f(x_{l+1})) for l=1,2,3
        w_preds = self.compute_w_predictions(latents)  # [w1, w2, w3]
        gen_errors = [
            h1 - w_preds[0],  # ε^gen_1: L1 - W_1(L2)
            h2 - w_preds[1],  # ε^gen_2: L2 - W_2(L3)
            h3 - w_preds[2],  # ε^gen_3: L3 - W_3(L4)
        ]
        
        # Discriminative errors: ε^disc_l = x_l - V_{l-1}(f(x_{l-1})) for l=2,3,4
        v_preds = self.compute_v_predictions(latents, x)  # [v1, v2, v3, v4]
        disc_errors = [
            h2 - v_preds[1],  # ε^disc_2: L2 - V_1(L1)
            h3 - v_preds[2],  # ε^disc_3: L3 - V_2(L2)
            h4 - v_preds[3],  # ε^disc_4: L4 - V_3(L3)
        ]
        
        # Generative energy: Σ (α_gen/2) * mean(ε^gen_l²)
        per_sample_gen = torch.zeros(B, device=h1.device)
        for e in gen_errors:
            e_flat = e.view(B, -1)
            per_sample_gen = per_sample_gen + (self.alpha_gen / 2) * e_flat.pow(2).mean(dim=1)
        
        # Discriminative energy: Σ (α_disc/2) * mean(ε^disc_l²)
        per_sample_disc = torch.zeros(B, device=h1.device)
        for e in disc_errors:
            e_flat = e.view(B, -1)
            per_sample_disc = per_sample_disc + (self.alpha_disc / 2) * e_flat.pow(2).mean(dim=1)
        
        per_sample_total = per_sample_gen + per_sample_disc
        return per_sample_total, per_sample_gen, per_sample_disc


def bpc_train_step(model: TinyConvBPC, x: torch.Tensor, y: torch.Tensor,
                   T: int = 32, eta_h: float = 5e-2, momentum_h: float = 0.5):
    """One bPC training step.
    
    Phase 1: Bottom-up feedforward sweep (V weights) for initialisation.
    Phase 2: T inference steps — update latents h1, h2, h3 by gradient
             descent on bPC energy. h4 is clamped to y_onehot. x is clamped.
    Phase 3: Compute bPC energy at settled state + encoder alignment loss,
             return for weight update.
    
    Key difference from standard PC train_step:
      - Energy is bidirectional MSE (no CE)
      - Both V and W predictions contribute to latent gradients
      - V weights are trained via encoder alignment (same mechanism as discPC)
      - W weights are trained via the generative component of the bPC energy
    
    Args:
        model: TinyConvBPC instance
        x: input images (B, 3, 32, 32)
        y: class labels (B,) integers
        T: number of inference steps
        eta_h: learning rate for latent updates
        momentum_h: momentum for latent SGD
    
    Returns:
        total_loss: scalar tensor for .backward()
        gen_loss: float, generative energy component
        disc_loss: float, discriminative energy component  
        enc_loss: float, encoder alignment loss
    """
    B = x.size(0)
    num_classes = model.num_classes
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    
    # Phase 1: amortised init via V pathway
    with torch.no_grad():
        latents_init = model.forward_v(x)
    
    # Detach and make leaf tensors for manual optimisation
    h1 = latents_init[0].clone().detach().requires_grad_(True)
    h2 = latents_init[1].clone().detach().requires_grad_(True)
    h3 = latents_init[2].clone().detach().requires_grad_(True)
    # h4 clamped to one-hot target (same as standard PC training)
    h4_clamped = y_onehot.clone().detach()
    
    # Momentum buffers
    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    m3 = torch.zeros_like(h3)
    
    # Phase 2: inference loop (bPC dynamics — both error streams)
    for t in range(T):
        latents = [h1, h2, h3, h4_clamped]
        per_sample_total, _, _ = model.bpc_energy(latents, x)
        total_energy_scalar = per_sample_total.sum()
        
        # Gradients of bPC energy w.r.t. free latents
        grads = torch.autograd.grad(
            total_energy_scalar,
            [h1, h2, h3],
            create_graph=False,
            retain_graph=False,
        )
        
        # SGD with momentum on latents
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
    
    # Phase 3: Compute loss for weight update
    # 3a: Encoder alignment — V pathway should predict settled latents
    latents_enc = model.forward_v(x)
    h1_enc, h2_enc, h3_enc, h4_enc = latents_enc
    enc_loss = (
        F.mse_loss(h1_enc, h1.detach())
        + F.mse_loss(h2_enc, h2.detach())
        + F.mse_loss(h3_enc, h3.detach())
    )
    
    # 3b: bPC energy at settled state (trains W weights via gen errors,
    # and V weights indirectly via disc errors through the computation graph)
    latents_settled = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
    per_sample_total, per_sample_gen, per_sample_disc = model.bpc_energy(
        latents_settled, x
    )
    energy_loss = per_sample_total.mean()
    
    total_loss = energy_loss + enc_loss
    return (total_loss, 
            per_sample_gen.mean().item(), 
            per_sample_disc.mean().item(),
            enc_loss.item())


@torch.no_grad()
def bpc_classify_energy_based(model: TinyConvBPC, x: torch.Tensor,
                               T: int = 100, eta_h: float = 5e-2, 
                               momentum_h: float = 0.5):
    """K-way energy-based classification on a trained bPC network.
    
    For each hypothesis k, clamp h4 = one-hot(k), run bPC inference,
    record settled total energy. Predict argmin.
    
    Returns:
        pred: (B,) predicted class indices
        all_energies: (B, K) per-sample per-hypothesis total energy
        all_gen_energies: (B, K) generative component
        all_disc_energies: (B, K) discriminative component
    """
    B = x.size(0)
    K = model.num_classes
    device = x.device
    
    all_energies = torch.zeros(B, K, device=device)
    all_gen_energies = torch.zeros(B, K, device=device)
    all_disc_energies = torch.zeros(B, K, device=device)
    
    for k in range(K):
        # Re-init latents from V pathway (fresh per hypothesis)
        latents_init = model.forward_v(x)
        
        h1 = latents_init[0].clone().requires_grad_(True)
        h2 = latents_init[1].clone().requires_grad_(True)
        h3 = latents_init[2].clone().requires_grad_(True)
        
        # Clamp hypothesis
        y_k = torch.zeros(B, K, device=device)
        y_k[:, k] = 1.0
        h4_clamped = y_k
        
        m1 = torch.zeros_like(h1)
        m2 = torch.zeros_like(h2)
        m3 = torch.zeros_like(h3)
        
        # bPC inference loop
        with torch.enable_grad():
            for t in range(T):
                latents = [h1, h2, h3, h4_clamped]
                per_sample_total, _, _ = model.bpc_energy(latents, x)
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
                    h1 -= eta_h * m1
                    h2 -= eta_h * m2
                    h3 -= eta_h * m3
                h1.requires_grad_(True)
                h2.requires_grad_(True)
                h3.requires_grad_(True)
        
        # Record settled energy
        latents_final = [h1.detach(), h2.detach(), h3.detach(), h4_clamped]
        per_sample_total, per_sample_gen, per_sample_disc = model.bpc_energy(
            latents_final, x
        )
        all_energies[:, k] = per_sample_total
        all_gen_energies[:, k] = per_sample_gen
        all_disc_energies[:, k] = per_sample_disc
    
    pred = all_energies.argmin(dim=1)
    return pred, all_energies, all_gen_energies, all_disc_energies


def measure_latent_movement(model: TinyConvBPC, x: torch.Tensor,
                            T: int = 100, eta_h: float = 5e-2,
                            momentum_h: float = 0.5,
                            clamp_class: Optional[int] = None):
    """Measure per-layer latent displacement between feedforward init 
    and settled state (H3 diagnostic).
    
    If clamp_class is None, uses the network's own argmax prediction.
    
    Returns:
        movements: list of per-layer mean per-element |Δh_l|
        max_movement: maximum across layers
    """
    B = x.size(0)
    K = model.num_classes
    device = x.device
    
    with torch.no_grad():
        latents_init = model.forward_v(x)
    
    # Store initial values
    h1_init = latents_init[0].clone()
    h2_init = latents_init[1].clone()
    h3_init = latents_init[2].clone()
    
    # Determine clamped class
    if clamp_class is not None:
        y_k = torch.zeros(B, K, device=device)
        y_k[:, clamp_class] = 1.0
    else:
        # Use V-pathway argmax
        y_k = torch.zeros(B, K, device=device)
        pred = latents_init[3].argmax(dim=1)
        y_k.scatter_(1, pred.unsqueeze(1), 1.0)
    
    h1 = h1_init.clone().requires_grad_(True)
    h2 = h2_init.clone().requires_grad_(True)
    h3 = h3_init.clone().requires_grad_(True)
    h4_clamped = y_k
    
    m1 = torch.zeros_like(h1)
    m2 = torch.zeros_like(h2)
    m3 = torch.zeros_like(h3)
    
    # Run inference
    with torch.enable_grad():
        for t in range(T):
            latents = [h1, h2, h3, h4_clamped]
            per_sample_total, _, _ = model.bpc_energy(latents, x)
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
                h1 -= eta_h * m1
                h2 -= eta_h * m2
                h3 -= eta_h * m3
            h1.requires_grad_(True)
            h2.requires_grad_(True)
            h3.requires_grad_(True)
    
    # Compute per-layer movement
    with torch.no_grad():
        mov1 = (h1 - h1_init).abs().mean().item()
        mov2 = (h2 - h2_init).abs().mean().item()
        mov3 = (h3 - h3_init).abs().mean().item()
    
    movements = [mov1, mov2, mov3]
    max_movement = max(movements)
    
    return movements, max_movement


# =============================================================================
# Quick verification
# =============================================================================

if __name__ == '__main__':
    print("TinyConvBPC Architecture Test")
    print("=" * 60)
    
    model = TinyConvBPC(num_classes=10, alpha_gen=1e-5, alpha_disc=1.0)
    
    # Count parameters — should match TinyConvPCN exactly
    total_params = sum(p.numel() for p in model.parameters())
    
    v_params = (
        sum(p.numel() for p in model.v_conv1.parameters())
        + sum(p.numel() for p in model.v_bn1.parameters())
        + sum(p.numel() for p in model.v_conv2.parameters())
        + sum(p.numel() for p in model.v_bn2.parameters())
        + sum(p.numel() for p in model.v_fc1.parameters())
        + sum(p.numel() for p in model.v_fc2.parameters())
    )
    
    w_params = (
        sum(p.numel() for p in model.w_fc3.parameters())
        + sum(p.numel() for p in model.w_fc2.parameters())
        + sum(p.numel() for p in model.w_conv1.parameters())
    )
    
    print(f"Total parameters:  {total_params:,}")
    print(f"  V pathway:       {v_params:,}")
    print(f"  W pathway:       {w_params:,}")
    print(f"  Expected total:  2,144,938")
    print(f"  Match: {'YES' if total_params == 2_144_938 else 'NO — CHECK!'}")
    print()
    
    # Test forward V pass
    x = torch.randn(4, 3, 32, 32)
    print("Testing V pathway forward pass...")
    latents = model.forward_v(x)
    for i, h in enumerate(latents):
        print(f"  h{i+1}: {h.shape}")
    
    # Test energy computation
    print("\nTesting bPC energy computation...")
    per_total, per_gen, per_disc = model.bpc_energy(latents, x)
    print(f"  Per-sample total energy: {per_total.shape}, mean={per_total.mean():.6f}")
    print(f"  Generative component:    mean={per_gen.mean():.6f}")
    print(f"  Discriminative component: mean={per_disc.mean():.6f}")
    print(f"  α_gen={model.alpha_gen}, α_disc={model.alpha_disc}")
    
    # Test training step
    print("\nTesting bPC training step (T=3, quick)...")
    y = torch.randint(0, 10, (4,))
    model.train()
    loss, gen_l, disc_l, enc_l = bpc_train_step(model, x, y, T=3)
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Gen loss:   {gen_l:.6f}")
    print(f"  Disc loss:  {disc_l:.6f}")
    print(f"  Enc loss:   {enc_l:.6f}")
    
    # Test K-way classification (T=3 for speed)
    print("\nTesting K-way energy classification (T=3, quick)...")
    model.eval()
    pred, energies, gen_e, disc_e = bpc_classify_energy_based(model, x, T=3)
    print(f"  Predictions: {pred}")
    print(f"  Energy shape: {energies.shape}")
    print(f"  Energy range: [{energies.min():.4f}, {energies.max():.4f}]")
    
    # Test latent movement
    print("\nTesting latent movement measurement (T=3)...")
    movements, max_mov = measure_latent_movement(model, x, T=3)
    print(f"  Per-layer movements: {[f'{m:.6f}' for m in movements]}")
    print(f"  Max movement: {max_mov:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed. Ready for training.")
    print("=" * 60)
