"""
IMA: Convolutional Predictive Coding Network — Production Implementation
=========================================================================

Built from: Stenlund (2025, arXiv:2506.06332) PCN reference implementation
Extended to: Convolutional architecture following Pinchetti et al. (2024) VGG pattern

Pre-registration v3.1 architecture spec:
  - Encoder: VGG-style, 4 conv blocks (64→128→256→256), BatchNorm, GELU, MaxPool
  - Generative layers: Transposed convolutions (top-down)
  - FC layer from L4 (256-dim) to L3 (256*4*4)
  - Readout: Linear from L4 (256) to num_classes
  - Amortised latent initialisation (hybrid PC)
  - T_infer=20, eta_infer=0.1
  - ~4.2M parameters

Layer hierarchy:
  L0: (3, 32, 32)   — input, clamped
  L1: (64, 16, 16)   — after block 1 + pool
  L2: (128, 8, 8)    — after block 2 + pool
  L3: (256, 4, 4)    — after block 3 + pool
  L4: (256,)          — FC latent

Prediction errors ε_l = x_l - x_hat_l at L0–L3 are the M pathway input.

Author: JP Cacioli
Date: April 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# ROCm workaround: MIOpen kernel cache has SQLite schema issues on this
# build (PyTorch 2.8.0a0+gitfc14c65, ROCm, gfx1100). Disabling cudnn
# forces fallback kernels for conv/BN ops. Slower but functional.
# Does not affect architecture or numerical results.
torch.backends.cudnn.enabled = False

# CIFAR-10 normalisation (used by training/eval scripts)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# =============================================================================
# Architecture constants (from pre-registration v3.1 §3.2)
# =============================================================================

LAYER_SHAPES = [
    (3, 32, 32),    # L0 — input
    (64, 16, 16),   # L1
    (128, 8, 8),    # L2
    (256, 4, 4),    # L3
    (256,),         # L4 — FC latent
]

# M pathway input: GAP squared errors per channel
# D = 3 + 64 + 128 + 256 = 451
M_INPUT_DIM = sum(s[0] for s in LAYER_SHAPES[:4])  # 451


@dataclass
class InferenceResult:
    """Container for PCN inference output."""
    logits: torch.Tensor        # (B, num_classes)
    probs: torch.Tensor         # (B, num_classes)
    m_input: torch.Tensor       # (B, 451) — aggregated errors for M pathway
    errors: Optional[List[torch.Tensor]] = None      # [ε_0, ..., ε_3]
    energy_trace: Optional[List[float]] = None        # per-step total energy
    error_norms_trace: Optional[List[List[float]]] = None  # per-step per-layer norms
    latents: Optional[List[torch.Tensor]] = None      # settled latents


# =============================================================================
# Generative (top-down) convolutional layer
# =============================================================================

class GenConvLayer(nn.Module):
    """Transposed convolution for top-down generative prediction.
    
    Predicts layer l from layer l+1: x_hat_l = f(ConvT(x_{l+1}))
    Following Stenlund's pattern: layer.forward returns (x_hat, preactivation).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 2,
                 padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding,
            bias=False
        )
        nn.init.xavier_uniform_(self.conv_t.weight)

    def forward(self, x_above: torch.Tensor,
                target_size: Optional[Tuple[int, int]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_above: (B, C_in, H_in, W_in) from layer above
            target_size: (H, W) expected output spatial dims
        Returns:
            x_hat: prediction of layer below (after GELU)
            a: pre-activations (before GELU)
        """
        a = self.conv_t(x_above)
        if target_size is not None:
            a = a[:, :, :target_size[0], :target_size[1]]
        x_hat = F.gelu(a)
        return x_hat, a


# =============================================================================
# Convolutional PCN
# =============================================================================

class ConvPCN(nn.Module):
    """Convolutional Predictive Coding Network for image classification.
    
    Follows Stenlund (2025) PCN structure with convolutional extension.
    Key IMA integration points:
      - compute_errors() returns ε_l = x_l - x_hat_l at all layers
      - aggregate_errors_for_m() produces 451-dim M pathway input
      - classify() runs full discriminative inference (no target clamping)
      - Energy tracking for sanity checks and confidence signals
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # =================================================================
        # Feedforward encoder (amortised latent initialisation)
        # Hybrid PC: encoder gives good starting point, PC refines.
        # =================================================================
        self.encoder = nn.Sequential(
            # Block 1: 3→64, 32×32 → pool → 64, 16×16
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2: 64→128, 16×16 → pool → 128, 8×8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3: 128→256, 8×8 → pool → 256, 4×4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
            # Block 4: 256→256, 4×4 (no pool)
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.fc_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.GELU(),
        )

        # Store encoder block boundaries for latent extraction
        # Block 1 ends at index 6 (after MaxPool2d)
        # Block 2 ends at index 13
        # Block 3 ends at index 20
        # Block 4 ends at index 23
        self._encoder_layers = list(self.encoder.children())

        # =================================================================
        # Generative layers (top-down predictions)
        # Following Stenlund: layer l predicts layer l-1
        # =================================================================
        # L4 (256,) → L3 (256, 4, 4): FC
        self.gen_fc = nn.Linear(256, 256 * 4 * 4, bias=False)
        nn.init.xavier_uniform_(self.gen_fc.weight)

        # L3 (256, 4, 4) → L2 (128, 8, 8): ConvT with stride 2
        self.gen_3to2 = GenConvLayer(256, 128)
        # L2 (128, 8, 8) → L1 (64, 16, 16): ConvT with stride 2
        self.gen_2to1 = GenConvLayer(128, 64)
        # L1 (64, 16, 16) → L0 (3, 32, 32): ConvT with stride 2
        self.gen_1to0 = GenConvLayer(64, 3)

        # =================================================================
        # Readout: L4 → class logits
        # =================================================================
        self.readout_layer = nn.Linear(256, num_classes, bias=False)
        nn.init.xavier_uniform_(self.readout_layer.weight)

        self.L = 4  # number of latent layers

    # -----------------------------------------------------------------
    # Latent initialisation
    # -----------------------------------------------------------------

    def init_latents_amortised(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Initialise latents via feedforward encoder (hybrid PC).
        
        Returns [L1, L2, L3, L4], each detached (inference is manual).
        """
        latents = []
        h = x

        # Block 1 → L1 (64, 16, 16)
        for layer in self._encoder_layers[:7]:  # conv,bn,gelu,conv,bn,gelu,maxpool
            h = layer(h)
        latents.append(h.detach().clone())

        # Block 2 → L2 (128, 8, 8)
        for layer in self._encoder_layers[7:14]:
            h = layer(h)
        latents.append(h.detach().clone())

        # Block 3 → L3 (256, 4, 4)
        for layer in self._encoder_layers[14:21]:
            h = layer(h)
        latents.append(h.detach().clone())

        # Block 4 + FC → L4 (256,)
        for layer in self._encoder_layers[21:]:
            h = layer(h)
        h = self.fc_encoder(h)
        latents.append(h.detach().clone())

        return latents

    # -----------------------------------------------------------------
    # Prediction errors — the core of PCN and the M pathway input
    # Following Stenlund: errors[l] = inputs_latents[l] - x_hat[l]
    # -----------------------------------------------------------------

    def compute_errors(self, x: torch.Tensor, latents: List[torch.Tensor]
                       ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute prediction errors ε_l = x_l - x_hat_l at all layers.
        
        Following Stenlund's compute_errors pattern exactly.
        
        Args:
            x: Input images (L0), (B, 3, 32, 32)
            latents: [L1, L2, L3, L4]
        Returns:
            errors: [ε_0, ε_1, ε_2, ε_3] (bottom to top)
            preactivations: [a_0, a_1, a_2, a_3] (for gain modulation)
        """
        all_layers = [x] + latents  # [L0, L1, L2, L3, L4]
        errors = []
        preacts = []

        # L4 → L3: FC generative prediction
        a3 = self.gen_fc(all_layers[4]).view(-1, 256, 4, 4)
        x_hat_3 = F.gelu(a3)
        errors.append(all_layers[3] - x_hat_3)   # ε_3
        preacts.append(a3)

        # L3 → L2
        x_hat_2, a2 = self.gen_3to2(all_layers[3], target_size=(8, 8))
        errors.append(all_layers[2] - x_hat_2)   # ε_2
        preacts.append(a2)

        # L2 → L1
        x_hat_1, a1 = self.gen_2to1(all_layers[2], target_size=(16, 16))
        errors.append(all_layers[1] - x_hat_1)   # ε_1
        preacts.append(a1)

        # L1 → L0
        x_hat_0, a0 = self.gen_1to0(all_layers[1], target_size=(32, 32))
        errors.append(all_layers[0] - x_hat_0)   # ε_0
        preacts.append(a0)

        # Reverse: errors[0]=ε_0 (bottom), errors[3]=ε_3 (top)
        errors.reverse()
        preacts.reverse()

        return errors, preacts

    # -----------------------------------------------------------------
    # PC inference — Stenlund-style latent update
    # -----------------------------------------------------------------

    def _gelu_deriv(self, a: torch.Tensor) -> torch.Tensor:
        """Approximate GELU derivative for gain modulation."""
        s = torch.sigmoid(1.702 * a)
        return s * (1.0 + 1.702 * a * (1.0 - s))

    def infer(self, x: torch.Tensor,
              y_onehot: Optional[torch.Tensor],
              latents: List[torch.Tensor],
              T: int = 20, eta: float = 0.1,
              track_energy: bool = False,
              track_error_norms: bool = False
              ) -> Tuple[List[torch.Tensor], Optional[List[float]],
                         Optional[List[List[float]]]]:
        """Run PC inference: iteratively update latents to minimise energy.
        
        Follows Stenlund's inference loop structure:
          1. Compute errors and gain-modulated errors
          2. Compute supervised error from readout
          3. Update latents: X_l -= eta * (ε_l - gm_ε_{l-1} @ W_{l-1})
        
        Args:
            x: Input (L0), clamped. (B, 3, 32, 32)
            y_onehot: Target labels for training (None at eval)
            latents: [L1, L2, L3, L4] initial states
            T: Inference iterations
            eta: Inference learning rate
            track_energy: Record total energy per step
            track_error_norms: Record per-layer ‖ε_l‖ per step
        Returns:
            latents: Updated after T steps
            energy_trace: [E_0, ..., E_T] if tracked
            error_norms_trace: [[‖ε_0‖,...,‖ε_3‖] at each step] if tracked
        """
        B = x.size(0)
        energy_trace = [] if track_energy else None
        error_norms_trace = [] if track_error_norms else None

        with torch.no_grad():
            for t in range(T):
                # Step 1: Compute prediction errors
                errors, preacts = self.compute_errors(x, latents)

                # Step 2: Supervised error
                logits = self.readout_layer(latents[3])  # L4 → logits
                if y_onehot is not None:
                    eps_sup = logits - y_onehot
                else:
                    # Eval: self-prediction (drives toward consistent state)
                    eps_sup = logits - F.softmax(logits, dim=-1)

                # Track energy: 0.5 * Σ‖ε_l‖² + 0.5 * ‖eps_sup‖²
                if track_energy:
                    lat_e = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    sup_e = 0.5 * eps_sup.pow(2).sum().item() / B
                    energy_trace.append(lat_e + sup_e)

                if track_error_norms:
                    norms = [e.pow(2).sum().item() ** 0.5 / B for e in errors]
                    error_norms_trace.append(norms)

                # Step 3: Gain-modulated errors for backprojection
                gm_errors = [e * self._gelu_deriv(a) for e, a in zip(errors, preacts)]

                # Step 4: Latent updates (Stenlund Eq. — adapted for conv)
                # L4: grad = eps_sup @ W_readout + gm_ε_3 backprojected through gen_fc
                grad_L4 = eps_sup @ self.readout_layer.weight
                gm_3_flat = gm_errors[3].view(B, -1)  # (B, 256*4*4)
                grad_L4 = grad_L4 + gm_3_flat @ self.gen_fc.weight
                latents[3] = latents[3] - eta * grad_L4

                # L3: grad = ε_3 - gm_ε_2 backprojected through gen_3to2
                grad_L3 = errors[3]
                bp_2 = F.conv2d(gm_errors[2], self.gen_3to2.conv_t.weight,
                                stride=2, padding=1)
                bp_2 = bp_2[:, :, :4, :4]  # match L3 spatial size
                grad_L3 = grad_L3 - bp_2
                latents[2] = latents[2] - eta * grad_L3

                # L2: grad = ε_2 - gm_ε_1 backprojected through gen_2to1
                grad_L2 = errors[2]
                bp_1 = F.conv2d(gm_errors[1], self.gen_2to1.conv_t.weight,
                                stride=2, padding=1)
                bp_1 = bp_1[:, :, :8, :8]
                grad_L2 = grad_L2 - bp_1
                latents[1] = latents[1] - eta * grad_L2

                # L1: grad = ε_1 - gm_ε_0 backprojected through gen_1to0
                grad_L1 = errors[1]
                bp_0 = F.conv2d(gm_errors[0], self.gen_1to0.conv_t.weight,
                                stride=2, padding=1)
                bp_0 = bp_0[:, :, :16, :16]
                grad_L1 = grad_L1 - bp_0
                latents[0] = latents[0] - eta * grad_L1

        # Record final-step energy/norms
        if track_energy or track_error_norms:
            errors_final, _ = self.compute_errors(x, latents)
            logits_final = self.readout_layer(latents[3])
            if y_onehot is not None:
                eps_sup_final = logits_final - y_onehot
            else:
                eps_sup_final = logits_final - F.softmax(logits_final, dim=-1)
            if track_energy:
                lat_e = 0.5 * sum(e.pow(2).sum().item() for e in errors_final) / B
                sup_e = 0.5 * eps_sup_final.pow(2).sum().item() / B
                energy_trace.append(lat_e + sup_e)
            if track_error_norms:
                norms = [e.pow(2).sum().item() ** 0.5 / B for e in errors_final]
                error_norms_trace.append(norms)

        return latents, energy_trace, error_norms_trace

    # -----------------------------------------------------------------
    # M pathway input aggregation (pre-reg §4.2: GAP squared errors)
    # -----------------------------------------------------------------

    def aggregate_errors_for_m(self, errors: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate per-layer errors into 451-dim M pathway input.
        
        Pre-reg §4.2: "global average pooled squared errors per channel:
        3 + 64 + 128 + 256 = 451"
        
        Args:
            errors: [ε_0 (3,32,32), ε_1 (64,16,16), ε_2 (128,8,8), ε_3 (256,4,4)]
        Returns:
            (B, 451) — mean squared error per channel per layer
        """
        parts = []
        for eps in errors:
            if eps.dim() == 4:  # (B, C, H, W)
                parts.append(eps.pow(2).mean(dim=(2, 3)))  # (B, C)
            elif eps.dim() == 2:  # (B, D)
                parts.append(eps.pow(2))
            else:
                raise ValueError(f"Unexpected error shape: {eps.shape}")
        return torch.cat(parts, dim=1)  # (B, 451)

    # -----------------------------------------------------------------
    # Discriminative classification (eval mode — no target clamping)
    # -----------------------------------------------------------------

    def classify(self, x: torch.Tensor,
                 T_infer: int = 20, eta_infer: float = 0.1,
                 return_errors: bool = False,
                 return_energy_trace: bool = False,
                 return_error_norms_trace: bool = False,
                 return_latents: bool = False) -> InferenceResult:
        """Run full discriminative inference pipeline.
        
        1. Amortised init from encoder
        2. PC inference (no target clamping)
        3. Readout from settled L4
        4. Extract prediction errors for M pathway
        """
        latents = self.init_latents_amortised(x)
        latents, e_trace, en_trace = self.infer(
            x, y_onehot=None, latents=latents,
            T=T_infer, eta=eta_infer,
            track_energy=return_energy_trace,
            track_error_norms=return_error_norms_trace,
        )

        logits = self.readout_layer(latents[3])
        probs = F.softmax(logits, dim=-1)

        errors, _ = self.compute_errors(x, latents)
        m_input = self.aggregate_errors_for_m(errors)

        return InferenceResult(
            logits=logits,
            probs=probs,
            m_input=m_input,
            errors=errors if return_errors else None,
            energy_trace=e_trace,
            error_norms_trace=en_trace,
            latents=latents if return_latents else None,
        )


# =============================================================================
# Training functions
# =============================================================================

def train_pcn_epoch(model: ConvPCN, data_loader,
                    optimizer: torch.optim.Optimizer,
                    T_infer: int = 20, eta_infer: float = 0.1,
                    device: str = 'cuda') -> Dict[str, float]:
    """Train ConvPCN for one epoch.
    
    Alternating inference/learning following Stenlund (2025):
      1. Amortised init
      2. PC inference with target clamping (T steps)
      3. Weight update: generative weights via PC rule, encoder via MSE to settled
    
    Pre-reg §3.4: AdamW, lr=1e-3, weight_decay=1e-4, batch_size=128
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        B = x_batch.size(0)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()

        # Phase 1: Amortised init
        latents = model.init_latents_amortised(x_batch)

        # Phase 2: PC inference with target clamping
        latents, _, _ = model.infer(
            x_batch, y_onehot=y_onehot, latents=latents,
            T=T_infer, eta=eta_infer,
        )

        # Phase 3: Weight update
        optimizer.zero_grad()

        # 3a: Generative weight + readout loss
        # Recompute errors WITH gradients for generative weights
        errors, _ = model.compute_errors(x_batch, [l.detach() for l in latents])
        logits = model.readout_layer(latents[3].detach())

        latent_energy = 0.5 * sum(e.pow(2).sum() for e in errors) / B
        sup_loss = F.cross_entropy(logits, y_batch)

        # 3b: Encoder loss — match encoder output to settled latents
        # Run encoder forward WITH gradients (don't use init_latents_amortised
        # which detaches). Instead run through the encoder directly.
        h = x_batch
        enc_outputs = []
        for layer in model._encoder_layers[:7]:
            h = layer(h)
        enc_outputs.append(h)  # L1 target
        for layer in model._encoder_layers[7:14]:
            h = layer(h)
        enc_outputs.append(h)  # L2 target
        for layer in model._encoder_layers[14:21]:
            h = layer(h)
        enc_outputs.append(h)  # L3 target
        for layer in model._encoder_layers[21:]:
            h = layer(h)
        h = model.fc_encoder(h)
        enc_outputs.append(h)  # L4 target

        enc_loss = sum(
            F.mse_loss(enc_l, settled_l.detach())
            for enc_l, settled_l in zip(enc_outputs, latents)
        )

        total_loss_val = latent_energy + sup_loss + enc_loss
        total_loss_val.backward()
        optimizer.step()

        total_loss += (latent_energy.item() + sup_loss.item()) * B
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += B

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def evaluate_pcn(model: ConvPCN, data_loader,
                 T_infer: int = 20, eta_infer: float = 0.1,
                 device: str = 'cuda',
                 collect_signals: bool = False) -> Dict:
    """Evaluate PCN — discriminative, no target clamping.
    
    When collect_signals=True, records per-trial data for Phase 0
    confidence signal analysis (pre-reg §3.5).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    trials = [] if collect_signals else None

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        B = x_batch.size(0)

        result = model.classify(
            x_batch, T_infer=T_infer, eta_infer=eta_infer,
            return_errors=collect_signals,
            return_energy_trace=collect_signals,
            return_error_norms_trace=collect_signals,
        )

        preds = result.logits.argmax(dim=1)
        correct = (preds == y_batch)
        total_correct += correct.sum().item()
        total_samples += B

        if collect_signals:
            for i in range(B):
                trial = {
                    'correct': correct[i].item(),
                    'pred': preds[i].item(),
                    'target': y_batch[i].item(),
                    # Signal 5: max softmax probability
                    'max_prob': result.probs[i].max().item(),
                    # Signal 6: negative entropy
                    'neg_entropy': (result.probs[i] * result.probs[i].clamp(min=1e-12).log()).sum().item(),
                    # Signal 3: negative SSE (sum of squared errors)
                    'neg_sse': -sum(
                        e[i].pow(2).sum().item() for e in result.errors
                    ),
                    # Signal 2: negative residual energy
                    'neg_residual_energy': -result.energy_trace[-1] if result.energy_trace else None,
                    # Signal 1: energy decay rate (E_{T-1} - E_T, higher = more settled)
                    'energy_decay_rate': (
                        result.energy_trace[-2] - result.energy_trace[-1]
                        if result.energy_trace and len(result.energy_trace) >= 2
                        else None
                    ),
                    # Signal 4: per-layer error norms
                    'error_norms': [
                        e[i].pow(2).sum().item() ** 0.5 for e in result.errors
                    ],
                    # Full energy trace (for sanity checks)
                    'energy_trace': result.energy_trace,
                    # M pathway input (for later use)
                    'm_input': result.m_input[i].cpu(),
                }
                # Error norms trace for non-degenerate inference check
                if result.error_norms_trace:
                    trial['error_norms_t0'] = result.error_norms_trace[0]
                    trial['error_norms_tT'] = result.error_norms_trace[-1]

                trials.append(trial)

    output = {'accuracy': total_correct / total_samples}
    if collect_signals:
        output['trials'] = trials
    return output


# =============================================================================
# Self-test
# =============================================================================

if __name__ == '__main__':
    print("ConvPCN Architecture Verification")
    print("=" * 60)

    model = ConvPCN(num_classes=10)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    enc_params = (sum(p.numel() for p in model.encoder.parameters())
                  + sum(p.numel() for p in model.fc_encoder.parameters()))
    gen_params = (sum(p.numel() for p in model.gen_fc.parameters())
                  + sum(p.numel() for p in model.gen_3to2.parameters())
                  + sum(p.numel() for p in model.gen_2to1.parameters())
                  + sum(p.numel() for p in model.gen_1to0.parameters()))
    readout_params = sum(p.numel() for p in model.readout_layer.parameters())

    print(f"Total parameters:     {total:,}")
    print(f"  Encoder:            {enc_params:,}")
    print(f"  Generative:         {gen_params:,}")
    print(f"  Readout:            {readout_params:,}")
    print(f"  Expected ~4.2M:     {'✓' if 3_500_000 < total < 5_000_000 else '✗ MISMATCH'}")
    print()

    # Forward pass test
    x = torch.randn(4, 3, 32, 32)

    print("Amortised init...")
    latents = model.init_latents_amortised(x)
    for i, l in enumerate(latents):
        expected = LAYER_SHAPES[i + 1]
        actual = tuple(l.shape[1:])
        match = '✓' if actual == expected else f'✗ expected {expected}'
        print(f"  L{i+1}: {l.shape} {match}")

    print("\nError computation...")
    errors, preacts = model.compute_errors(x, latents)
    for i, e in enumerate(errors):
        expected = LAYER_SHAPES[i]
        actual = tuple(e.shape[1:])
        match = '✓' if actual == expected else f'✗ expected {expected}'
        print(f"  ε_{i}: {e.shape} {match}")

    print("\nM pathway input...")
    m_input = model.aggregate_errors_for_m(errors)
    print(f"  Shape: {m_input.shape}")
    print(f"  Expected dim: {M_INPUT_DIM}")
    print(f"  Match: {'✓' if m_input.shape[1] == M_INPUT_DIM else '✗ MISMATCH'}")

    print("\nClassification (T=5)...")
    result = model.classify(x, T_infer=5, return_energy_trace=True,
                            return_error_norms_trace=True, return_errors=True)
    print(f"  Logits: {result.logits.shape}")
    print(f"  M input: {result.m_input.shape}")
    print(f"  Energy trace length: {len(result.energy_trace)}")
    print(f"  Energy monotonic: {all(result.energy_trace[i] >= result.energy_trace[i+1] for i in range(len(result.energy_trace)-1))}")

    print("\n✓ Architecture verification complete.")
