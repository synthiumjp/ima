# Archive: 20260410 Pre-Paper-1 Pivot

Files in this directory are from the initial IMA implementation attempt on
10 April 2026, before the strategic repositioning into a two-paper plan.
They were built against pre-registration v3.1, which has since been
superseded by v4 (Paper 1).

## Why these files are here, not deleted

1. Historical record of the design decisions and their failure modes
2. Reference for the v3.1-era pre-registration on OSF
3. The cifar10_data.py and project scaffolding they depended on are still
   in use; these files document the rest of the v3.1 stack that was discarded

## What each file was

- `conv_pcn.py` — ConvPCN backbone with Stenlund-style inference. Contained
  three attempted training protocols:
    (a) v3.1 as-written (no target at eval): failed, chance accuracy
    (b) Protocol 4a (no target at train either): failed, no discriminative coupling
    (c) Option A energy-based K-way inference: failed, hypothesis energies
        differed by <0.1% due to scale mismatch between per-layer generative
        errors and supervised error
  Also contained workarounds no longer needed with the AMD ROCm wheel:
  torch.backends.cudnn.enabled = False, HSA_OVERRIDE_GFX_VERSION env vars

- `train_phase0.py`, `evaluate_phase0.py` — v3.1 training and eval scripts
- `verify_setup.py` — initial GPU verification script (source-built PyTorch era)
- `run_phase0_all.bat`, `run_phase0_eval.bat` — batch runners for v3.1 Phase 0
- `best_model_v3_1_aborted.pt` — checkpoint from seed 42 training run that was
  aborted at epoch 11 (train 98%, test 10% chance) when the v3.1 protocol's
  "no target at eval" specification was found to be incompatible with standard
  discriminative PC formulations
- `training_log_v3_1_aborted.json` — training log for the above
- `constants.py` — v3.1-era constants including M pathway input dimensions,
  layer shapes, etc. These values are still correct for the architecture
  but need review against v4's pre-registration before reuse

## Key lesson from this attempt

The v3.1 pre-registration specified "discriminative evaluation — NO target
label clamping", which is mathematically impossible in standard PCN
formulations. At test time in the Pinchetti (2024) / JPC / Whittington &
Bogacz style, discriminative PC networks are equivalent to feedforward
passes — iterative inference dynamics exist only during training. This
foundational issue was not caught during pre-reg drafting. Lesson for v4:
before locking in a mechanistic claim, verify that the claim is testable
under the standard formulation of the target architecture.

See session_log_10apr2026_s3.md in the main session logs for the full
scientific reasoning.
