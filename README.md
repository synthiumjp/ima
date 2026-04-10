# IMA: Intrinsic Metacognitive Architecture

Pre-registered on OSF. Error-driven monitoring via predictive coding inference dynamics, evaluated with Type-2 SDT.

## Setup (Windows + AMD RX 7900 GRE + ROCm)

```cmd
:: 1. Clone or copy to D:\ima\
:: 2. Set environment variable (add to system env vars permanently)
set HSA_OVERRIDE_GFX_VERSION=11.0.0

:: 3. Verify PyTorch ROCm (should already work from M2)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

:: 4. Install remaining dependencies
pip install -r requirements.txt

:: 5. Clone Stenlund (2025) reference (for documentation only)
git clone https://github.com/Monadillo/pcn-intro.git reference/pcn-intro

:: 6. Run setup verification
python scripts\verify_setup.py
```

## Phase 0: Confidence Signal Validation

```cmd
:: Train all 3 seeds (pre-registered: 42, 123, 456)
scripts\run_phase0_all.bat

:: Or train individually:
python scripts\train_phase0.py --seed 42

:: Evaluate and run go/no-go
scripts\run_phase0_eval.bat

:: Or evaluate individually:
python scripts\evaluate_phase0.py --seed 42 --all-t
python scripts\evaluate_phase0.py --summarise
```

## Project Structure

```
D:\ima\
├── src/
│   ├── conv_pcn.py          # ConvPCN backbone (~4.2M params)
│   └── constants.py         # Shared constants
├── scripts/
│   ├── verify_setup.py      # Setup verification
│   ├── train_phase0.py      # Phase 0 training
│   ├── evaluate_phase0.py   # Phase 0 eval + go/no-go
│   ├── run_phase0_all.bat   # Batch: train all seeds
│   └── run_phase0_eval.bat  # Batch: eval + go/no-go
├── data/                    # CIFAR-10 (auto-downloaded)
├── checkpoints/phase0/      # Trained models per seed
├── results/phase0/          # Evaluation results + go/no-go
├── logs/                    # Training logs
├── reference/               # Stenlund (2025) repo
└── requirements.txt
```

## Pre-Registration

Registered on OSF (v3.1, locked). 12 external reviews across 4 rounds.
Architecture, hyperparameters, seeds, evaluation protocol, and analysis plan are immutable.
Any deviations documented as amendments per §12.

## Hardware

- GPU: AMD RX 7900 GRE (16GB VRAM, gfx1100)
- Framework: PyTorch + ROCm (native Windows)
- Estimated Phase 0 compute: ~14 GPU-hours
