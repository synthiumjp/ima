# IMA: Intrinsic Metacognitive Architecture

Investigation of K-way energy-based structural probes for metacognition in discriminative predictive coding networks. This repository contains the code, experimental scripts, and paper draft for a research programme that began with a dynamics-based hypothesis (pre-registered on OSF as v3.1) and, through empirical investigation, converged on a theoretical reduction showing why the entire class of structural probes on standard discriminative PCNs is bounded above by softmax on the same network.

**Current status:** Negative-result paper in draft. arXiv submission pending.

## What this repository contains

- **`scripts/`** — seven experimental spike scripts that together characterise the structural probe's behaviour across training procedures, inference protocols, and temperatures (see §4 of the paper for the scientific content of each spike)
- **`src/`** — shared infrastructure: CIFAR-10 data loader, TinyConvPCN architecture, utilities
- **`docs/`** — paper draft and supporting documentation

## Paper

**"K-Way Energy Probes for Metacognition Reduce to Softmax in Discriminative Predictive Coding Networks"** (Cacioli, 2026)

Draft: `docs/ima_paper_draft_v2.md`. arXiv link to be added on submission.

The paper proves that under standard discriminative PC with target-clamped CE-energy training and effectively-feedforward latent dynamics, the K-way energy margin reduces to a monotone function of the log-softmax margin plus a residual term that is structural noise. The reduction is confirmed empirically across four probe-vs-softmax comparison conditions and two supporting checks on CIFAR-10. Two qualitatively different training procedures — final-state weight updates and trajectory-integrated MCPC — produce probes whose AUROC₂ values differ by less than 10⁻³ at deterministic evaluation, consistent with the prediction that the probe ceiling is set by architecture, not by training procedure.

## Pre-registration history and scientific lineage

This project was originally pre-registered on OSF as v3.1 after four rounds of external review: [osf.io/n2zjp](https://osf.io/n2zjp/overview). The v3.1 pre-registration hypothesised that iterative inference dynamics in a predictive coding network would produce metacognitive signal beyond what feedforward readouts could provide. When initial experiments showed that standard discriminative PC inference is effectively a no-op at test time (latent movement on the order of 10⁻⁴ per element over 13 inference steps), v3.1's mechanism was superseded. Subsequent exploratory work pivoted to a K-way energy probe formulation, which the present paper refutes theoretically and empirically.

The v3.1 pre-registration remains on OSF as a permanent record of the original hypothesis and the documented sequence of revisions that led to the negative result. §1.5 of the paper discusses the pre-registration lineage as part of the credibility argument for the negative result.

## Experimental scripts

Each script corresponds to one confirmation in the paper:

| Script | Confirmation | Paper section | What it tests |
|--------|:------------:|:-------------:|--------------|
| `scripts/spike_dynamics.py`              | C1 | §4.2 | Standard PC structural probe (original spike) |
| `scripts/spike_diagnose_inference.py`    | C2 | §4.3 | Direct measurement of latent movement during inference |
| `scripts/spike_bp_decoder.py`            | C3 | §4.4 | BP + post-hoc trained decoder fairness control |
| `scripts/spike_pc_extended.py`           | C1 | §4.2 | Extended 25-epoch PC training with checkpoints |
| `scripts/spike_bp_extended.py`           | C4 | §4.5 | Matched-budget BP control |
| `scripts/spike_langevin_phase_a.py`      | C5 | §4.6 | Langevin-noisy inference with post-hoc σ sweep |
| `scripts/spike_langevin_phase_b.py`      | C6 | §4.7 | MCPC trajectory-integrated training |

All scripts use `seed=42` and train on CIFAR-10. Results are deterministic on the same hardware and software stack.

## Setup (Windows + AMD RX 7900 GRE + ROCm)

```cmd
:: 1. Clone to D:\ima\
git clone https://github.com/synthiumjp/ima.git D:\ima
cd D:\ima

:: 2. Set environment variable (add to system env vars permanently)
set HSA_OVERRIDE_GFX_VERSION=11.0.0

:: 3. Create and activate virtual environment
python -m venv .venv_ima
.venv_ima\Scripts\activate

:: 4. Install PyTorch for ROCm Windows
:: (AMD ROCm 6.4.4 Windows wheel; see AMD documentation for current install command)

:: 5. Install remaining dependencies
pip install -r requirements.txt

:: 6. Verify PyTorch ROCm
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Critical workaround

The AMD ROCm 6.4.4 Windows wheel has a known issue (ROCm/ROCm#5441) with MIOpen SQLite schema on BatchNorm training for RDNA3+ GPUs. All spike scripts include the following at the top of the file to work around it:

```python
import torch
torch.backends.cudnn.enabled = False
```

This must appear before any GPU code is executed. Do not remove it from the spike scripts.

## Running the spikes

Each spike is self-contained and runnable from the repository root with the environment activated:

```cmd
:: Activate environment
.venv_ima\Scripts\activate

:: Run individual spikes
python scripts\spike_pc_extended.py         :: C1: extended PC training (~18 min)
python scripts\spike_diagnose_inference.py  :: C2: latent movement diagnostic
python scripts\spike_bp_decoder.py          :: C3: BP + decoder
python scripts\spike_bp_extended.py         :: C4: matched-budget BP (~3 min)
python scripts\spike_langevin_phase_a.py    :: C5: Langevin + σ sweep (~20 min)
python scripts\spike_langevin_phase_b.py    :: C6: MCPC training (~20 min)
```

Total reproduction time: approximately 1.5 hours on an RX 7900 GRE.

## Hardware

- **GPU:** AMD RX 7900 GRE (16GB VRAM, gfx1100)
- **Framework:** PyTorch 2.8.0 with ROCm 6.4.4 (native Windows)
- **OS:** Windows 11
- **Total compute for all six confirmations:** approximately 1.5 GPU-hours

## Repository structure

```
D:\ima\
├── src/
│   ├── cifar10_data.py      # CIFAR-10 data loader
│   └── ...                  # shared utilities
├── scripts/
│   ├── spike_dynamics.py               # C1 original spike
│   ├── spike_diagnose_inference.py     # C2 latent movement
│   ├── spike_bp_decoder.py             # C3 BP+decoder
│   ├── spike_pc_extended.py            # C1 extended PC
│   ├── spike_bp_extended.py            # C4 matched-budget BP
│   ├── spike_langevin_phase_a.py       # C5 Langevin
│   └── spike_langevin_phase_b.py       # C6 MCPC
├── docs/
│   └── ima_paper_draft_v2.md           # paper draft
├── data/                    # CIFAR-10 (auto-downloaded on first run)
├── reference/               # Stenlund (2025) PC-intro reference repo
└── requirements.txt
```

## Citation

If you use this code or build on this work, please cite the arXiv preprint (link to be added on submission). Until then, the pre-registration on OSF can be cited as:

> Cacioli, J. (2026). Intrinsic Metacognitive Architecture: Error-Driven Monitoring via Predictive Coding Inference Dynamics, Evaluated with Type-2 Signal Detection Theory. Pre-registration v3.1, Open Science Framework. https://osf.io/n2zjp

The pre-registration describes the originally hypothesised mechanism, which this project's empirical work refuted. The paper (when posted) should be cited as the authoritative description of the negative result.

## Contact

JP Cacioli — Independent Researcher, Melbourne, Australia — ORCID 0009-0000-7054-2014
