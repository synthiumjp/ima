# IMA: Intrinsic Metacognitive Architecture

Investigation of K-way energy-based structural probes for metacognition in predictive coding networks. This repository contains code and experimental scripts for two papers in the programme.

**Paper 1** showed that the K-way energy probe on standard discriminative PC reduces to a monotone function of the log-softmax margin. **Paper 2** showed that cross-entropy at the output is the primary load-bearing assumption in that reduction, accounting for approximately two-thirds of the probe-softmax gap through logit-scale inflation and one-third through a genuine ranking advantage.

## Papers

**Paper 1.** Cacioli, J-P. (2026). K-way energy probes for metacognition reduce to softmax in discriminative predictive coding networks. *arXiv:2604.11011*. [https://arxiv.org/abs/2604.11011](https://arxiv.org/abs/2604.11011)

**Paper 2.** Cacioli, J-P. (2026). Cross-entropy is load-bearing: a pre-registered scope test of the K-way energy probe on bidirectional predictive coding. *arXiv preprint, submitted April 2026.*

## Pre-registration

Paper 1 was pre-registered on OSF as v3.1 after four rounds of external review: [osf.io/n2zjp](https://osf.io/n2zjp). The v3.1 pre-registration hypothesised that iterative inference dynamics would produce metacognitive signal beyond feedforward readouts. Empirical work refuted this hypothesis. The pre-registration remains on OSF as a permanent record.

Paper 2 was pre-registered separately on OSF before any training runs: [osf.io/2kvsp](https://osf.io/2kvsp). It specified three conditions (stdPC-CE, stdPC-MSE, bPC), four hypotheses with sequential gating, and all statistical tests. The temperature scaling ablation was not pre-registered and is labelled post-hoc in the paper.

## Repository structure

```
D:\ima\
├── src/
│   ├── cifar10_data.py                 # CIFAR-10 data loader ([0,1] normalisation)
│   └── conv_pcn.py                     # TinyConvPCN architecture
├── scripts/
│   ├── spike_dynamics.py               # Paper 1, C1: standard PC structural probe
│   ├── spike_diagnose_inference.py     # Paper 1, C2: latent movement diagnostic
│   ├── spike_bp_decoder.py             # Paper 1, C3: BP + post-hoc decoder
│   ├── spike_pc_extended.py            # Paper 1, C1: extended 25-epoch PC
│   ├── spike_bp_extended.py            # Paper 1, C4: matched-budget BP control
│   ├── spike_langevin_phase_a.py       # Paper 1, C5: Langevin noisy inference
│   ├── spike_langevin_phase_b.py       # Paper 1, C6: MCPC trajectory-integrated
│   ├── spike_langevin_dynamics.py      # Paper 1, supporting: Langevin dynamics
│   ├── diagnose_wheel.py              # Paper 1, supporting: diagnostic utility
│   ├── check_gpu.py                    # Utility: GPU verification
│   ├── tiny_conv_bpc.py                # Paper 2: TinyConvBPC (Condition C)
│   ├── tiny_conv_pcn_mse.py            # Paper 2: TinyConvPCN-MSE (Condition B)
│   ├── cifar10_data_bpc.py             # Paper 2: [-1,1] normalisation data loader
│   ├── calibration_sweep.py            # Paper 2, Stage 1: alpha_gen sweep
│   ├── verification.py                 # Paper 2, Stage 2: smoke test
│   ├── main_experiment.py              # Paper 2, Stage 3: 30-run main experiment
│   ├── analysis.py                     # Paper 2, Stage 4: pre-registered analysis
│   └── temperature_ablation.py         # Paper 2, post-hoc: temperature scaling
├── results/
│   ├── calibration_sweep.json          # Stage 1 data
│   ├── calibration_sweep_checkpoint.json
│   ├── selected_alpha_gen.txt          # "1e-05"
│   ├── main_experiment.json            # Stage 3 data (30 runs)
│   ├── main_experiment_checkpoint.json
│   ├── temperature_ablation.json       # Temperature scaling data
│   └── phase0/                         # Paper 1 Langevin results
├── data/                               # CIFAR-10 (auto-downloaded on first run)
├── reference/                          # Stenlund (2025) PC-intro reference repo
├── .gitignore
├── ima_env.ps1                         # Environment launcher
├── requirements.txt
└── README.md
```

## Setup (Windows + AMD RX 7900 GRE + ROCm)

```powershell
# 1. Clone
git clone https://github.com/synthiumjp/ima.git D:\ima
cd D:\ima

# 2. Set environment variable
set HSA_OVERRIDE_GFX_VERSION=11.0.0

# 3. Create and activate virtual environment
python -m venv .venv_ima
.venv_ima\Scripts\activate

# 4. Install PyTorch for ROCm Windows
# (AMD ROCm 6.4.4 Windows wheel; see AMD documentation for current install command)

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Critical workaround

The AMD ROCm 6.4.4 Windows wheel has a known issue (ROCm/ROCm#5441) with MIOpen SQLite schema on BatchNorm for RDNA3+ GPUs. All scripts include the following at the top:

```python
import torch
torch.backends.cudnn.enabled = False
```

This must appear before any GPU code. Do not remove it.

## Running the experiments

### Paper 1 (reduction theorem)

Each spike is self-contained and runnable from the repository root:

```powershell
python scripts\spike_pc_extended.py         # C1: extended PC training (~18 min)
python scripts\spike_diagnose_inference.py  # C2: latent movement diagnostic
python scripts\spike_bp_decoder.py          # C3: BP + decoder
python scripts\spike_bp_extended.py         # C4: matched-budget BP (~3 min)
python scripts\spike_langevin_phase_a.py    # C5: Langevin + sigma sweep (~20 min)
python scripts\spike_langevin_phase_b.py    # C6: MCPC training (~20 min)
```

Total: approximately 1.5 GPU-hours.

### Paper 2 (bPC scope test)

Run in order. Each stage depends on the previous:

```powershell
python scripts\calibration_sweep.py         # Stage 1: alpha_gen selection (~26 hrs)
python scripts\verification.py              # Stage 2: smoke test (~5 min)
python scripts\main_experiment.py           # Stage 3: 30 runs (~19 hrs)
python scripts\analysis.py                  # Stage 4: pre-registered analysis (~1 min)
python scripts\temperature_ablation.py      # Post-hoc: temperature scaling (~3 hrs)
```

Total: approximately 48 GPU-hours. Seeds 1 to 5 for calibration, 6 to 15 for the main experiment, ensuring zero seed overlap.

## Key results

### Paper 1

Under standard discriminative PC with CE at the output and effectively feedforward dynamics, the K-way energy margin reduces to a monotone function of the log-softmax margin plus a residual that is structural noise. Six empirical confirmations on CIFAR-10 support the reduction.

### Paper 2

Removing CE allows the probe to match or exceed softmax. Three conditions on 10 seeds:

| Condition | Softmax AUROC₂ | Probe AUROC₂ | Delta |
|-----------|---------------|-------------|-------|
| stdPC-CE  | 0.842 | 0.760 | -0.082 |
| stdPC-MSE | 0.836 | 0.799 | -0.037 |
| bPC       | 0.824 | 0.832 | +0.008 |

Temperature scaling on CE-trained models closes 66% of the gap. The remaining 34% is a scale-invariant ranking advantage of CE as a proper scoring rule.

## Hardware

- GPU: AMD RX 7900 GRE (16GB VRAM, gfx1100)
- Framework: PyTorch 2.8.0 with ROCm 6.4.4 (native Windows)
- OS: Windows 11

## Citation

```bibtex
@article{cacioli2026kway,
  author  = {Cacioli, Jon-Paul},
  title   = {K-Way Energy Probes for Metacognition Reduce to Softmax
             in Discriminative Predictive Coding Networks},
  journal = {arXiv preprint arXiv:2604.11011},
  year    = {2026}
}

@article{cacioli2026ce,
  author  = {Cacioli, Jon-Paul},
  title   = {Cross-Entropy Is Load-Bearing: A Pre-Registered Scope Test
             of the K-Way Energy Probe on Bidirectional Predictive Coding},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Submitted April 2026}
}
```

## Contact

Jon-Paul Cacioli. Independent Researcher, Melbourne, Australia. ORCID 0009-0000-7054-2014.
