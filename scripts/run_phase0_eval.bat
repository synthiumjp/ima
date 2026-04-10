@echo off
REM IMA Phase 0: Evaluate all seeds and run go/no-go
REM Run from D:\ima\ after training completes

set HSA_OVERRIDE_GFX_VERSION=11.0.0
set MIOPEN_DISABLE_CACHE=1

echo ============================================================
echo IMA Phase 0 — Evaluation and Go/No-Go
echo ============================================================
echo.

echo [1/4] Evaluating seed 42 at T={5,10,15,20}...
python scripts\evaluate_phase0.py --seed 42 --all-t --data-dir data --checkpoint-dir checkpoints\phase0 --output-dir results\phase0
echo.

echo [2/4] Evaluating seed 123 at T={5,10,15,20}...
python scripts\evaluate_phase0.py --seed 123 --all-t --data-dir data --checkpoint-dir checkpoints\phase0 --output-dir results\phase0
echo.

echo [3/4] Evaluating seed 456 at T={5,10,15,20}...
python scripts\evaluate_phase0.py --seed 456 --all-t --data-dir data --checkpoint-dir checkpoints\phase0 --output-dir results\phase0
echo.

echo [4/4] Running go/no-go decision...
python scripts\evaluate_phase0.py --summarise --output-dir results\phase0
echo.

echo ============================================================
echo Phase 0 complete. Check results\phase0\go_nogo_decision.json
echo ============================================================
pause
