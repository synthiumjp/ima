@echo off
REM IMA Phase 0: Train PCN at all 3 pre-registered seeds
REM Run from D:\ima\
REM
REM Pre-registration v3.1: Seeds 42, 123, 456
REM Estimated time: ~14 GPU-hours total on RX 7900 GRE

set HSA_OVERRIDE_GFX_VERSION=11.0.0
set MIOPEN_DISABLE_CACHE=1

echo ============================================================
echo IMA Phase 0 — Training all seeds
echo ============================================================
echo.

echo [1/3] Training seed 42...
python scripts\train_phase0.py --seed 42 --data-dir data --output-dir checkpoints\phase0
if %ERRORLEVEL% neq 0 (
    echo ERROR: Seed 42 training failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Training seed 123...
python scripts\train_phase0.py --seed 123 --data-dir data --output-dir checkpoints\phase0
if %ERRORLEVEL% neq 0 (
    echo ERROR: Seed 123 training failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Training seed 456...
python scripts\train_phase0.py --seed 456 --data-dir data --output-dir checkpoints\phase0
if %ERRORLEVEL% neq 0 (
    echo ERROR: Seed 456 training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo All seeds trained. Now run evaluation:
echo   python scripts\evaluate_phase0.py --seed 42 --all-t
echo   python scripts\evaluate_phase0.py --seed 123 --all-t
echo   python scripts\evaluate_phase0.py --seed 456 --all-t
echo   python scripts\evaluate_phase0.py --summarise
echo ============================================================
pause
