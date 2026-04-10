# IMA environment activation script
#
# Uses AMD's prebuilt ROCm PyTorch wheel (rocm-rel-6.4.4) on native Windows.
# No WSL, no source build, no environment overrides needed.

C:\sdt_calibration\.venv_ima\Scripts\Activate.ps1
cd D:\ima
Write-Host "IMA environment ready. GPU:" -ForegroundColor Green
python scripts\check_gpu.py
