$env:HSA_OVERRIDE_GFX_VERSION="11.0.0"
$env:MIOPEN_DISABLE_CACHE=1
C:\sdt_calibration\.venv_train\Scripts\Activate.ps1
cd D:\ima
Write-Host "IMA environment ready. GPU:" -ForegroundColor Green
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''N/A''}')"
