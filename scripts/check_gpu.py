import torch
cuda = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if cuda else "N/A"
print(f"  torch: {torch.__version__}, CUDA: {cuda}, device: {device}")
