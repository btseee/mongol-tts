import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs
print(torch.cuda.get_device_name(0)) 
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version
print(torch.backends.cudnn.version())  # Check cuDNN version
