import torch

def get_auto_batch_size(factor=0.5, min_batch=2, max_batch=64):
    if not torch.cuda.is_available():
        return min_batch

    gpu_memory = torch.cuda.get_device_properties(0).total_memory  
    available_memory = torch.cuda.memory_allocated(0)
    free_memory = gpu_memory - available_memory 

    estimated_batch_size = int((free_memory / (1024 ** 3)) * factor) 
    return max(min_batch, min(estimated_batch_size, max_batch))  