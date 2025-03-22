import torch
import os

# Function to calculate batch size based on GPU memory
def get_auto_batch_size(is_training=True, min_batch=2, max_batch=64):
    """
    Automatically determines batch size based on available GPU memory.
    
    Args:
        is_training (bool): If True, optimize for training; if False, for evaluation.
        min_batch (int): Minimum batch size to ensure functionality.
        max_batch (int): Maximum batch size to prevent memory overflow.
    
    Returns:
        int: Calculated batch size.
    """
    if not torch.cuda.is_available():
        return min_batch  # Fallback to minimum if no GPU is available

    # Get total GPU memory and currently allocated memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory  # in bytes
    available_memory = torch.cuda.memory_allocated(0)  # in bytes
    free_memory = gpu_memory - available_memory  # in bytes

    # Adjust factor based on training or evaluation
    # Training needs more memory for gradients, so use a lower factor
    factor = 0.5 if is_training else 0.8  # Higher factor for eval allows larger batches

    # Convert free memory to GB and estimate batch size
    estimated_batch_size = int((free_memory / (1024 ** 3)) * factor)
    
    # Clamp the batch size between min_batch and max_batch
    return max(min_batch, min(estimated_batch_size, max_batch))

# Function to determine the number of data loader workers based on CPU cores
def get_auto_num_workers(is_training=True):
    """
    Automatically sets the number of data loader workers based on CPU cores.
    
    Args:
        is_training (bool): If True, optimize for training; if False, for evaluation.
    
    Returns:
        int: Number of workers.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4  # Default fallback if CPU count is unavailable

    # More workers for training (data-intensive), fewer for evaluation
    return min(8, cpu_count) if is_training else min(4, cpu_count)

# Example usage in a configuration
def setup_config():
    """
    Sets up a configuration with automatically determined parameters.
    
    Returns:
        dict: Configuration with batch sizes and worker numbers.
    """
    # Calculate parameters
    batch_size = get_auto_batch_size(is_training=True, max_batch=64)
    eval_batch_size = get_auto_batch_size(is_training=False, max_batch=128)  # Larger max for eval
    num_loader_workers = get_auto_num_workers(is_training=True)
    num_eval_loader_workers = get_auto_num_workers(is_training=False)

    # Example configuration dictionary
    config = {
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_loader_workers": num_loader_workers,
        "num_eval_loader_workers": num_eval_loader_workers
    }
    
    return config

# Test the configuration
if __name__ == "__main__":
    config = setup_config()
    print("Configuration:")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  eval_batch_size: {config['eval_batch_size']}")
    print(f"  num_loader_workers: {config['num_loader_workers']}")
    print(f"  num_eval_loader_workers: {config['num_eval_loader_workers']}")