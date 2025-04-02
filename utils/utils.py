"""
General utility functions.

Author: Erdene-Ochir Tuguldur
"""

import os
import sys
import glob
import math
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from skimage.io import imsave
from skimage import img_as_ubyte
import requests


def get_last_checkpoint_file_name(logdir: str) -> Optional[str]:
    """
    Retrieve the most recent checkpoint file from a directory.
    
    Args:
        logdir: Directory path containing checkpoint files.
    
    Returns:
        The filename of the latest checkpoint, or None if no checkpoints found.
    """
    checkpoints = glob.glob(os.path.join(logdir, '*.pth'))
    checkpoints.sort()
    if not checkpoints:
        return None
    return checkpoints[-1]


def load_checkpoint(checkpoint_file_name: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, int]:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        checkpoint_file_name: Path to the checkpoint file.
        model: Model to load the state into.
        optimizer: Optimizer to load the state into.
    
    Returns:
        A tuple (start_epoch, global_step).
    """
    checkpoint = torch.load(checkpoint_file_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    del checkpoint
    print(f"Loaded checkpoint: epoch={start_epoch} step={global_step}")
    return start_epoch, global_step


def save_checkpoint(logdir: str, epoch: int, global_step: int, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> None:
    """
    Save the current training state to a checkpoint file.
    
    Args:
        logdir: Directory to save the checkpoint.
        epoch: Current epoch number.
        global_step: Current global step.
        model: Model whose state to save.
        optimizer: Optimizer whose state to save.
    """
    checkpoint_file_name = os.path.join(logdir, f'step-{global_step // 1000:03d}K.pth')
    print(f"Saving checkpoint file '{checkpoint_file_name}'...")
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file_name)
    del checkpoint


def download_file(url: str, file_path: str) -> None:
    """
    Download a file from a URL and save it locally.
    
    Args:
        url: The URL to download the file from.
        file_path: Local file path to save the downloaded file.
    """
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    wrote = 0
    with open(file_path, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size / block_size), unit='MB'):
            wrote += len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print("Downloading failed")
        sys.exit(1)


def save_to_png(file_name: str, array) -> None:
    """
    Save a numpy array as a PNG image.
    
    Args:
        file_name: Path to save the PNG file.
        array: Numpy array representing the image.
    """
    imsave(file_name, img_as_ubyte(array))
