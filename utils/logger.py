"""
Logger utilities for TensorBoard and comet.ml integration.

Author: Erdene-Ochir Tuguldur
"""

import os
from typing import Dict, Any

from tensorboardX import SummaryWriter
from models.hparams import HParams as hp


class Logger:
    def __init__(self, dataset_name: str, model_name: str) -> None:
        """
        Initialize the logger.
        
        Args:
            dataset_name: Name of the dataset.
            model_name: Name of the model.
        """
        self.model_name = model_name
        self.project_name = f"{dataset_name}-{self.model_name}"
        self.logdir = os.path.join(hp.logdir, self.project_name)
        self.writer = SummaryWriter(log_dir=self.logdir)

    def log_step(self, phase: str, step: int, loss_dict: Dict[str, float], image_dict: Dict[str, Any]) -> None:
        """
        Log step-level scalar values and images.
        
        Args:
            phase: Current phase ('train' or 'valid').
            step: The global step count.
            loss_dict: Dictionary of loss values.
            image_dict: Dictionary of images (as numpy arrays or torch tensors).
        """
        if phase == 'train':
            if step % 50 == 0:
                for key in sorted(loss_dict):
                    self.writer.add_scalar(f'{phase}-step/{key}', loss_dict[key], step)
            if step % 1000 == 0:
                for key in sorted(image_dict):
                    self.writer.add_image(f'{self.model_name}/{key}', image_dict[key], step)

    def log_epoch(self, phase: str, step: int, loss_dict: Dict[str, float]) -> None:
        """
        Log epoch-level scalar values.
        
        Args:
            phase: Phase name.
            step: Epoch number or global step.
            loss_dict: Dictionary of loss values.
        """
        for key in sorted(loss_dict):
            self.writer.add_scalar(f'{phase}/{key}', loss_dict[key], step)
