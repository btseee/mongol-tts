import random
from typing import Any, Dict, List, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler

__all__ = ['Text2MelDataLoader', 'SSRNDataLoader']


class Text2MelDataLoader(DataLoader):
    def __init__(self, text2mel_dataset: Dataset, batch_size: int, mode: str = 'train', num_workers: int = 8) -> None:
        """
        DataLoader for text-to-mel datasets.
        Depending on the mode, slices the dataset to provide training or validation data.
        """
        if mode == 'train':
            text2mel_dataset.slice(0, -batch_size)
        elif mode == 'valid':
            text2mel_dataset.slice(len(text2mel_dataset) - batch_size, None)
        else:
            raise ValueError("mode must be either 'train' or 'valid'")
        super().__init__(text2mel_dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         shuffle=True)


class SSRNDataLoader(DataLoader):
    def __init__(self, ssrn_dataset: Dataset, batch_size: int, mode: str = 'train', num_workers: int = 8) -> None:
        """
        DataLoader for SSRN datasets.
        Uses a specialized sampler for training mode.
        """
        if mode == 'train':
            ssrn_dataset.slice(0, -batch_size)
            sampler = PartiallyRandomizedSimilarTimeLengthSampler(
                lengths=ssrn_dataset.text_lengths,
                data_source=ssrn_dataset,
                batch_size=batch_size
            )
            super().__init__(ssrn_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             sampler=sampler)
        elif mode == 'valid':
            ssrn_dataset.slice(len(ssrn_dataset) - batch_size, None)
            super().__init__(ssrn_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             shuffle=True)
        else:
            raise ValueError("mode must be either 'train' or 'valid'")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that pads each tensor in the batch to the maximum length along its first dimension.
    """
    keys = batch[0].keys()
    max_lengths = {key: 0 for key in keys}
    collated_batch = {key: [] for key in keys}

    # Determine max length per key
    for row in batch:
        for key in keys:
            max_lengths[key] = max(max_lengths[key], row[key].shape[0])

    # Pad each array to the max length
    for row in batch:
        for key in keys:
            array = row[key]
            dim = len(array.shape)
            if dim == 1:
                padded_array = np.pad(array, (0, max_lengths[key] - array.shape[0]), mode='constant')
            elif dim == 2:
                padded_array = np.pad(array, ((0, max_lengths[key] - array.shape[0]), (0, 0)), mode='constant')
            else:
                raise ValueError("Array must be 1D or 2D")
            collated_batch[key].append(padded_array)

    # Convert lists to tensors using default_collate
    for key in keys:
        collated_batch[key] = default_collate(collated_batch[key])
    return collated_batch


class PartiallyRandomizedSimilarTimeLengthSampler(Sampler):
    """
    Partially randomized sampler.
    
    1. Sort indices by lengths.
    2. Within groups, randomize order.
    3. Permute mini-batches if desired.
    
    This approach helps grouping similar length sequences together while adding randomness.
    """
    def __init__(self, lengths: List[int], data_source: Dataset, batch_size: int = 16,
                 batch_group_size: int = None, permutate: bool = True) -> None:
        super().__init__(data_source)
        self.lengths, self.sorted_indices = torch.sort(torch.tensor(lengths, dtype=torch.long))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            batch_group_size -= batch_group_size % batch_size
        self.batch_group_size = batch_group_size
        if self.batch_group_size % self.batch_size != 0:
            raise ValueError("batch_group_size must be a multiple of batch_size")
        self.permutate = permutate

    def __iter__(self) -> Iterator[int]:
        indices = self.sorted_indices.clone()
        total_len = len(indices)
        batch_group_size = self.batch_group_size
        # Shuffle within each group block using PyTorch permutation
        for i in range(total_len // batch_group_size):
            start = i * batch_group_size
            end = start + batch_group_size
            perm = torch.randperm(batch_group_size)
            indices[start:end] = indices[start:end][perm]
        
        # Permutate mini-batches if enabled
        valid_group_len = (total_len // batch_group_size) * batch_group_size
        if self.permutate and valid_group_len > 0:
            num_batches = valid_group_len // self.batch_size
            indices_view = indices[:valid_group_len].view(num_batches, self.batch_size)
            perm = torch.randperm(num_batches)
            indices[:valid_group_len] = indices_view[perm].reshape(-1)
        
        # Shuffle the remaining indices
        if valid_group_len < total_len:
            remaining = indices[valid_group_len:]
            perm = torch.randperm(len(remaining))
            indices[valid_group_len:] = remaining[perm]
        
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.sorted_indices)
