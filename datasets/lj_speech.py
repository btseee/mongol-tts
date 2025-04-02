"""
Data loader for the LJSpeech dataset.
See: https://keithito.com/LJ-Speech-Dataset/
"""
import os
import re
import codecs
import unicodedata
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

# Define vocabulary and mapping dictionaries
vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text: str) -> str:
    """Normalize text: strip accents, lowercase, remove unwanted characters, and collapse spaces."""
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub(f"[^{re.escape(vocab)}]", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    return text


def read_metadata(metadata_path: str) -> Tuple[List[str], List[int], List[np.ndarray]]:
    """Read and process metadata from a CSV file."""
    fnames, text_lengths, texts = [], [], []
    with codecs.open(metadata_path, 'r', 'utf-8') as f:
        lines = f.readlines()
    for line in lines:
        fname, _, text = line.strip().split("|")
        fnames.append(fname)
        text = text_normalize(text).strip() + "E"  # Append EOS
        text_ids = [char2idx[char] for char in text]
        text_lengths.append(len(text_ids))
        texts.append(np.array(text_ids, np.int64))
    return fnames, text_lengths, texts


def get_test_data(sentences: List[str], max_n: int) -> np.ndarray:
    """Convert a list of sentences into a padded numpy array of indices."""
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.int64)
    for i, sent in enumerate(normalized_sentences):
        text_ids = [char2idx[char] for char in sent]
        texts[i, :len(text_ids)] = text_ids
    return texts


class LJSpeech(Dataset):
    def __init__(self, keys: List[str], dir_name: str = 'LJSpeech-1.1') -> None:
        """
        Dataset for LJSpeech.
        
        :param keys: List of data keys to load (e.g., 'texts', 'mels', 'mags').
        :param dir_name: Name of the dataset directory.
        """
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        metadata_file = os.path.join(self.path, 'metadata.csv')
        self.fnames, self.text_lengths, self.texts = read_metadata(metadata_file)

    def slice(self, start: int, end: int) -> None:
        """Slice the dataset to only include items in [start:end]."""
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> dict:
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            mel_path = os.path.join(self.path, 'mels', f"{self.fnames[index]}.npy")
            data['mels'] = np.load(mel_path)
        if 'mags' in self.keys:
            mag_path = os.path.join(self.path, 'mags', f"{self.fnames[index]}.npy")
            data['mags'] = np.load(mag_path)
        if 'mel_gates' in self.keys and 'mels' in data:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int64)
        if 'mag_gates' in self.keys and 'mags' in data:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int64)
        return data
