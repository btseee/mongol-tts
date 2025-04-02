"""
Data loader for the Mozilla Common Voice dataset for Mongolian.

Assumes:
  - A metadata file "validated.tsv" with columns: "path" and "sentence".
  - Audio clips are stored in a subfolder "clips".
  
Author: Battseren Badral
"""

import os
import csv
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
import librosa

from models.hparams import HParams as hp

# Define vocabulary and mapping dictionaries for Mongolian (adjust as needed)
vocab = "PE абвгдеёжзийклмноөпрстуүфхцчшъыьэюя-.,!?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text: str) -> str:
    """Normalize Mongolian text."""
    text = text.lower()
    for c in "-—:":
        text = text.replace(c, "-")
    for c in "()\"«»“”'":
        text = text.replace(c, ",")
    return text


def read_metadata(metadata_path: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Read metadata from a TSV file.

    Returns:
        A tuple of:
         - A list of audio file paths (relative to the "clips" folder).
         - A list of corresponding transcripts as numpy arrays of text indices.
    """
    audio_files = []
    texts = []
    with open(metadata_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            audio_files.append(row["path"])
            sentence = text_normalize(row["sentence"]).strip() + "E"  # Append EOS token
            text_ids = [char2idx.get(char, 0) for char in sentence]
            texts.append(np.array(text_ids, np.int64))
    return audio_files, texts


def get_test_data(sentences: List[str], max_n: int) -> np.ndarray:
    """
    Convert a list of sentences into a padded numpy array of indices.
    
    Args:
        sentences: List of input sentences.
        max_n: Maximum sequence length for padding.
    Returns:
        A numpy array of shape (num_sentences, max_n+1).
    """
    normalized = [text_normalize(s).strip() + "E" for s in sentences]
    texts_array = np.zeros((len(normalized), max_n + 1), np.int64)
    for i, s in enumerate(normalized):
        text_ids = [char2idx.get(char, 0) for char in s]
        texts_array[i, :len(text_ids)] = text_ids
    return texts_array


class CommonVoiceMongolian(Dataset):
    def __init__(self, keys: List[str], dir_name: str = "CommonVoice-Mongolian", sample_rate: int = 16000) -> None:
        """
        Initialize the Common Voice Mongolian dataset.
        
        Args:
            keys: List of keys to load (e.g., ['audio', 'texts']).
            dir_name: Folder name where the dataset is stored.
            sample_rate: Audio sampling rate.
        """
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        metadata_file = os.path.join(self.path, "validated.tsv")
        self.audio_files, self.texts = read_metadata(metadata_file)
        self.sample_rate = sample_rate

    def slice(self, start: int, end: int) -> None:
        """Slice the dataset between the given indices."""
        self.audio_files = self.audio_files[start:end]
        self.texts = self.texts[start:end]

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> dict:
        data = {}
        if "texts" in self.keys:
            data["texts"] = self.texts[index]
        if "audio" in self.keys:
            audio_path = os.path.join(self.path, "clips", self.audio_files[index])
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            data["audio"] = audio.astype(np.float32)
        return data
