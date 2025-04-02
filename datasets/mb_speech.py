"""
Data loader for the Mongolian Bible dataset.
"""

import os
import codecs
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

# Define vocabulary and mapping dictionaries for Mongolian Bible
vocab = "PE абвгдеёжзийклмноөпрстуүфхцчшъыьэюя-.,!?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text: str) -> str:
    """Normalize Mongolian text by lowercasing and replacing certain punctuation."""
    text = text.lower()
    for c in "-—:":
        text = text.replace(c, "-")
    for c in "()\"«»“”'":
        text = text.replace(c, ",")
    return text


def read_metadata(metadata_path: str) -> Tuple[List[str], List[int], List[np.ndarray]]:
    """Read and process metadata from a CSV file for Mongolian Bible."""
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
    """Convert a list of sentences into a padded numpy array of indices for Mongolian text."""
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.int64)
    for i, sent in enumerate(normalized_sentences):
        text_ids = [char2idx[char] for char in sent]
        texts[i, :len(text_ids)] = text_ids
    return texts


class MBSpeech(Dataset):
    def __init__(self, keys: List[str], dir_name: str = 'MBSpeech-1.0') -> None:
        """
        Dataset for Mongolian Bible.
        
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


# Utility functions for converting Mongolian numbers to text
def number2word(number: str) -> str:
    """
    Convert a numeric string into its Mongolian textual representation.
    """
    digit_len = len(number)
    digit_name = {1: '', 2: 'мянга', 3: 'сая', 4: 'тэрбум', 5: 'их наяд', 6: 'тунамал'}

    if digit_len == 1:
        return _last_digit_2_str(number)
    if digit_len == 2:
        return _2_digits_2_str(number)
    if digit_len == 3:
        return _3_digits_to_str(number)
    if digit_len < 7:
        return _3_digits_to_str(number[:-3], False) + ' ' + digit_name[2] + ' ' + _3_digits_to_str(number[-3:])
    
    digitgroup = [number[max(0, i - 3):i] for i in range(len(number), 0, -3)][::-1]
    count = len(digitgroup)
    result = []
    for i in range(count - 1):
        result.append(f"{_3_digits_to_str(digitgroup[i], False)} {digit_name[count - i]}")
    result.append(_3_digits_to_str(digitgroup[-1]))
    return ' '.join(result).strip()


def _1_digit_2_str(digit: str) -> str:
    return {'0': '', '1': 'нэгэн', '2': 'хоёр', '3': 'гурван', '4': 'дөрвөн', '5': 'таван', 
            '6': 'зургаан', '7': 'долоон', '8': 'найман', '9': 'есөн'}[digit]


def _last_digit_2_str(digit: str) -> str:
    return {'0': 'тэг', '1': 'нэг', '2': 'хоёр', '3': 'гурав', '4': 'дөрөв', '5': 'тав', 
            '6': 'зургаа', '7': 'долоо', '8': 'найм', '9': 'ес'}[digit]


def _2_digits_2_str(digit: str, is_fina: bool = True) -> str:
    word2 = {'0': '', '1': 'арван', '2': 'хорин', '3': 'гучин', '4': 'дөчин', 
             '5': 'тавин', '6': 'жаран', '7': 'далан', '8': 'наян', '9': 'ерэн'}
    word2fina = {'10': 'арав', '20': 'хорь', '30': 'гуч', '40': 'дөч', 
                 '50': 'тавь', '60': 'жар', '70': 'дал', '80': 'ная', '90': 'ер'}
    if digit[1] == '0':
        return word2fina[digit] if is_fina else word2[digit[0]]
    digit1 = _last_digit_2_str(digit[1]) if is_fina else _1_digit_2_str(digit[1])
    return f"{word2[digit[0]]} {digit1}".strip()


def _3_digits_to_str(digit: str, is_fina: bool = True) -> str:
    digstr = digit.lstrip('0')
    if not digstr:
        return ''
    if len(digstr) == 1:
        return _1_digit_2_str(digstr)
    if len(digstr) == 2:
        return _2_digits_2_str(digstr, is_fina)
    if digit[-2:] == '00':
        return f"{_1_digit_2_str(digit[0])} зуу" if is_fina else f"{_1_digit_2_str(digit[0])} зуун"
    else:
        return f"{_1_digit_2_str(digit[0])} зуун {_2_digits_2_str(digit[-2:], is_fina)}"
