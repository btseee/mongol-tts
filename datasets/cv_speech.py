import os
import re
import codecs
import unicodedata
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

vocab = "PE абвгдеёжзийклмноөпрстуүфхцчшъыьэюя-.,!?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def text_normalize(text):
    text = text.lower()
    for c in "-—:":
        text = text.replace(c, "-")
    for c in "()\"«»“”'":
        text = text.replace(c, ",")

    # Remove characters not in vocab, except space
    allowed_chars = set(vocab[2:]) # Exclude P and E
    allowed_chars.add(' ')
    text = ''.join(char for char in text if char in allowed_chars)
    text = re.sub("[ ]+", " ", text).strip() # Remove multiple spaces and trim
    return text


def read_metadata(metadata_file):
    fnames, text_lengths, texts = [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        # Assuming metadata.csv format: fname|normalized_text|normalized_text
        parts = line.strip().split("|")
        if len(parts) < 3:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        fname, _, text = parts[0], parts[1], parts[2]

        fnames.append(fname)

        # Text should already be normalized during preprocessing, just add EOS
        text = text + "E"  # E: EOS
        try:
            text_indices = [char2idx[char] for char in text]
        except KeyError as e:
            print(f"Warning: Character '{e}' not in vocab for text: '{text[:-1]}' in file {fname}. Skipping this file.")
            fnames.pop() # Remove the filename if text contains unknown chars
            continue

        text_lengths.append(len(text_indices))
        texts.append(np.array(text_indices, np.longlong))

    return fnames, text_lengths, texts


def get_test_data(sentences, max_n):
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.longlong)
    for i, sent in enumerate(normalized_sentences):
        # Ensure all characters are in vocab before converting
        valid_sent = "".join([char for char in sent if char in char2idx])
        if len(valid_sent) != len(sent):
             print(f"Warning: Some characters removed from sentence during test data prep: {sent}")
        texts[i, :len(valid_sent)] = [char2idx[char] for char in valid_sent]
    return texts


class CVSpeech(Dataset):
    # Expects dir_name like 'cv-corpus-21.0-2024-04-01-mn'
    def __init__(self, keys, dir_name='cv-corpus-21.0-2025-03-14-mn'):
        self.keys = keys
        # Navigate up one level from __file__ (datasets/) then into dir_name
        self.path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets', dir_name)
        self.fnames, self.text_lengths, self.texts = read_metadata(os.path.join(self.path, 'metadata.csv'))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        fname = self.fnames[index]
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            mel_path = os.path.join(self.path, 'mels', f"{fname}.npy")
            if not os.path.exists(mel_path):
                 raise FileNotFoundError(f"Mel file not found: {mel_path}")
            data['mels'] = np.load(mel_path)
        if 'mags' in self.keys:
            mag_path = os.path.join(self.path, 'mags', f"{fname}.npy")
            if not os.path.exists(mag_path):
                raise FileNotFoundError(f"Mag file not found: {mag_path}")
            data['mags'] = np.load(mag_path)
        # Gates are generated based on the loaded mel/mag shape
        if 'mel_gates' in self.keys and 'mels' in data:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int64)
        if 'mag_gates' in self.keys and 'mags' in data:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int64)

        # Ensure all requested keys were processed or generated
        for key in self.keys:
            if key not in data:
                 # This might happen if e.g. 'mels' was requested but the file was missing,
                 # or if 'mel_gates' was requested but 'mels' wasn't.
                 print(f"Warning: Key '{key}' could not be loaded/generated for index {index}, fname {fname}")
                 # Depending on strictness, you might want to return None or raise an error
                 # For now, let's return an empty dict to signal failure upstream
                 # return {} # Or handle appropriately
        return data

# Simple number to word functions (copied from mb_speech.py, potentially useful for normalization if needed later)
def number2word(number):
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

    digitgroup = [number[0 if i - 3 < 0 else i - 3:i] for i in reversed(range(len(number), 0, -3))]
    count = len(digitgroup)
    i = 0
    result = ''
    while i < count - 1:
        result += ' ' + (_3_digits_to_str(digitgroup[i], False) + ' ' + digit_name[count - i])
        i += 1
    return result.strip() + ' ' + _3_digits_to_str(digitgroup[-1])

def _1_digit_2_str(digit):
    return {'0': '', '1': 'нэгэн', '2': 'хоёр', '3': 'гурван', '4': 'дөрвөн', '5': 'таван', '6': 'зургаан',
            '7': 'долоон', '8': 'найман', '9': 'есөн'}[digit]

def _last_digit_2_str(digit):
    return {'0': 'тэг', '1': 'нэг', '2': 'хоёр', '3': 'гурав', '4': 'дөрөв', '5': 'тав', '6': 'зургаа', '7': 'долоо',
            '8': 'найм', '9': 'ес'}[digit]

def _2_digits_2_str(digit, is_fina=True):
    word2 = {'0': '', '1': 'арван', '2': 'хорин', '3': 'гучин', '4': 'дөчин', '5': 'тавин', '6': 'жаран', '7': 'далан',
             '8': 'наян', '9': 'ерэн'}
    word2fina = {'10': 'арав', '20': 'хорь', '30': 'гуч', '40': 'дөч', '50': 'тавь', '60': 'жар', '70': 'дал',
                 '80': 'ная', '90': 'ер'}
    if digit[1] == '0':
        return word2fina[digit] if is_fina else word2[digit[0]]
    digit1 = _last_digit_2_str(digit[1]) if is_fina else _1_digit_2_str(digit[1])
    return (word2[digit[0]] + ' ' + digit1).strip()

def _3_digits_to_str(digit, is_fina=True):
    digstr = digit.lstrip('0')
    if len(digstr) == 0:
        return ''
    if len(digstr) == 1:
        return _1_digit_2_str(digstr) # Use _1_digit, not _last_digit
    if len(digstr) == 2:
        return _2_digits_2_str(digstr, is_fina)
    if digit[-2:] == '00':
        return _1_digit_2_str(digit[0]) + (' зуу' if is_fina else ' зуун')
    else:
        return _1_digit_2_str(digit[0]) + ' зуун ' + _2_digits_2_str(digit[-2:], is_fina)