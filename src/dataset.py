import os
import re
import csv
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm
import logging

# Configuration
ACCEPTED_CHARS = (
    'абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя '
    '.,!?-:'
)
PUNCT = '.,!?-:'
TARGET_SR = 22050


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\u00A0\u202F\n]+', ' ', text)
    text = ''.join(ch for ch in text if ch in ACCEPTED_CHARS)
    text = text.strip().strip(PUNCT)
    text = re.sub(r'\s+', ' ', text)
    return text


def resample_and_trim(audio, orig_sr, target_sr, trim=True):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    return audio


def process_split(split_name, dataset, out_dir, wavs_dir, trim=True):
    meta_path = os.path.join(out_dir, f"{split_name}.csv")
    with open(meta_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            text = normalize_text(item["sentence"])
            if not text:
                continue

            audio = np.array(item["audio"]["array"])
            sr = item["audio"]["sampling_rate"]
            audio = resample_and_trim(audio, sr, TARGET_SR, trim=trim)

            uid = f"{split_name}_{i:05d}"
            wav_path = os.path.join(wavs_dir, f"{uid}.wav")
            sf.write(wav_path, audio, TARGET_SR, subtype='PCM_16')
            speaker = item["gender"] if item.get("gender") else "unknown"
            writer.writerow([uid,speaker,text])


def prepare_dataset(data_set_path):
    os.makedirs(data_set_path, exist_ok=True)
    wavs_dir = os.path.join(data_set_path, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    train_ds = load_dataset("btsee/common_voice_21_mn_cyrillic", split="train")
    test_ds = load_dataset("btsee/common_voice_21_mn_cyrillic", split="test")

    process_split("train", train_ds, data_set_path, wavs_dir)
    process_split("test", test_ds, data_set_path, wavs_dir)
    
    logging.info(f"Processed {len(train_ds)} training samples and {len(test_ds)} test samples.")