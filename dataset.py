import os
import csv
import soundfile as sf
from datasets import load_dataset

DATASET_NAME = "btsee/mbspeech_mn"
OUTPUT_DIR = "dataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
os.makedirs(WAV_DIR, exist_ok=True)

ds = load_dataset(DATASET_NAME, split="train")

with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="|")
    for idx, item in enumerate(ds):
        array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        text_orig = item["sentence_orig"]
        text_norm= item["sentence_norm"]
        fname = f"{idx:05d}"
        path = os.path.join(WAV_DIR, f"{fname}.wav")
        sf.write(path, array, sr, subtype="PCM_16")
        writer.writerow([fname, text_orig, text_norm])
