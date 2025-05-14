#!/usr/bin/env python3
import os
import csv
import re
import argparse
from tqdm import tqdm
import soundfile as sf
import numpy as np
import librosa
from utils.num2words.num2words import num2words

# --- CONFIGURABLE CHARSETS & NORMALIZATION MAPS ---
_MONGOLIAN_CYRILLIC_CHARS = (
    'абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя'
    'АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ'
)
_ACCEPTED_CHARS = _MONGOLIAN_CYRILLIC_CHARS + ' .,!?-:'
_PUNCT = '.,!?-:'
_REPL = {
    '–': '-', '—': '-', '‘': "'", '’': "'",
    '“': '"', '”': '"', '„': '"',
    '\u00A0': ' ', '\u202F': ' ',
    '\n': ' '
}
_LATIN_TO_MN = {ch: rd for ch, rd in zip(
    list("abcdefghijklmnopqrstuvwxyz"),
    ["эй","би","си","ди","и","эф","жи","эйч","ай",
     "жей","кей","эл","эм","эн","о","пи","кью","ар",
     "эс","ти","ю","ви","дабл-ю","икс","вай","зед"]
)}

# --- TEXT NORMALIZATION FUNCTIONS ---
def expand_numbers_and_letters(text: str) -> str:
    text = re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='mn'), text)
    text = re.sub(r'[A-Za-z]', lambda m: _LATIN_TO_MN.get(m.group().lower(), m.group()), text)
    return text

def normalize_text(text: str) -> str:
    text = text.lower()
    for old, new in _REPL.items():
        text = text.replace(old, new)
    text = expand_numbers_and_letters(text)
    text = ''.join(ch for ch in text if ch in _ACCEPTED_CHARS)
    text = text.strip().strip(_PUNCT)
    text = re.sub(r'\s+', ' ', text)
    return text

# --- AUDIO PROCESSING FUNCTIONS ---
def load_and_resample(path: str, sr: int) -> np.ndarray:
    wav, orig_sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(y=wav, orig_sr=orig_sr, target_sr=sr)
    return wav

def trim_silence(wav: np.ndarray, sr: int,
                 top_db: float, frame_length: int, hop_length: int) -> np.ndarray:
    wav_trimmed, _ = librosa.effects.trim(
        wav,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return wav_trimmed

def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    wav_int16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    sf.write(path, wav_int16, sr, subtype='PCM_16')

# --- MAIN METADATA PREPARATION ---
def prepare_metadata(args):
    tsv_path = os.path.join(args.cv_data_dir, args.tsv_file)
    clips_dir = os.path.join(args.cv_data_dir, "clips")
    wavs_out = os.path.join(args.output_dir, "wavs")
    os.makedirs(wavs_out, exist_ok=True)

    meta_lines = []
    skipped = 0
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(reader, desc="Clips"):
            txt = row.get('sentence', '').strip()
            if not txt:
                skipped += 1
                continue

            in_path = os.path.join(clips_dir, row['path'])
            if not os.path.isfile(in_path):
                skipped += 1
                continue

            norm = normalize_text(txt)
            if not norm:
                skipped += 1
                continue

            # load, resample, trim
            wav = load_and_resample(in_path, args.target_sr)
            if args.trim_silence:
                wav = trim_silence(
                    wav,
                    args.target_sr,
                    args.trim_threshold_in_db,
                    args.trim_frame_size,
                    args.trim_hop_size
                )

            # write out
            clip_id = os.path.splitext(row['path'])[0]
            speaker = row.get('client_id', 'spk0')
            out_path = os.path.join(wavs_out, f"{clip_id}.wav")
            write_wav(out_path, wav, args.target_sr)

            meta_lines.append(f"{clip_id}|{speaker}|{norm}")

    meta_path = os.path.join(args.output_dir, args.metadata_filename)
    with open(meta_path, 'w', encoding='utf-8') as f:
        for L in meta_lines:
            f.write(L + "\n")

    print(f"Processed: {len(meta_lines)}, skipped: {skipped}")
    print(f"Metadata: {meta_path}")
    print(f"WAVs in: {wavs_out}")

# --- CMDLINE INTERFACE ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("cv_data_dir", help="CommonVoice root dir")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("--tsv_file", default="validated.tsv",
                   help="CommonVoice TSV file name (e.g. validated.tsv)")
    p.add_argument("--metadata_filename", default="metadata.csv")
    p.add_argument("--target_sr", type=int, default=22050,
                   help="Audio sampling rate")
    p.add_argument("--trim_silence", action="store_true",
                   help="Trim leading/trailing silence")
    p.add_argument("--trim_threshold_in_db", type=float, default=30.0)
    p.add_argument("--trim_frame_size", type=int, default=2048)
    p.add_argument("--trim_hop_size", type=int, default=512)
    args = p.parse_args()

    prepare_metadata(args)