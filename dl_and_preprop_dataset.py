#!/usr/bin/env python
"""
Download and preprocess datasets. Supported datasets are:
  * English female: LJSpeech (https://keithito.com/LJ-Speech-Dataset/)
  * Mongolian male: MBSpeech (Mongolian Bible)
  
Author: Erdene-Ochir Tuguldur
"""

import os
import sys
import csv
import time
import argparse
import fnmatch
import tarfile
import librosa
import pandas as pd
from zipfile import ZipFile
from typing import Optional

from models.hparams import HParams as hp
from utils.audio import preprocess
from utils.utils import download_file
from datasets.mb_speech import MBSpeech
from datasets.lj_speech import LJSpeech
from tqdm import tqdm
from utils.audio import get_spectrograms
import numpy as np

def extract_tar_bz2(archive_path: str, extract_path: str) -> None:
    """Extract a .tar.bz2 archive using tarfile."""
    print(f"Extracting '{archive_path}'...")
    with tarfile.open(archive_path, mode='r:bz2') as tar:
        tar.extractall(path=extract_path)

def extract_tar_gz(archive_path: str, extract_path: str) -> None:
    """Extract a .tar.gz archive using tarfile."""
    print(f"Extracting '{archive_path}'...")
    with tarfile.open(archive_path, mode='r:gz') as tar:
        tar.extractall(path=extract_path)

def download_and_extract_zip(url: str, zip_path: str, extract_path: str) -> None:
    """Download a zip file if not present and extract it."""
    if not os.path.isfile(zip_path):
        download_file(url, zip_path)
    else:
        print(f"'{os.path.basename(zip_path)}' already exists")
    print(f"Extracting '{os.path.basename(zip_path)}'...")
    with ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(path=extract_path)

def preprocess_commonvoice(dataset_path: str) -> None:
    """
    Preprocess the Common Voice dataset.
    
    Expects:
      - dataset_path contains:
           validated.tsv
           clips/          (audio files)
      
    Creates in dataset_path:
      - mels/ and mags/ directories with precomputed spectrograms.
      - metadata.csv file with format: [file_id | transcript | transcript]
    """
    clips_path = os.path.join(dataset_path, "clips")
    mels_path = os.path.join(dataset_path, "mels")
    mags_path = os.path.join(dataset_path, "mags")
    metadata_csv_path = os.path.join(dataset_path, "metadata.csv")
    
    os.makedirs(mels_path, exist_ok=True)
    os.makedirs(mags_path, exist_ok=True)
    
    # Read metadata from validated.tsv
    validated_tsv = os.path.join(dataset_path, "validated.tsv")
    if not os.path.isfile(validated_tsv):
        print(f"Error: {validated_tsv} not found.")
        sys.exit(1)
        
    # Open metadata CSV to write processed entries
    metadata_file = open(metadata_csv_path, "w", newline="", encoding="utf-8")
    metadata_writer = csv.writer(metadata_file, delimiter="|")
    
    # Optionally, you can define a normalization function for the transcript.
    def normalize_text(text: str) -> str:
        text = text.strip().lower()
        for c in "-—:":
            text = text.replace(c, "-")
        for c in "()\"«»“”'":
            text = text.replace(c, ",")
        return text

    total_duration = 0.0
    processed_count = 0

    # Read TSV metadata; TSV is assumed to have headers "path" and "sentence"
    with open(validated_tsv, newline="", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in tqdm(reader, desc="Preprocessing CommonVoice"):
            rel_audio_path = row["path"]
            transcript = normalize_text(row["sentence"]) + "E"  # Append EOS token

            # Construct full audio file path (inside the clips folder)
            audio_file = os.path.join(clips_path, rel_audio_path)
            if not os.path.isfile(audio_file):
                print(f"Warning: audio file {audio_file} not found; skipping.")
                continue

            # Process audio to generate spectrograms
            try:
                mel, mag = get_spectrograms(audio_file)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

            # Get time frames and apply padding for reduction (as in your original preprocess)
            t_frames = mel.shape[0]
            num_paddings = (hp.reduction_rate - (t_frames % hp.reduction_rate)) % hp.reduction_rate
            mel = np.pad(mel, ((0, num_paddings), (0, 0)), mode="constant")
            mag = np.pad(mag, ((0, num_paddings), (0, 0)), mode="constant")
            # Apply reduction (downsample along time axis) for mel spectrogram only
            mel_reduced = mel[::hp.reduction_rate, :]

            # Generate a unique file ID (e.g., based on processed count)
            file_id = f"CV{processed_count:05d}"
            np.save(os.path.join(mels_path, f"{file_id}.npy"), mel_reduced)
            np.save(os.path.join(mags_path, f"{file_id}.npy"), mag)
            
            # Optionally, accumulate duration (using length of audio and sample rate)
            y, sr = librosa.load(audio_file, sr=hp.sr)
            total_duration += len(y) / sr

            # Write metadata entry: file_id, transcript, transcript (format similar to LJSpeech)
            metadata_writer.writerow([file_id, transcript, transcript])
            processed_count += 1

    metadata_file.close()
    print(f"Preprocessing complete: {processed_count} files processed.")
    print(f"Total audio duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}")
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'commonvoice'], help='dataset name')
    args = parser.parse_args()

    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    
    if args.dataset == 'ljspeech':
        dataset_file_name = 'LJSpeech-1.1.tar.bz2'
        dataset_path = os.path.join(datasets_path, 'LJSpeech-1.1')
        archive_path = os.path.join(datasets_path, dataset_file_name)

        if not os.path.isdir(dataset_path):
            if not os.path.isfile(archive_path):
                url = f"http://data.keithito.com/data/speech/{dataset_file_name}"
                download_file(url, archive_path)
            else:
                print(f"'{dataset_file_name}' already exists")
            extract_tar_bz2(archive_path, datasets_path)
        else:
            print("LJSpeech dataset folder already exists")

        print("Preprocessing LJSpeech...")
        lj_speech = LJSpeech([])  # dataset keys can be set inside the preprocess function
        preprocess(dataset_path, lj_speech)

    elif args.dataset == 'mbspeech':
        dataset_name = 'MBSpeech-1.0'
        dataset_path = os.path.join(datasets_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        # Download and extract Bible books
        bible_books = ['01_Genesis', '02_Exodus', '03_Leviticus']
        for book in bible_books:
            zip_name = f"{book}.zip"
            zip_path = os.path.join(datasets_path, zip_name)
            url = f"https://s3.us-east-2.amazonaws.com/bible.davarpartners.com/Mongolian/{zip_name}"
            download_and_extract_zip(url, zip_path, datasets_path)

        # Download and extract CSV metadata
        csv_zip_name = f"{dataset_name}-csv.zip"
        csv_zip_path = os.path.join(datasets_path, csv_zip_name)
        download_url = f"https://www.dropbox.com/s/dafueq0w278lbz6/{csv_zip_name}?dl=1"
        download_and_extract_zip(download_url, csv_zip_path, datasets_path)
        dataset_csv_extracted_path = os.path.join(datasets_path, f"{dataset_name}-csv")

        # Prepare destination folders
        wavs_path = os.path.join(dataset_path, 'wavs')
        os.makedirs(wavs_path, exist_ok=True)
        metadata_csv_path = os.path.join(dataset_path, 'metadata.csv')

        total_duration_s = 0
        with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as metadata_csv:
            metadata_writer = csv.writer(metadata_csv, delimiter='|')

            def _normalize(s: str) -> str:
                s = s.strip()
                if s and (s[0] in ('—', '-')):
                    s = s[1:].strip()
                return s

            def _get_mp3_file(book_name: str, chapter: int) -> Optional[str]:
                book_download_path = os.path.join(datasets_path, book_name)
                wildcard = f"*{chapter:02d} - DPI.mp3"
                for file_name in os.listdir(book_download_path):
                    if fnmatch.fnmatch(file_name, wildcard):
                        return os.path.join(book_download_path, file_name)
                return None

            def _convert_mp3_to_wav(book_name: str, book_nr: int) -> None:
                nonlocal total_duration_s
                chapter = 1
                while True:
                    try:
                        chapter_csv = os.path.join(dataset_csv_extracted_path, f"{book_name}_{chapter:02d}.csv")
                        df = pd.read_csv(chapter_csv, sep="|")
                        print(f"Processing {chapter_csv}...")
                        mp3_file = _get_mp3_file(book_name, chapter)
                        if mp3_file is None:
                            raise FileNotFoundError(f"No mp3 file found for chapter {chapter}")
                        print(f"Processing {mp3_file}...")
                        samples, sr = librosa.load(mp3_file, sr=44100, mono=True)
                        assert sr == 44100, "Unexpected sample rate"

                        i = 0
                        for _, row in df.iterrows():
                            start, end, sentence = int(row['start']), int(row['end']), _normalize(row['sentence'])
                            if end <= start:
                                continue
                            duration = end - start
                            duration_s = int(duration / 44100)
                            if duration_s > 10:
                                continue  # only audios shorter than 10s

                            total_duration_s += duration_s
                            i += 1
                            fn = f"MB{book_nr}{chapter:02d}-{i:04d}"
                            metadata_writer.writerow([fn, sentence, sentence])
                            wav = samples[start:end]
                            # Resample to the target sample rate
                            wav_resampled = librosa.resample(wav, orig_sr=44100, target_sr=hp.sr)
                            librosa.output.write_wav(os.path.join(wavs_path, fn + ".wav"), wav_resampled, hp.sr)
                        chapter += 1
                    except FileNotFoundError:
                        break

            _convert_mp3_to_wav('01_Genesis', 1)
            _convert_mp3_to_wav('02_Exodus', 2)
            _convert_mp3_to_wav('03_Leviticus', 3)
        print(f"Total audio duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration_s))}")

        print("Preprocessing MBSpeech...")
        mb_speech = MBSpeech([])
        preprocess(dataset_path, mb_speech)

    elif args.dataset == 'commonvoice':
        base_dir = os.path.dirname(os.path.realpath(__file__))
        datasets_path = os.path.join(base_dir, "datasets")
        os.makedirs(datasets_path, exist_ok=True)

        # Name of the archive file (update the file name and URL as needed)
        dataset_archive = "cv-corpus-21.0-2025-03-14-mn.tar.gz"
        dataset_archive_path = os.path.join(datasets_path, dataset_archive)
        dataset_folder = "cv-corpus-21.0-2025-03-14/mn"
        dataset_path = os.path.join(datasets_path, dataset_folder)

        # Download if the archive does not exist
        if not os.path.isfile(dataset_archive_path):
            download_url = "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-21.0-2025-03-14/cv-corpus-21.0-2025-03-14-mn.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250402%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250402T121214Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=45646f408b97be9bf8ffdc94cdee9b04865aacaf985bd2770a3953208cb4fbe27bb5bc4b13bdba941bc59760ffeddc68191a9de2bcf01546ef8a71d29da2679b237fa6ddaa93407573169e10ae015ce39bc84a2f37cb894d60d72c4f6c018e85027f32904cdbe6865c776c9ed39c23dbde416a1bc898ca69498aa4c974400f342e246f032a7dade9684fe2811e3549d204302948c7c171a0d909c00332adbe5d3cae0c967d8335138529f117268acce6fce36ced80cf3663de60a6ae43f6d6233fbe20fd4b37164a5f332f5250fe4a84e214dd17fa26f23537959a2ea58c434accbddd2e816293d133a7df260454492820a2c16e8ddafc9aa17d5f362d9c02e7"  # <-- Replace with the real URL!
            print(f"Downloading Common Voice Mongolian dataset from {download_url}...")
            download_file(download_url, dataset_archive_path)
        else:
            print(f"Archive '{dataset_archive}' already exists.")

        # Extract if dataset folder does not exist
        if not os.path.isdir(dataset_path):
            extract_tar_gz(dataset_archive_path, datasets_path)
        else:
            print(f"Dataset folder '{dataset_folder}' already exists.")

        # Preprocess the dataset
        print("Preprocessing Common Voice dataset...")
        preprocess_commonvoice(dataset_path)
        print("All done.")
if __name__ == "__main__":
    main()
