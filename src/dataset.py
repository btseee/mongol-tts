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
PUNCT = '.,!?-:' # Used for stripping from ends of sentences
TARGET_SR = 22050
MIN_AUDIO_DURATION_SEC = 0.5 # Minimum audio duration in seconds after processing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_text(text):
    """Normalizes the input text."""
    text = text.lower()
    text = re.sub(r'[\u00A0\u202F\n\r\t]+', ' ', text)  # Replace various whitespace with single space
    text = ''.join(ch for ch in text if ch in ACCEPTED_CHARS)
    text = text.strip().strip(PUNCT)  # Strip leading/trailing spaces and then specified punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    return text


def resample_and_trim(audio, orig_sr, target_sr, trim=True, top_db=30):
    """Resamples audio to target_sr and optionally trims silence."""
    if audio.dtype != np.float32: # Librosa expects float32
        audio = audio.astype(np.float32)
    if len(audio.shape) > 1: # Convert to mono if stereo
        audio = np.mean(audio, axis=1)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio


def process_split(split_name, dataset, out_dir, wavs_dir, trim=True):
    """Processes a single dataset split (train/test)."""
    meta_path = os.path.join(out_dir, f"{split_name}.csv")
    processed_count = 0
    skipped_count = 0
    
    with open(meta_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i, item in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            text = normalize_text(item["sentence"])
            if not text:
                logging.warning(f"Skipping item {i} in {split_name} due to empty normalized text. Original: '{item['sentence']}'")
                skipped_count += 1
                continue

            try:
                audio_data = item["audio"]["array"]
                sr = item["audio"]["sampling_rate"]
                
                # Ensure audio_data is a numpy array
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)

                audio = resample_and_trim(audio_data, sr, TARGET_SR, trim=trim)

                if len(audio) < TARGET_SR * MIN_AUDIO_DURATION_SEC:
                    logging.warning(f"Skipping item {i} in {split_name} (UID potential: {split_name}_{i:05d}) - audio too short ({len(audio)/TARGET_SR:.2f}s) after processing.")
                    skipped_count += 1
                    continue

                uid = f"{split_name}_{i:05d}"
                wav_filename = f"{uid}.wav"
                wav_path = os.path.join(wavs_dir, wav_filename)
                
                sf.write(wav_path, audio, TARGET_SR, subtype='PCM_16')
                
                # Determine speaker: use 'gender' if available, otherwise 'unknown'
                speaker = item.get("gender")
                if speaker not in ["male_masculine", "female_feminine"]: # Normalize speaker tags
                    speaker = "unknown"
                    
                writer.writerow([wav_filename, speaker, text]) # Store filename instead of full uid for easier parsing
                processed_count += 1
            except Exception as e:
                logging.error(f"Error processing item {i} in {split_name}: {e}. Original sentence: '{item['sentence']}'. Skipping.")
                skipped_count +=1
                continue
                
    logging.info(f"Finished processing {split_name}: {processed_count} samples written, {skipped_count} samples skipped.")


def prepare_dataset(data_set_path, hf_dataset_name="btsee/common_voice_21_mn_cyrillic"):
    """Loads dataset from Hugging Face, processes, and saves it."""
    logging.info(f"Preparing dataset in {data_set_path} from {hf_dataset_name}")
    os.makedirs(data_set_path, exist_ok=True)
    wavs_dir = os.path.join(data_set_path, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    try:
        logging.info("Loading training data...")
        train_ds = load_dataset(hf_dataset_name, split="train", trust_remote_code=True)
        logging.info("Loading test data...")
        test_ds = load_dataset(hf_dataset_name, split="test", trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load dataset from Hugging Face: {e}")
        raise

    process_split("train", train_ds, data_set_path, wavs_dir)
    process_split("test", test_ds, data_set_path, wavs_dir)
    
    logging.info(f"Dataset preparation complete. Check logs for details on processed/skipped samples.")