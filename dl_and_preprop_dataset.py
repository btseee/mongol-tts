#!/usr/bin/env python
import os
import sys
import csv
import time
import argparse
import fnmatch
import librosa
import pandas as pd
import tarfile
import shutil
from tqdm import tqdm

from hparams import HParams as hp
from zipfile import ZipFile
from audio import preprocess
from utils import download_file
from datasets.mb_speech import MBSpeech, text_normalize as mb_text_normalize
from datasets.lj_speech import LJSpeech, text_normalize as lj_text_normalize
from datasets.cv_speech import CVSpeech, text_normalize as cv_text_normalize # Import new dataset
import soundfile as sf

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'cvspeech'], help='dataset name') # Add cvspeech
args = parser.parse_args()

datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

if args.dataset == 'ljspeech':
    dataset_name = 'LJSpeech-1.1'
    dataset_archive_name = 'LJSpeech-1.1.tar.bz2'
    dataset_path = os.path.join(datasets_path, dataset_name)
    text_normalizer = lj_text_normalize

    if os.path.isdir(dataset_path):
        print(f"{dataset_name} dataset folder already exists")
    else:
        dataset_archive_path = os.path.join(datasets_path, dataset_archive_name)
        if not os.path.isfile(dataset_archive_path):
            url = "http://data.keithito.com/data/speech/%s" % dataset_archive_name
            print(f"Downloading {dataset_name}...")
            download_file(url, dataset_archive_path)
        else:
            print(f"'{dataset_archive_name}' already exists")

        print(f"Extracting '{dataset_archive_name}'...")
        os.makedirs(dataset_path, exist_ok=True)
        os.system(f'tar xvjf "{dataset_archive_path}" -C "{datasets_path}"') # Extract to datasets_path

    # Preprocess
    print("Pre-processing LJSpeech...")
    lj_speech_instance = LJSpeech([])
    preprocess(dataset_path, lj_speech_instance) # Assumes preprocess looks for wavs in dataset_path/wavs

elif args.dataset == 'mbspeech':
    dataset_name = 'MBSpeech-1.0'
    dataset_path = os.path.join(datasets_path, dataset_name)
    text_normalizer = mb_text_normalize

    if os.path.isdir(dataset_path):
         print("MBSpeech dataset folder already exists, skipping download and initial prep.")
         # Still run preprocess in case it was interrupted
    else:
        os.makedirs(dataset_path, exist_ok=True)
        wavs_path = os.path.join(dataset_path, 'wavs')
        os.makedirs(wavs_path, exist_ok=True)

        # Download audio zip files
        bible_books = ['01_Genesis', '02_Exodus', '03_Leviticus']
        for bible_book_name in bible_books:
            bible_book_file_name = f'{bible_book_name}.zip'
            bible_book_file_path = os.path.join(datasets_path, bible_book_file_name)
            if not os.path.isfile(bible_book_file_path):
                url = f"https://s3.us-east-2.amazonaws.com/bible.davarpartners.com/Mongolian/{bible_book_file_name}"
                print(f"Downloading {bible_book_file_name}...")
                download_file(url, bible_book_file_path)
            else:
                print(f"'{bible_book_file_name}' already exists")

            print(f"Extracting '{bible_book_file_name}'...")
            # Extract directly into datasets_path, should create folder like '01_Genesis'
            with ZipFile(bible_book_file_path, 'r') as zip_ref:
                zip_ref.extractall(datasets_path)

        # Download and extract CSV metadata
        dataset_csv_file_name = f'{dataset_name}-csv.zip'
        dataset_csv_file_path = os.path.join(datasets_path, dataset_csv_file_name)
        dataset_csv_extracted_path = os.path.join(datasets_path, f'{dataset_name}-csv')
        if not os.path.isfile(dataset_csv_file_path):
            url = f"https://www.dropbox.com/s/dafueq0w278lbz6/{dataset_name}-csv.zip?dl=1"
            print(f"Downloading {dataset_csv_file_name}...")
            download_file(url, dataset_csv_file_path)
        else:
            print(f"'{dataset_csv_file_name}' already exists")

        print(f"Extracting '{dataset_csv_file_name}'...")
        with ZipFile(dataset_csv_file_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_path) # Extracts to 'MBSpeech-1.0-csv' folder

        # Process MP3s to WAVs and create metadata.csv
        sample_rate = 44100  # original sample rate
        total_duration_s = 0

        metadata_csv_path = os.path.join(dataset_path, 'metadata.csv')
        print(f"Creating metadata file at: {metadata_csv_path}")
        with open(metadata_csv_path, 'w', encoding='utf-8', newline='') as metadata_csv:
            metadata_csv_writer = csv.writer(metadata_csv, delimiter='|')

            def _normalize_mb(s):
                s = s.strip()
                if s and (s[0] == '—' or s[0] == '-'):
                    s = s[1:].strip()
                return text_normalizer(s) # Apply mb_speech normalization

            def _get_mp3_file(book_name, chapter):
                book_download_path = os.path.join(datasets_path, book_name)
                wildcard = f"* {chapter:02d} - DPI.mp3" # Adjusted wildcard based on observed filenames
                for file_name in os.listdir(book_download_path):
                    if fnmatch.fnmatch(file_name, wildcard):
                        return os.path.join(book_download_path, file_name)
                print(f"Warning: MP3 file not found for {book_name} chapter {chapter} with wildcard '{wildcard}' in {book_download_path}")
                return None

            def _convert_mp3_to_wav(book_name, book_nr):
                global total_duration_s
                chapter = 1
                while True:
                    chapter_csv_file_name = os.path.join(dataset_csv_extracted_path, f"{book_name}_{chapter:02d}.csv")
                    if not os.path.exists(chapter_csv_file_name):
                        print(f"CSV file not found for chapter {chapter}, stopping for book {book_name}.")
                        break # Stop processing this book

                    try:
                        df = pd.read_csv(chapter_csv_file_name, sep="|")
                        print(f"Processing {chapter_csv_file_name}...")
                        mp3_file = _get_mp3_file(book_name, chapter)

                        if mp3_file is None:
                            chapter += 1
                            continue # Skip to next chapter if mp3 not found

                        print(f"Loading MP3: {mp3_file}...")
                        samples, sr = librosa.load(mp3_file, sr=sample_rate, mono=True)
                        assert sr == sample_rate

                        i = 0
                        for index, row in df.iterrows():
                            start, end, sentence = row['start'], row['end'], row['sentence']
                            if not isinstance(sentence, str) or pd.isna(sentence):
                                print(f"Skipping row {index+1} due to invalid sentence: {sentence}")
                                continue
                            if pd.isna(start) or pd.isna(end) or end <= start:
                                print(f"Skipping row {index+1} due to invalid start/end times: start={start}, end={end}")
                                continue

                            start, end = int(start), int(end) # Ensure integers
                            duration = end - start
                            duration_s = duration / sample_rate # Duration in seconds

                            if duration_s < 0.5 or duration_s > 15: # Filter audio length (adjust as needed)
                                continue

                            total_duration_s += duration_s
                            i += 1
                            normalized_sentence = _normalize_mb(sentence)
                            if not normalized_sentence: # Skip if normalization results in empty string
                                print(f"Skipping row {index+1} due to empty normalized sentence: original='{sentence}'")
                                continue

                            fn = f"MB{book_nr}{chapter:02d}-{i:04d}"
                            metadata_csv_writer.writerow([fn, normalized_sentence, normalized_sentence])
                            wav = samples[start:end]
                            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=hp.sr)
                            wav_path = os.path.join(wavs_path, f"{fn}.wav")
                            librosa.output.write_wav(wav_path, wav, hp.sr)

                        chapter += 1
                    except FileNotFoundError:
                         print(f"CSV file not found: {chapter_csv_file_name}, stopping for book {book_name}.")
                         break # Stop if CSV is missing
                    except Exception as e:
                         print(f"Error processing chapter {chapter} of {book_name}: {e}")
                         chapter += 1 # Try next chapter

            _convert_mp3_to_wav('01_Genesis', 1)
            _convert_mp3_to_wav('02_Exodus', 2)
            _convert_mp3_to_wav('03_Leviticus', 3)

        print("Finished converting MP3s and creating metadata.csv")
        print("Total audio duration (filtered): %s" % (time.strftime('%H:%M:%S', time.gmtime(total_duration_s))))

    # Preprocess Mels/Mags
    print("Pre-processing MBSpeech...")
    mb_speech_instance = MBSpeech([])
    preprocess(dataset_path, mb_speech_instance) # Assumes preprocess looks for wavs in dataset_path/wavs
elif args.dataset == 'cvspeech':
    cv_download_url = "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-21.0-2025-03-14/cv-corpus-21.0-2025-03-14-mn.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250426%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250426T145654Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=062439cfc04ca1307a5e3f37a68ad74ae123e7fd33798a83ae2a7c9f9e6f9ff81c53d34ef6c172267eeedeb2985b4462b5ab13a8e39632bb28a2b3c0ca85943b271ccd7f75125801ad75976f44bddd998df3cf6bd78c6eb5295ef20c005c00c6487308659d1c6e0661a85241cc0c15a2306d7133af8650a795482848c6dd4765f59c89e817c32f523221131466db7349103d6baf6f7ff101664441d66204994f29c5e189f3dbaa4daba4a54326936c8a1bacbcd6fa13c7801147d6902839fec60d0df7e614f8e4ca24d00bdcd54af3c78e56680910eccd1560ee17df7b1e1b365dedc09bcf09979e189520a15972f57c86d3d74d054100327b952a76491e1904"
    # Extract expected dataset name and archive name from URL (best guess)
    try:
         url_path = cv_download_url.split('?')[0] # Get part before query params
         url_parts = url_path.split('/')
         dataset_archive_name = url_parts[-1] # e.g., cv-corpus-21.0-2025-03-14-mn.tar.gz
         # Try to infer dataset name from archive name (might need adjustment)
         dataset_name = dataset_archive_name.replace('.tar.gz', '') # e.g., cv-corpus-21.0-2025-03-14-mn
         language = dataset_name.split('-')[-1] # 'mn'
         cv_internal_folder = f'{dataset_name}/{language}/' # Expected path inside tar
    except Exception as e:
         print(f"Warning: Could not reliably parse dataset name from URL '{cv_download_url}'. Using defaults. Error: {e}")
         # Fallback defaults if parsing fails
         dataset_name = 'cv-corpus-21.0-download' # Generic name
         dataset_archive_name = 'cv-corpus-mn.tar.gz' # Generic name
         language = 'mn'
         cv_internal_folder = None # Need manual check if extraction fails


    dataset_path = os.path.join(datasets_path, dataset_name)
    text_normalizer = cv_text_normalize


    if os.path.isdir(dataset_path):
         print(f"Common Voice dataset folder '{dataset_path}' already exists, skipping download and initial prep.")
    else:
        os.makedirs(dataset_path, exist_ok=True)
        wavs_path = os.path.join(dataset_path, 'wavs')
        os.makedirs(wavs_path, exist_ok=True)

        dataset_archive_path = os.path.join(datasets_path, dataset_archive_name)
        if not os.path.isfile(dataset_archive_path):
            print(f"Downloading Common Voice dataset using provided URL...")
            try:
                download_file(cv_download_url, dataset_archive_path)
            except Exception as e:
                print(f"\nError downloading file from the provided URL.")
                print(f"URL: {cv_download_url}")
                print(f"Error: {e}")
                print("The URL might have expired or be incorrect. Please obtain a fresh link from Common Voice.")
                # Clean up potentially incomplete download
                if os.path.exists(dataset_archive_path): os.remove(dataset_archive_path)
                sys.exit(1)
        else:
            print(f"'{dataset_archive_name}' already exists")

        print(f"Extracting '{dataset_archive_name}'...")
        try:
            with tarfile.open(dataset_archive_path, "r:gz") as tar:
                # Attempt to extract intelligently, removing top-level directory if possible
                members_to_extract = []
                # Determine the common prefix (likely the folder inside the tar)
                common_prefix = os.path.commonprefix(tar.getnames())
                if common_prefix and common_prefix.endswith('/'):
                     print(f"Common prefix detected: {common_prefix}. Stripping prefix during extraction.")
                     strip_len = len(common_prefix)
                     for member in tar.getmembers():
                         if member.name.startswith(common_prefix):
                             member.name = member.name[strip_len:] # Remove prefix
                             if member.name: # Don't add empty names (the directory itself)
                                members_to_extract.append(member)
                         else:
                             # Keep members not starting with prefix (unlikely but possible)
                             members_to_extract.append(member)
                else:
                    print("No common prefix detected or not ending with '/', extracting all members.")
                    members_to_extract = tar.getmembers()


                if not members_to_extract:
                     raise ValueError(f"No members found to extract from the tar archive '{dataset_archive_path}'. It might be empty or corrupted.")

                print(f"Extracting {len(members_to_extract)} files/dirs to {dataset_path}...")
                # Extract selected members directly into dataset_path
                tar.extractall(path=dataset_path, members=members_to_extract)

        except tarfile.ReadError as e:
             print(f"Error reading tar file {dataset_archive_path}. It might be corrupted or incomplete. {e}")
             sys.exit(1)
        except Exception as e:
             print(f"An error occurred during extraction: {e}")
             sys.exit(1)

        print("Extraction complete.")


        # --- Locate essential files after extraction ---
        # The exact paths depend on how the tar was structured and extracted.
        # Common possibilities:
        # 1. Files directly in dataset_path (validated.tsv, clips/)
        # 2. Files inside a language subfolder (dataset_path/mn/validated.tsv, dataset_path/mn/clips/)
        # 3. Files inside the full named folder (dataset_path/cv-corpus-21.0...-mn/validated.tsv, ...) - less likely with prefix stripping

        possible_tsv_paths = [
            os.path.join(dataset_path, 'validated.tsv'),
            os.path.join(dataset_path, language, 'validated.tsv')
        ]
        possible_clips_paths = [
            os.path.join(dataset_path, 'clips'),
            os.path.join(dataset_path, language, 'clips')
        ]

        tsv_file_path = None
        for path in possible_tsv_paths:
            if os.path.isfile(path):
                tsv_file_path = path
                print(f"Found TSV file at: {tsv_file_path}")
                break

        clips_path = None
        for path in possible_clips_paths:
            if os.path.isdir(path):
                clips_path = path
                print(f"Found clips directory at: {clips_path}")
                break

        if not tsv_file_path:
             print(f"Error: validated.tsv not found in expected locations within '{dataset_path}'. Check extraction structure.")
             print(f"Looked in: {possible_tsv_paths}")
             sys.exit(1)
        if not clips_path:
             print(f"Error: clips directory not found in expected locations within '{dataset_path}'. Check extraction structure.")
             print(f"Looked in: {possible_clips_paths}")
             sys.exit(1)
        # --- End file location ---

        print(f"Processing TSV file: {tsv_file_path}")
        metadata_csv_path = os.path.join(dataset_path, 'metadata.csv')
        total_duration_s = 0
        file_counter = 0
        skipped_count = 0

        try:
            # Try reading with QUOTE_NONE first, as TSVs might not be quoted
            df = pd.read_csv(tsv_file_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='warn')
            print(f"Read {len(df)} entries from {os.path.basename(tsv_file_path)} (quoting=NONE).")
        except Exception as e:
            print(f"Warning: Failed reading TSV with QUOTE_NONE ({e}). Retrying with default quoting...")
            try:
                 # Fallback to default quoting if QUOTE_NONE fails
                 df = pd.read_csv(tsv_file_path, sep='\t', on_bad_lines='warn')
                 print(f"Read {len(df)} entries from {os.path.basename(tsv_file_path)} (default quoting).")
            except Exception as e2:
                 print(f"Error: Failed to read TSV file {tsv_file_path} with multiple quoting options: {e2}")
                 sys.exit(1)


        with open(metadata_csv_path, 'w', encoding='utf-8', newline='') as metadata_csv:
            metadata_csv_writer = csv.writer(metadata_csv, delimiter='|')

            required_columns = ['path', 'sentence']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: TSV file {tsv_file_path} is missing required columns. Expected: {required_columns}, Found: {list(df.columns)}")
                sys.exit(1)

            for index, row in tqdm(df.iterrows(), total=len(df), desc="Converting MP3 to WAV"):
                try:
                    mp3_filename = row['path']
                    sentence = row['sentence']

                    # Basic validation
                    if not isinstance(sentence, str) or pd.isna(sentence) or not sentence.strip(): skipped_count+=1; continue
                    if not isinstance(mp3_filename, str) or pd.isna(mp3_filename): skipped_count+=1; continue

                    mp3_filepath = os.path.join(clips_path, mp3_filename)
                    if not os.path.isfile(mp3_filepath): skipped_count+=1; continue

                    # Load MP3, get duration
                    try:
                         y, sr = librosa.load(mp3_filepath, sr=None, mono=True) # Load original SR
                         duration_s = librosa.get_duration(y=y, sr=sr)
                    except Exception as load_err:
                         print(f"Warning: Error loading MP3 '{mp3_filepath}' (row {index+1}): {load_err}")
                         skipped_count+=1; continue

                    # Filter by duration
                    if duration_s < 0.5 or duration_s > 15.0: skipped_count+=1; continue

                    normalized_sentence = text_normalizer(sentence)
                    if not normalized_sentence: skipped_count+=1; continue

                    # Create new filename
                    file_counter += 1
                    new_wav_basename = f"CVMN-{file_counter:07d}" # Padded more for larger datasets
                    wav_filepath = os.path.join(wavs_path, f"{new_wav_basename}.wav")

                    # Resample and save WAV
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=hp.sr)
                    sf.write(wav_filepath, y_resampled, hp.sr, format='WAV')

                    # Write to metadata CSV
                    metadata_csv_writer.writerow([new_wav_basename, normalized_sentence, normalized_sentence])
                    total_duration_s += duration_s

                except KeyError as e:
                    print(f"Warning: Missing expected column '{e}' in TSV row {index+1}. Skipping.")
                    skipped_count+=1; continue
                except Exception as e:
                    print(f"Warning: Unexpected error processing row {index+1} ('{mp3_filename}'): {e}")
                    skipped_count+=1; continue

        print(f"Finished converting MP3s and creating metadata.csv.")
        print(f"Processed {file_counter} valid audio files.")
        print(f"Skipped {skipped_count} entries due to errors or filtering.")
        print("Total audio duration (original, filtered): %s" % (time.strftime('%H:%M:%S', time.gmtime(total_duration_s))))

        # Optional: Clean up raw extracted files (Use with caution!)
        # print("Cleaning up extracted raw Common Voice files...")
        # files_to_remove = [...] # List files like README, TSVs etc. in dataset_path
        # for f_or_d in files_to_remove:
        #    item_path = os.path.join(dataset_path, f_or_d)
        #    try:
        #        if os.path.isfile(item_path): os.remove(item_path)
        #        elif os.path.isdir(item_path): shutil.rmtree(item_path)
        #    except Exception as e: print(f"Error removing {item_path}: {e}")
        # if os.path.isdir(clips_path) and clips_path != os.path.join(dataset_path, 'clips'): # Avoid deleting clips if it's the main folder
        #      try: shutil.rmtree(clips_path)
        #      except Exception as e: print(f"Error removing clips dir {clips_path}: {e}")


    # Preprocess Mels/Mags
    print("Pre-processing Common Voice Speech...")
    try:
        # Instantiate the dataset class to pass path info to preprocess
        cv_speech_instance = CVSpeech(keys=[], dir_name=dataset_name) # Pass the actual dataset folder name
        preprocess(dataset_path, cv_speech_instance) # Assumes preprocess uses dataset_path/wavs and dataset_path/metadata.csv
    except Exception as e:
        print(f"Error during final preprocessing step for Common Voice: {e}")
        print(f"Check if 'preprocess' function correctly uses path: {dataset_path}")
        print(f"Check if '{os.path.join(dataset_path, 'metadata.csv')}' and '{os.path.join(dataset_path, 'wavs')}' exist and are populated.")
        sys.exit(1)

else:
    print(f"Unknown dataset: {args.dataset}")
    sys.exit(1)

print("Dataset preparation and preprocessing finished.")