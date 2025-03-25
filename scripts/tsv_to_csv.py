import os
import csv

# Define paths
tsv_file = "/home/tsee/Downloads/cv-corpus-21.0-2025-03-14-mn/cv-corpus-21.0-2025-03-14/mn/train.tsv"
output_csv = "dataset/metadata_train.csv"  # Output LJSpeech format file
audio_dir = "wavs"  # Folder where .wav files are stored

# Open and process the TSV file
with open(tsv_file, "r", encoding="utf-8") as infile, open(output_csv, "w", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile, delimiter="\t")
    for row in reader:
        audio_filename = row["path"].replace(".mp3", "")  # Convert .mp3 to .wav
        text = row["sentence"]
        speaker_id = row["client_id"][:8]  # Use a short version of client_id as speaker ID
        
        # Construct LJSpeech metadata format
        outfile.write(f"{audio_filename}|{text}\n")

print(f"Conversion completed! Metadata saved to {output_csv}")
