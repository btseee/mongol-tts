import os
import csv

# Paths
text_file = "data/processed/text.txt"
audio_folder = "data/processed/audio_segments"
metadata_file = "data/processed/metadata.csv"

# Read sentences from text file (line by line)
with open(text_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

# Sort audio files numerically (e.g., segment_1.wav, segment_2.wav, ...)
audio_files = sorted(
    [f for f in os.listdir(audio_folder) if f.endswith(".wav")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Ensure we only process the minimum count
min_length = min(len(lines), len(audio_files))
lines = lines[:min_length]
audio_files = audio_files[:min_length]

# Write metadata.csv
with open(metadata_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for text, audio in zip(lines, audio_files):
        audio_path = f"audio_segments/{audio}"
        writer.writerow([audio_path, text])

print(f"✅ metadata.csv saved at: {metadata_file}")