import csv
import random

# Load metadata
with open('data/metadata.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    data = list(reader)

# Shuffle the data to ensure random splitting
random.shuffle(data)

# Adjust paths to be relative to the mongolian-tts/ directory
for row in data:
    row[0] = 'data/' + row[0]  # Changes audio_segments/segment_1.wav to data/audio_segments/segment_1.wav

# Split into training (90%) and validation (10%) sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# Write training CSV
with open('data/train.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerows(train_data)

# Write validation CSV
with open('data/val.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerows(val_data)