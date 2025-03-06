import os
import librosa
import soundfile as sf

input_dir = "data/audio_segments_processed"
output_dir = "data/audio_segments"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        y, sr = librosa.load(filepath, sr=22050, mono=True)
        y = librosa.util.normalize(y)  # Normalize amplitude
        sf.write(os.path.join(output_dir, filename), y, sr)