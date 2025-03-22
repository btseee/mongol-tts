import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load audio file
audio_path = os.path.join(BASE_PATH, "new_dataset", "monte.wav")

audio = AudioSegment.from_file(audio_path, format="wav")

# Detect non-silent chunks
silence_threshold = -50  # dBFS
min_silence_len = 900  # milliseconds
non_silent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)

# Create a folder to save segments
output_folder = os.path.join(BASE_PATH, "new_dataset", "wavs")
os.makedirs(output_folder, exist_ok=True)
print(f"Saving segments to: {output_folder}")

# Extract and save segments
for i, (start, end) in enumerate(non_silent_chunks):
    segment = audio[start:end]
    segment.export(os.path.join(output_folder, f"segment_{i+1}.wav"), format="wav")
    print(f"Saved: {output_folder}/segment_{i}.wav ({start}ms - {end}ms)")
