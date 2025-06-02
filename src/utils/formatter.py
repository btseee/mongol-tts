import os

def common_voice(root_path, meta_file, **kwargs): 
    """
    Formatter for Common Voice like datasets.
    Assumes meta_file CSV format: audio_filename|speaker_name|text
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            line = line.strip()
            if not line:
                continue
            try:
                cols = line.split("|")
                if len(cols) < 3:
                    # print(f"Warning: Skipping malformed line in {meta_file}: {line}")
                    continue
                
                wav_filename = cols[0] # Expecting just the filename, e.g., train_00000.wav
                wav_file = os.path.join(root_path, "wavs", wav_filename)
                text = cols[2]
                speaker_name = cols[1]
                
                # Basic validation
                if not os.path.exists(wav_file):
                    # print(f"Warning: Audio file not found: {wav_file}. Skipping entry: {line}")
                    continue
                if not text:
                    # print(f"Warning: Empty text for audio file: {wav_file}. Skipping entry: {line}")
                    continue

                items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
            except Exception as e:
                # print(f"Error parsing line in {meta_file}: {line} - {e}")
                continue
    return items