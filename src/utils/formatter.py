
import os

def common_voice(root_path, meta_file, **kwargs): 
    txt_file = os.path.join(root_path, meta_file)
    items = []
    
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            speaker_name = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items
