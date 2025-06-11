import os

def mbspeech(root_path, meta_file, **kwargs): 
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mbspeech"
    language = "mn"
    
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1].strip()
            items.append({
                "text": text, 
                "audio_file": wav_file, 
                "speaker_name": speaker_name, 
                "root_path": root_path, 
                "language": language
            })
    return items