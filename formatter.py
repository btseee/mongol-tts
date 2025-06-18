import os

def formatter(root_path, meta_file, **kwargs):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mbspeech"
    language = "mn"
    
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            wav_file = os.path.join(root_path, "wavs", f"{cols[0]}.wav")
            text = cols[2].strip()
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path,
                "language": language,
            })
    return items
