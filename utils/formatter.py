import os, logging

def common_voices_mn(root_path, meta_file, **kwargs):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_hash_to_label_map = {}
    next_speaker_id_counter = 1

    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            line = line.strip()
            if not line:
                continue
            
            cols = line.split("|")
            if len(cols) < 3:
                logging.warning(f"Skipping malformed line: {line}")
                continue
 
            text = cols[2]
            speaker= cols[1]

            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker,
                "root_path": root_path
            })
    return items