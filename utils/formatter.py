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
            speaker_hash = cols[1]

            if speaker_hash not in speaker_hash_to_label_map:
                speaker_hash_to_label_map[speaker_hash] = f"CV_Speaker_{next_speaker_id_counter:04d}"
                next_speaker_id_counter += 1
            
            readable_speaker_name = speaker_hash_to_label_map[speaker_hash]

            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": readable_speaker_name,
                "root_path": root_path
            })
    return items