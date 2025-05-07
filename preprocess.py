import os
import csv
import re
import argparse
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf
from utils.num2words.num2words import num2words

_MONGOLIAN_CYRILLIC_CHARS = 'абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ'
_ACCEPTED_CHARS = _MONGOLIAN_CYRILLIC_CHARS + ' .,!?-:'
_PUNCTUATION = '.,!?-:'

_REPLACEMENTS = {
    '–': '-', '—': '-', '‘': "'", '’': "'",
    '“': '"', '”': '"', '„': '"', '\u00A0': ' ', '\u202F': ' ', '\n': '',
}

_LATIN_TO_MN = {
    'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
    'g': 'жи', 'h': 'эйч', 'i': 'ай', 'j': 'жей', 'k': 'кей', 'l': 'эл',
    'm': 'эм', 'n': 'эн', 'o': 'о', 'p': 'пи', 'q': 'кью', 'r': 'ар',
    's': 'эс', 't': 'ти', 'u': 'ю', 'v': 'ви', 'w': 'дабл-ю', 'x': 'икс',
    'y': 'вай', 'z': 'зед'
}

def expand_numbers_and_letters(text):
    def replace_num(match):
        num = int(match.group(0))
        return num2words(num, lang='mn')
    text = re.sub(r'\d+', replace_num, text)
    def replace_letter(match):
        letter = match.group(0).lower()
        return _LATIN_TO_MN.get(letter, letter)
    text = re.sub(r'[A-Za-z]', replace_letter, text)
    return text

def normalize_text(text):
    text = text.lower()
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = expand_numbers_and_letters(text)
    text = ''.join(c for c in text if c in _ACCEPTED_CHARS)
    text = text.strip().strip(_PUNCTUATION)
    text = re.sub(r'\s+', ' ', text)
    return text

def resample_audio(input_path, output_path, target_sr=22050):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sr)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-acodec", "pcm_s16le"])
        return True
    except:
        return False

def prepare_metadata(cv_data_dir, output_dir, metadata_filename="metadata.csv", target_sr=22050, tsv_filename="validated.tsv"):
    tsv_path = os.path.join(cv_data_dir, tsv_filename)
    clips_dir = os.path.join(cv_data_dir, "clips")
    output_metadata_path = os.path.join(output_dir, metadata_filename)
    output_wavs_dir = os.path.join(output_dir, "wavs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_wavs_dir, exist_ok=True)

    metadata = []
    skipped_count = 0
    resample_errors = 0

    with open(tsv_path, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for row in open(tsv_path, 'r', encoding='utf-8')) - 1

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(reader, total=total_rows, desc="Processing clips"):
            try:
                clip_filename = row['path']
                text = row['sentence']

                if not text:
                    skipped_count += 1
                    continue

                original_clip_path = os.path.join(clips_dir, clip_filename)
                if not os.path.exists(original_clip_path):
                    skipped_count += 1
                    continue

                normalized_text = normalize_text(text)
                if not normalized_text:
                    skipped_count += 1
                    continue

                output_wav_filename = os.path.splitext(clip_filename)[0] + ".wav"
                processed_clip_path = os.path.join(output_wavs_dir, output_wav_filename)

                resampled_successfully = True
                if target_sr is not None:
                    try:
                        info = sf.info(original_clip_path)
                        if info.samplerate != target_sr:
                            resampled_successfully = resample_audio(original_clip_path, processed_clip_path, target_sr)
                        else:
                            audio = AudioSegment.from_file(original_clip_path)
                            audio.export(processed_clip_path, format="wav", parameters=["-ac", "1", "-acodec", "pcm_s16le"])
                    except:
                        resampled_successfully = False
                        resample_errors += 1
                else:
                    try:
                        audio = AudioSegment.from_file(original_clip_path)
                        audio.export(processed_clip_path, format="wav", parameters=["-ac", "1", "-acodec", "pcm_s16le"])
                    except:
                        resampled_successfully = False
                        resample_errors += 1

                if resampled_successfully:
                    clip_id = os.path.splitext(clip_filename)[0]
                    metadata.append(f"{clip_id}|{normalized_text}|{normalized_text}")
                else:
                    skipped_count += 1
                    resample_errors += 1

            except:
                skipped_count += 1

    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        for line in tqdm(metadata, desc="Writing metadata"):
            f.write(line + '\n')

    print(f"Total processed entries: {len(metadata)}")
    print(f"Skipped entries: {skipped_count}")
    if target_sr is not None:
        print(f"Resampling/Conversion errors: {resample_errors}")
    print(f"Metadata file saved to: {output_metadata_path}")
    print(f"Processed audio files saved in: {output_wavs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("cv_data_dir", type=str)
    parser.add_argument("output_dir", type=str, default="data/common_voice_mn")
    parser.add_argument("--metadata_filename", type=str, default="metadata.csv")
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--tsv_file", type=str, default="validated.tsv")

    args = parser.parse_args()
    target_sr_value = args.target_sr if args.target_sr > 0 else None

    prepare_metadata(
        cv_data_dir=args.cv_data_dir,
        output_dir=args.output_dir,
        metadata_filename=args.metadata_filename,
        target_sr=target_sr_value,
        tsv_filename=args.tsv_file
    )
