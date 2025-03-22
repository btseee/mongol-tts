import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import argparse

def separate_audio(audio_path, output_folder, silence_threshold=-50, min_silence_len=900):
    """
    Аудио файлыг чимээгүй хэсгүүдээр нь тасалж, тусдаа WAV файлууд болгон хадгална.
    
    Аргументууд:
        audio_path (str): Оролтын аудио файлын зам.
        output_folder (str): Сегментүүдийг хадгалах директорын зам.
        silence_threshold (int): Чимээгүй байдлын босго (dBFS).
        min_silence_len (int): Хамгийн бага чимээгүй хугацаа (миллисекунд).
    """
    # Аудио файлыг ачаалах
    audio = AudioSegment.from_file(audio_path, format="wav")

    # Чимээгүй хэсгүүдийг илрүүлэх
    non_silent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)

    # Сегментүүдийг хадгалах директор үүсгэх
    os.makedirs(output_folder, exist_ok=True)
    print(f"Сегментүүдийг хадгалах газар: {output_folder}")

    # Сегментүүдийг тасалж хадгалах
    total_segments = len(non_silent_chunks)
    for i, (start, end) in enumerate(non_silent_chunks):
        segment = audio[start:end]
        segment_path = os.path.join(output_folder, f"segment_{i+1}.wav")
        segment.export(segment_path, format="wav")
        print(f"Хадгалсан: {segment_path} ({start}ms - {end}ms) - {i+1}/{total_segments}")

if __name__ == "__main__":
    # Командын мөрний аргументыг унших
    parser = argparse.ArgumentParser(description="Аудио файлыг сегмент болгон хуваах скрипт")
    parser.add_argument("audio_path", help="Оролтын аудио файлын зам")
    parser.add_argument("output_folder", help="Сегментүүдийг хадгалах директорын зам")
    parser.add_argument("--silence_threshold", type=int, default=-50, help="Чимээгүй байдлын босго (dBFS)")
    parser.add_argument("--min_silence_len", type=int, default=900, help="Хамгийн бага чимээгүй хугацаа (миллисекунд)")
    args = parser.parse_args()
    separate_audio(args.audio_path, args.output_folder, args.silence_threshold, args.min_silence_len)