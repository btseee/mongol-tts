import os
import subprocess
import shutil
import argparse
import logging

def convert_wavs(audio_dir):
    """
    Аудио директор дахь бүх WAV файлуудыг 16-бит PCM, 22050 Hz, моно болгон хөрвүүлнэ.
    Анхны файлуудын нөөцлөлтийг 'backup' дэд директорт хадгална.
    
    Аргументууд:
        audio_dir (str): WAV файлууд агуулсан директорын зам.
    """
    # Нөөцлөлтийн директор үүсгэх
    backup_dir = os.path.join(audio_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # Цэвэрлэсэн файлуудыг хадгалах директор үүсгэх
    cleaned_dir = os.path.join(audio_dir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    # Алдааны лог файл тохируулах
    logging.basicConfig(filename=os.path.join(audio_dir, 'conversion.log'), level=logging.ERROR)

    # WAV файлуудыг боловсруулах
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            input_path = os.path.join(audio_dir, filename)
            backup_path = os.path.join(backup_dir, filename)
            output_path = os.path.join(cleaned_dir, filename)

            # Анхны файлыг нөөцлөх
            shutil.copy(input_path, backup_path)

            # ffmpeg ашиглан хөрвүүлэх
            command = [
                "ffmpeg",
                "-i", input_path,
                "-ac", "1",           # Моно
                "-ar", "22050",       # 22050 Hz давтамж
                "-acodec", "pcm_s16le",  # 16-бит PCM
                "-y",                 # Зөвшөөрөлгүйгээр дахин бичих
                output_path
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Хөрвүүлсэн: {filename}")
            except subprocess.CalledProcessError as e:
                error_message = f"{filename} хөрвүүлэхэд алдаа гарлаа: {e.stderr.decode()}"
                print(error_message)
                logging.error(error_message)

if __name__ == "__main__":
    # Командын мөрний аргументыг унших
    parser = argparse.ArgumentParser(description="WAV файлуудыг цэвэрлэх скрипт")
    parser.add_argument("audio_dir", help="WAV файлууд агуулсан директорын зам")
    args = parser.parse_args()
    convert_wavs(args.audio_dir)