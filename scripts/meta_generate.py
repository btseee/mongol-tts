import os
import csv
import argparse
import logging

def generate_metadata(text_file, audio_folder, metadata_file):
    """
    Текст файл болон аудио хавтасаас metadata.csv файлыг үүсгэх

    Параметрүүд:
        text_file (str): Өгүүлбэрүүдийг агуулсан текст файлын зам.
        audio_folder (str): WAV форматаар аудио файлууд агуулагдсан хавтас.
        metadata_file (str): Гаралтын metadata.csv файлын зам.

    Алдаа:
        FileNotFoundError: Хэрэв текст файл эсвэл аудио хавтас олдсонгүй.
        ValueError: Хэрэв текст мөрүүд болон аудио файлуудын тоо тохирохгүй бол.

    Буцаах утга:
        Тохирох утга буцаахгүй
    """
    # Алдаа бичлэгийг тохируулах
    logging.basicConfig(filename='metadata_generation.log', level=logging.ERROR)

    # Текст файл болон аудио хавтас байгаа эсэхийг шалгах
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Текст файл олдсонгүй: {text_file}")
    if not os.path.exists(audio_folder):
        raise FileNotFoundError(f"Аудио хавтас олдсонгүй: {audio_folder}")

    # Текст файлаас өгүүлбэрүүдийг унших (хоосон мөрүүдийг орхих)
    with open(text_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Аудио файлуудыг дугаарын дагуу эрэмбэлэх (жишээ нь, segment_1.wav, segment_2.wav, ...)
    audio_files = sorted(
        [f for f in os.listdir(audio_folder) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    # Текст мөрүүдийн тоо болон аудио файлуудын тоо тохирч байгаа эсэхийг шалгах
    if len(lines) != len(audio_files):
        raise ValueError(
            f"Текст мөрүүдийн тоо ({len(lines)}) аудио файлуудын тоотой ({len(audio_files)}) тохирохгүй байна"
        )

    # Гаралтын хавтас байхгүй бол үүсгэх
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    # metadata.csv файлыг бичих
    with open(metadata_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for text, audio in zip(lines, audio_files):
            # 'segment_X.wav' файлнээс 'segment_X' нэрийг авах
            segment_name = audio.split(".")[0]
            writer.writerow([segment_name, "unused", text])

    print(f"✅ metadata.csv файл хадгалагдлаа: {metadata_file}")

def main():
    """
    Скриптийг ажиллуулах гол функц. Командын мөрийн аргументуудыг парс хийж авна.
    """
    parser = argparse.ArgumentParser(description="Текст болон аудио файлуудаас metadata.csv үүсгэх")
    parser.add_argument("text_file", help="Өгүүлбэрүүдийг агуулсан текст файлын зам")
    parser.add_argument("audio_folder", help="WAV аудио файлуудыг агуулсан хавтасны зам")
    parser.add_argument("metadata_file", help="Гаралтын metadata.csv файлын зам")

    args = parser.parse_args()

    try:
        generate_metadata(args.text_file, args.audio_folder, args.metadata_file)
    except Exception as e:
        error_message = f"Алдаа гарсан: {e}"
        print(error_message)
        logging.error(error_message)

if __name__ == "__main__":
    main()
