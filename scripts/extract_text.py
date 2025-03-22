import os
import re
from PyPDF2 import PdfReader
import argparse

def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    PDF файлаас тодорхой хуудаснуудын текстийг гаргаж авна.
    
    Аргументууд:
        pdf_path (str): PDF файлын хаяг/зам.
        start_page (int): Эхлэх хуудасны дугаар (0-ээс эхэлнэ).
        end_page (int): Дуусах хуудасны дугаар (0-ээс эхэлнэ).
    
    Буцаах:
        str: Гаргаж авсан текст, хуудас тус бүрээр мөрөөр тусгаарлагдана.
    
    Алдаа:
        FileNotFoundError: Хэрэв PDF файл олдволгүй бол алдаа гарна.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл олдсонгүй: {pdf_path}")
    
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if start_page < 0 or end_page >= total_pages or start_page > end_page:
            raise ValueError(f"Хуудасны хязгаар буруу: 0-{total_pages-1} хооронд байх ёстой")
        
        text = "\n".join(reader.pages[i].extract_text() or "" for i in range(start_page, end_page + 1))
    return text

def clean_text(text):
    """
    Текстийг цэвэрлэж, илүүдэл зай, мөрүүдийг арилгана.
    
    Аргументууд:
        text (str): Цэвэрлэх текст.
    
    Буцаах:
        str: Цэвэрлэсэн, нэг мөр болгосон текст.
    """
    text = re.sub(r'[\n\f]+', ' ', text)  # Шинэ мөрүүдийг зай болгох
    text = re.sub(r'\s+', ' ', text).strip()  # Илүүдэл зай арилгах
    return text

def preserve_quoted_text(text):
    """
    Ишлэлүүдийг хадгалж, текстийг өгүүлбэр болгон хуваана.
    
    Аргументууд:
        text (str): Хуваах текст.
    
    Буцаах:
        list: Хоосон биш өгүүлбэрүүдийн жагсаалт.
    """
    quoted_pattern = r'“[^”]+”|"[^"]+"'  # Ишлэлийн загвар (буржгар ба энгийн)
    quoted_texts = re.findall(quoted_pattern, text)
    placeholders = [f'<QUOTE{i}>' for i in range(len(quoted_texts))]
    
    # Ишлэлүүдийг placeholders-ээр солих
    for placeholder, quoted_text in zip(placeholders, quoted_texts):
        text = text.replace(quoted_text, placeholder)
    
    # Өгүүлбэр болгон хуваах (.!? дараах зай дээр)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Ишлэлүүдийг буцааж оруулах
    for i, sentence in enumerate(sentences):
        for placeholder, quoted_text in zip(placeholders, quoted_texts):
            sentence = sentence.replace(placeholder, quoted_text)
        sentences[i] = sentence.strip()
    
    # Хоосон өгүүлбэрүүдийг шүүх
    return [s for s in sentences if s]

def save_sentences_to_file(sentences, output_path):
    """
    Өгүүлбэрүүдийг текст файлд мөр бүрт нэгээр хадгална.
    
    Аргументууд:
        sentences (list): Хадгалах өгүүлбэрүүдийн жагсаалт.
        output_path (str): Гаралтын файлын хаяг/зам.
    
    Алдаа:
        IOError: Хэрэв файл бичихэд алдаа гарвал.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Гаралтын хавтас үүсгэх
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(f"{s}\n")

def main():
    """
    Скриптийг ажиллуулах гол функц. Командын мөрний аргументуудыг уншина.
    """
    parser = argparse.ArgumentParser(description="PDF файлаас текст гаргаж, өгүүлбэр тус бүрээр файлд хадгална.")
    parser.add_argument("pdf_path", help="PDF файлын хаяг/зам")
    parser.add_argument("start_page", type=int, help="Эхлэх хуудасны дугаар (0-ээс эхэлнэ)")
    parser.add_argument("end_page", type=int, help="Дуусах хуудасны дугаар (0-ээс эхэлнэ)")
    parser.add_argument("output_path", help="Гаралтын текст файлын хаяг/зам")
    
    args = parser.parse_args()
    
    try:
        # Текст гаргаж авах
        text = extract_text_from_pdf(args.pdf_path, args.start_page, args.end_page)
        # Текстийг цэвэрлэх
        cleaned_text = clean_text(text)
        # Өгүүлбэр болгон хуваах
        sentences = preserve_quoted_text(cleaned_text)
        # Файлд хадгалах
        save_sentences_to_file(sentences, args.output_path)
        print("Боловсруулалт амжилттай дууслаа. Өгүүлбэрүүд хадгалагдлаа.")
    except Exception as e:
        print(f"Алдаа гарлаа: {e}")

if __name__ == "__main__":
    main()