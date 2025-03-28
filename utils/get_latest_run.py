import os
import logging
import re

def get_latest_model(output_path):
    """
    Сүүлийн үеийн загварыг тодорхойлж, түүний замыг буцаана.
    
    Аргументууд:
        output_path (str): Загвар хадгалагдаж буй хавтасны зам.
    
    Буцаах:
        str: Сүүлийн загварын зам эсвэл None (хэрэв загвар олдсонгүй бол).
    """
    # Хэрэв өгөгдсөн зам байхгүй бол None буцаана
    if not os.path.exists(output_path):
        return None

    # Хавтаснуудыг жагсаах (бүрдүүлсэн файлуудыг тусад нь авч үзэх)
    run_folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    
    # Хэрэв ямар ч хавтас байхгүй бол None буцаана
    if not run_folders:
        return None

    # Сүүлийн үеийн "run" хавтасыг олж авна
    latest_run = max(run_folders, key=lambda f: os.path.getctime(os.path.join(output_path, f)))
    
    # Сүүлийн загварын файлын зам
    model_path = os.path.join(output_path, latest_run, "best_model.pth")

    # Хэрэв загварын файл байхгүй бол лог үүсгэн None буцаана
    if not os.path.isfile(model_path):
        logging.warning("Файл олдсонгүй: %s", model_path)
        return None

    # Сүүлийн загварын замыг буцаана
    return model_path

def get_latest_checkpoint(output_path):
    """
    Find and return the path to the latest checkpoint file in the specified directory.

    Args:
        output_path (str): Path to the directory containing checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint file, or None if no valid checkpoint is found.
    """
    if not os.path.exists(output_path):
        return None

    # Regular expression to match checkpoint files with numerical suffixes
    checkpoint_pattern = re.compile(r"^checkpoint_(\d+)\.pth$")

    latest_checkpoint = None
    max_number = -1

    # Iterate over files in the directory
    for filename in os.listdir(output_path):
        match = checkpoint_pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                latest_checkpoint = filename

    if latest_checkpoint is None:
        return None

    return os.path.join(output_path, latest_checkpoint)

def get_latest_path(output_path):
    """
    Сүүлийн үеийн сургалтын хавтасын замыг тодорхойлж, буцаана.
    
    Аргументууд:
        output_path (str): Загвар хадгалагдаж буй хавтасны зам.
    
    Буцаах:
        str: Сүүлийн сургалтын хавтасын зам эсвэл None (хэрэв ямар ч хавтас байхгүй бол).
    """
    # Хэрэв өгөгдсөн зам байхгүй бол None буцаана
    if not os.path.exists(output_path):
        return None

    # Хавтаснуудыг жагсаах (бүрдүүлсэн файлуудыг тусад нь авч үзэх)
    run_folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    
    # Хэрэв ямар ч хавтас байхгүй бол None буцаана
    if not run_folders:
        return None

    # Сүүлийн үеийн "run" хавтасыг олж авна
    latest_run = max(run_folders, key=lambda f: os.path.getctime(os.path.join(output_path, f)))
    
    # Сүүлийн сургалтын хавтасын замыг буцаана
    return os.path.join(output_path, latest_run)

# Туршилтын код
if __name__ == "__main__":
    # "models" болон "mongol-tts" хавтасны замыг тодорхойлох
    output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_path, "models", "mongol-tts")
    
    # Сүүлийн загварыг олох
    latest_model = get_latest_model(output_path)
    
    # Сүүлийн сургалтын хавтасыг олох
    latest_path = get_latest_path(output_path)

    # Үр дүнг хэвлэх
    print("Сүүлийн үеийн модел:", latest_model)
    print("Сүүлийн сургалтын хуудас:", latest_path)
