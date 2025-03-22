import os
import logging

def get_latest_model(output_path):
    if not os.path.exists(output_path):
        return None

    run_folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    if not run_folders:
        return None

    latest_run = max(run_folders, key=lambda f: os.path.getctime(os.path.join(output_path, f)))
    model_path = os.path.join(output_path, latest_run, "best_model.pth")

    if not os.path.isfile(model_path):
        logging.warning("Файл олдсонгүй: %s", model_path)
        return None

    return model_path

def get_latest_path(output_path):
    if not os.path.exists(output_path):
        return None

    run_folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
    if not run_folders:
        return None

    latest_run = max(run_folders, key=lambda f: os.path.getctime(os.path.join(output_path, f)))
    return os.path.join(output_path, latest_run)

# Туршилтын код
if __name__ == "__main__":
    output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_path, "models", "mongol-tts")
    
    latest_model = get_latest_model(output_path)
    latest_path = get_latest_path(output_path)

    print("Сүүлийн үеийн модел:", latest_model)
    print("Сүүлийн сургалтын хуудас:", latest_path)