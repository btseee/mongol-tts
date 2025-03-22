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
        logging.warning("Best model file does not exist: %s", model_path)
        return None

    return model_path