import os
import logging
import torch

from TTS.TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
from TTS.TTS.tts.configs.vits_config import VitsConfig
from TTS.TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
from utils.get_latest_run import get_latest_model
from utils.get_batch_size import get_auto_batch_size, get_auto_num_workers

def setup_logging():
    """Лог үүсгэх"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def main():
    """Үндсэн функц"""
    setup_logging()

    # Тохиргоо
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(BASE_PATH, "models", "mongol-tts")
    DATASET_PATH = os.path.join(BASE_PATH, "dataset")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Үсгийн жагсаалт
    mongolian_cyrillic = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    punctuations = "!'(),-.:;? “”\"…"
    characters = mongolian_cyrillic + punctuations

    # Cuda санах ойг цэвэрлэх
    torch.cuda.empty_cache()

    # Үсгийн тохиргоо
    characters_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters=characters,
        punctuations=punctuations,
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    )

    # Дууны тохиргоо
    audio_config = VitsAudioConfig(
        sample_rate=22050, # 22050Hz
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    # Vits тохиргоо
    config = VitsConfig(
        output_path=OUTPUT_PATH,
        run_name="mongolian_tts_run",
        audio=audio_config,
        characters=characters_config,
        batch_size = get_auto_batch_size(is_training=True, max_batch=64),
        eval_batch_size = get_auto_batch_size(is_training=False, max_batch=128),
        num_loader_workers = get_auto_num_workers(is_training=True),
        num_eval_loader_workers = get_auto_num_workers(is_training=False),
        epochs=1000,
        test_delay_epochs=10,
        use_phonemes=False,
        text_cleaner="multilingual_cleaners",
        print_step=50,
        save_step=500,
        log_model_step=100,
        mixed_precision=True,
        test_sentences=[
            "Сайн байна уу?",   
            "Би монгол хүн.",          
            "Өнөөдөр цаг агаар сайхан байна."
        ],
    )

    # Датасет тохиргоо
    dataset_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    if not dataset_folders:
        logging.error("Датасет алга %s", DATASET_PATH)
        return

    dataset_configs = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="mongolian_tts",
            meta_file_train="metadata.csv",
            meta_file_val="",
            path=os.path.join(DATASET_PATH, folder),
            language="mn",
        )
        for folder in dataset_folders
    ]

    # Сургалтын багцуудыг дуудах
    train_samples, eval_samples = load_tts_samples(
        dataset_configs,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.1
    )

    # Лог хийх
    logging.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

    # Тохиргоо болон моделийг үүсгэх
    model = Vits.init_from_config(config)

    # Trainer үүсгэх
    trainer = Trainer(
        TrainerArgs(
            restore_path=get_latest_model(OUTPUT_PATH),
        ),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Сургалтыг эхлүүлэх
    trainer.fit()

if __name__ == '__main__':
    main()