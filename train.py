import os
import logging
import torch

from TTS.TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
from TTS.TTS.tts.configs.vits_config import VitsConfig
from TTS.TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
from utils.get_latest_run import get_latest_model
from utils.get_batch_size import get_auto_batch_size

def setup_logging():
    """Configure the logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def main():
    """Main function to set up and start the training process."""
    setup_logging()

    # Define base paths for organization
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(BASE_PATH, "models", "mongol-tts")
    DATASET_PATH = os.path.join(BASE_PATH, "dataset")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Define Mongolian Cyrillic character set and punctuation
    mongolian_cyrillic = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    punctuations = "!'(),-.:;? “”\"…"
    characters = mongolian_cyrillic + punctuations

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Configure character set for the model
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

    # Audio configuration matching the dataset
    audio_config = VitsAudioConfig(
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    # Vits model configuration
    config = VitsConfig(
        output_path=OUTPUT_PATH,
        run_name="mongolian_tts_run",
        audio=audio_config,
        characters=characters_config,
        batch_size=get_auto_batch_size(),
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        epochs=1000,
        test_delay_epochs=100,
        use_phonemes=False,
        text_cleaner="multilingual_cleaners",
        print_step=50,
        save_step=500,
        log_model_step=100,
        mixed_precision=True,
    )

    #DATASET CONFIG
    dataset_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

    dataset_configs = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name=f"mongolian_tts_{folder}",
            meta_file_train="metadata.csv",
            meta_file_val="",
            path=os.path.join(DATASET_PATH, folder),
            language="mn",
        )
        for folder in dataset_folders
    ]
    
    
    # Load training and evaluation samples with an evaluation split
    train_samples, eval_samples = load_tts_samples(
        dataset_configs,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.1,
    )

    # Initialize the Vits model using the provided configuration
    model = Vits.init_from_config(config)

    # Set up the trainer with the given configurations and samples
    trainer = Trainer(
        TrainerArgs(restore_path=get_latest_model(OUTPUT_PATH)),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    trainer.fit()

if __name__ == '__main__':
    main()
