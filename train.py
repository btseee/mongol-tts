import os
import argparse
import logging
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from utils.clean_cache import clean_gpu_cache
from utils.get_latest_run import get_latest_checkpoint, get_latest_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(dataset_path: str, output_path: str, restore_path: str, best_path: str, epochs: int):
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata_train.csv", 
        meta_file_val="metadata_test.csv",
        path=dataset_path,
        language="mn-cyrl",     
    )
    
    audio_config = VitsAudioConfig(
        sample_rate=22050, 
        win_length=1024, 
        hop_length=256, 
        num_mels=80, 
        mel_fmin=0, 
        mel_fmax=None,
    )
    
    config = VitsConfig(
        audio=audio_config,
        run_name="mongol-tts",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=epochs,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        characters=CharactersConfig(
            characters="АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя",
            punctuations=" !\"'(),-.:;?[]{}«»“”‘’",
        ),
        test_sentences=[
            "Сайн байна уу?",   
            "Би монгол хүн.",          
            "Өнөөдөр цаг агаар сайхан байна."
        ],
    )

    ap = AudioProcessor.init_from_config(config)
    
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    logger.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")
    
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    trainer = Trainer(
        TrainerArgs(restore_path=restore_path, best_path=best_path),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TTS model for Mongolian Cyrillic language using CoquiTTS."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to the dataset directory",
        default="dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output directory for the model",
        default="models/mongol-tts"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Number of training epochs",
        default=1000
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output
    epochs = args.epochs

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Created output directory at {output_path}")

    restore_path = get_latest_checkpoint(output_path)
    best_path = get_latest_model(output_path)

    clean_gpu_cache()

    train(dataset_path, output_path, restore_path, best_path, epochs)