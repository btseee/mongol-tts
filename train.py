import os
import argparse
import logging
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
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
    
    config = GlowTTSConfig(
        run_name="mongol-tts",
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        print_step=25,
        print_eval=False,
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
    
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    
    trainer = Trainer(
        TrainerArgs(), 
        config, 
        output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )
    
    trainer.fit()
    
    
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
        default="/workspace"
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