import os

from trainer import Trainer, TrainerArgs
import torch
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from utils.clean_cache import clean_gpu_cache
from utils.get_latest_run import get_latest_checkpoint, get_latest_model
import argparse

def train(dataset_path: str, output_path: str, restore_path: str, best_path: str):
    torch.set_num_threads(16)
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata_train.csv", 
        meta_file_val="metadata_test.csv",
        path=dataset_path,
        language="mn-cyrl",     
    )
    
    config = Tacotron2Config(
        batch_size=128,
        eval_batch_size=64,
        num_loader_workers=16,
        num_eval_loader_workers=16,
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,
        text_cleaner="basic_cleaners",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(
        config, 
        ap, 
        tokenizer, 
        speaker_manager=None
    ).to(device=device)
    
    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            best_path=best_path,
        ), 
        config, 
        output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )
    
    trainer.fit()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", default="models/mongol-tts")
    args = parser.parse_args()
    directory = args.output_path
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    restore_path = get_latest_checkpoint(directory)    
    best_path = get_latest_model(directory)
    
    clean_gpu_cache()
    train("dataset", directory, restore_path, best_path)