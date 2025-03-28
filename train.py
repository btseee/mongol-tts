import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from utils.clean_cache import clean_gpu_cache
from utils.get_batch_size import get_auto_batch_size, get_auto_num_workers
from utils.get_latest_run import get_latest_checkpoint, get_latest_model
import argparse

def train(dataset_path: str, output_path: str, restore_path: str, best_path: str):
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata_train.csv", 
        meta_file_val="metadata_test.csv",
        path=dataset_path,
        language="mn-mn"
    )
    
    config = GlowTTSConfig(
        batch_size=get_auto_batch_size(is_training=True),
        eval_batch_size=get_auto_batch_size(is_training=False),
        num_loader_workers=get_auto_num_workers(is_training=True),
        num_eval_loader_workers=get_auto_num_workers(is_training=False),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="mn-cyrl", 
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
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
    
    model = GlowTTS(
        config, 
        ap, 
        tokenizer, 
        speaker_manager=None
    )
    
    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            best_path=best_path
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