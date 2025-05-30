import os
import torch
import logging

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig, BaseAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.models.forward_tts import ForwardTTS

from src.dataset import prepare_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    batch_size = 32
    eval_batch_size = 16
    learning_rate = 1e-3
else:
    batch_size = 8
    eval_batch_size = 4
    learning_rate = 5e-4
    
logging.info("Using device: %s", device)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH , "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

prepare_dataset(DATASET_PATH)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", 
    meta_file_train="train.csv",
    meta_file_val="test.csv",
    path=DATASET_PATH,
    language="mn",
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    pitch_fmax= 1100,
    pitch_fmin= 50
)

characters_config = CharactersConfig(
    characters="абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя ",
    punctuations=".,-:;!?()[]{}'\"",
)

config = Fastspeech2Config(
    run_name="fastspeech2_mn_run",
    project_name="fastspeech2_mn",
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=50,
    print_eval=False,
    save_step=500,
    log_model_step=100,
    use_phonemes=False,
    text_cleaner="multilingual_cleaners",
    characters=characters_config,
    mixed_precision=True,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    lr=learning_rate,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    datasets=[dataset_config],
    output_path=OUTPUT_PATH,
    f0_cache_path=os.path.join(OUTPUT_PATH, "f0_cache"),
    energy_cache_path=os.path.join(OUTPUT_PATH, "energy_cache"),  
    audio=audio_config, 
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size
)

speaker_manager = SpeakerManager(use_cuda=(device.type=="cuda"))

model = ForwardTTS(
    config=config, 
    ap=ap, 
    tokenizer=tokenizer, 
    speaker_manager=speaker_manager
)

trainer = Trainer(
    TrainerArgs(), config, OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

trainer.fit()