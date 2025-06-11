import os
import torch

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from formatter import mbspeech

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset_config = BaseDatasetConfig(
    dataset_name="mbspeech_mn",
    meta_file_train="metadata.csv",
    path=DATASET_PATH,
    language="mn",
    formatter=mbspeech,
)

audio_config = VitsAudioConfig(
    sample_rate=16000,
    fft_size= 512,
    hop_length= 160,
    win_length= 400,
    num_mels= 80,
    mel_fmin= 0,
    mel_fmax= 8000    
)

config = VitsConfig(
    audio=audio_config,
    datasets=[dataset_config],
    output_path=OUTPUT_PATH,
    run_name="vits_mn_run",
    project_name="vits_mn",
    batch_size=48,
    eval_batch_size=24,
    num_loader_workers=12,
    num_eval_loader_workers=6,
    mixed_precision=True,
    epochs=2000,
    run_eval=True,
    print_step=50,
    print_eval=True,
    save_step=1000,
    log_model_step=100,
    test_delay_epochs=0,
    lr_gen=0.0002,
    lr_disc=0.0002,
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_disc="ExponentialLR",
    num_speakers=1,
    use_speaker_embedding=False,
    use_phonemes=False,
    text_cleaner="multilingual_cleaners",
    characters= CharactersConfig(
        characters="абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя",
        punctuations=".,-:;!?()[]{}'\" ",
    ),
    test_sentences=[
        "Сайн байна уу?",
        "Та хэрхэн байна?",
        "Би сайн байна.",
        "Та юу хийж байна вэ?",
        "Бид хамтдаа суралцаж байна.",
    ],
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=mbspeech,
)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()