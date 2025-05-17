import os
import math

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets.dataset import F0Dataset, EnergyDataset

from utils.formatter import common_voices_mn

import torch
torch.cuda.empty_cache()

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, "output")
dataset_path = os.path.join(base_path, "dataset", "commonvoice")
phoneme_cache_path = os.path.join(output_path, "phoneme_cache")
f0_cache_path = os.path.join(output_path, "f0_cache")
energy_cache_path = os.path.join(output_path, "energy_cache")

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)

dataset_config = BaseDatasetConfig(
    meta_file_train="metadata.csv",
    path=dataset_path
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=30,
    win_length=1024,
    hop_length=512,
    pitch_fmin=50.0,
    pitch_fmax=500.0,
)

config = Fastspeech2Config(
    run_name="fastspeech2_mn",
    project_name="fastspeech2_mn",
    
    epochs=100,
    test_delay_epochs=10,
    print_step=100,
    run_eval=True,
    print_eval=False,
    
    batch_size=48,
    eval_batch_size=25,
    num_loader_workers=6,
    num_eval_loader_workers=4,
    
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    characters = CharactersConfig(
        characters="абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя ",
        punctuations=".,-:;!?()[]{}'\"",
    ),
    audio=audio_config,
    max_audio_len=math.ceil(audio_config.sample_rate * 25),
    max_text_len=200,
    
    mixed_precision=True,
    grad_clip=1.0,
    
    num_speakers=510,
    use_speaker_embedding=True,
    speakers_file=os.path.join(output_path, "speakers_list.txt"),
    
    lr=1e-3,
    optimizer="Adam",
    optimizer_params={"betas": [0.9, 0.98], "weight_decay": 1e-6},
    lr_scheduler="NoamLR",
    lr_scheduler_params={"warmup_steps": 8000},
    
    spec_loss_alpha=1.0,
    duration_loss_type="mse",
    pitch_loss_alpha=0.1,
    energy_loss_alpha=0.1,
    
    phoneme_cache_path=phoneme_cache_path,
    f0_cache_path=f0_cache_path,
    energy_cache_path=energy_cache_path,
    
    output_path=output_path,
    datasets=[dataset_config],
    
    test_sentences=[
        "Сайн байна уу?",
        "Монгол хэл бол гайхамшигтай.",
        "Өнөөдөр цаг агаар сайхан байна.",
        "Би ном унших дуртай.",
        "Таны нэр хэн бэ?",
        "Бид хамтдаа ажиллах болно.",
        "Энэ бол миний гэр бүл.",
        "Та хаанаас ирсэн бэ?",
        "Би кофе уухыг хүсч байна.",
        "Амжилт хүсье!"
    ],
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=common_voices_mn,
)

f0_ds = F0Dataset(
    samples=train_samples + eval_samples,
    ap=ap,
    cache_path=f0_cache_path,
    precompute_num_workers=4,
)

energy_ds = EnergyDataset(
    samples=train_samples + eval_samples,
    ap=ap,
    cache_path=energy_cache_path,
    precompute_num_workers=4,
)

model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
