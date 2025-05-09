

import os
import math

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.fastspeech2_config import Fastspeech2Config

from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig, CharactersConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

from utils.formatter import common_voices_mn

import torch
torch.cuda.empty_cache()

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, "output")
dataset_path = os.path.join(base_path, "dataset", "commonvoice")

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
    epochs=1000,
    num_speakers=510,
    audio=audio_config,
    batch_size=6,
    eval_batch_size=3,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    text_cleaner="basic_cleaners",
    characters=CharactersConfig(
        characters="абвгґдеєжзийклмнопрстуфхцчшщьъыьэюя",
        punctuations="!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        pad="[PAD]",
        eos="[EOS]",
        bos="[BOS]",
    ),
    output_path=output_path,
    datasets=[dataset_config],
    mixed_precision=False,
    print_step=50,
    print_eval=False,
    run_eval=True,
    test_delay_epochs=-1,
    use_speaker_embedding=True,
    max_audio_len=math.ceil(audio_config.sample_rate * 20.016009),
    min_audio_len=0,
    min_text_len=0,
    max_text_len=200,
    use_phonemes=False,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    f0_cache_path=os.path.join(output_path, "f0_cache"),
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

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

trainer = Trainer(
    TrainerArgs(), 
    config, 
    output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

trainer.fit()