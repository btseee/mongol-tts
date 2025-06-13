import os
import torch

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

from formatter import formatter as mbspeech

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset_config = BaseDatasetConfig(
    dataset_name="mbspeech_mn",
    meta_file_train="metadata.csv",
    path=DATASET_PATH,
    language="mn",
)

audio_config = VitsAudioConfig(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    num_mels=80,
    mel_fmin=0.0,
    mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    datasets=[dataset_config],
    output_path=OUTPUT_PATH,
    run_name="vits_mn_run_optimized_v1",
    project_name="vits_mn",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=16,
    num_eval_loader_workers=8,
    mixed_precision=True,
    epochs=2000,
    lr_gen=1e-4,
    lr_disc=1e-4,
    lr_scheduler_gen="StepLR",
    lr_scheduler_disc="StepLR",
    lr_scheduler_params={"gamma": 0.999, "step_size": 1000},
    grad_clip=[1000.0, 1000.0],
    batch_group_size=48,
    run_eval=True,
    print_step=50,
    print_eval=True,
    save_step=1000,
    log_model_step=100,
    test_delay_epochs=5,
    use_phonemes=False,
    text_cleaner="multilingual_cleaners",
    use_speaker_embedding=False,
    compute_f0=True,
    compute_energy=True,
    characters = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="абвгдежзийклмнопрстуфхцчшщъыьэюяёүө ",
        punctuations="!\"'(),-.:;?[]{}–—"
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
    [dataset_config],
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=mbspeech,
)

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

model = Vits(config, ap, tokenizer, speaker_manager)

trainer = Trainer(
    args=TrainerArgs(),
    config=config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
