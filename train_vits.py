import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig, VitsArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import VitsAudioConfig, Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from formatter import formatter as mbspeech

BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset_config = BaseDatasetConfig(
    dataset_name="mbspeech_mn",
    meta_file_train="metadata.csv",
    path=DATASET_PATH,
    language="mn",
)

audio_config = VitsAudioConfig(sample_rate=16000)

model_args = VitsArgs(
    hidden_channels=256,
    filter_channels=1024,
    n_heads=4,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    n_flows=4,
    gin_channels=0,
)

config = VitsConfig(
    model_args=model_args,
    audio=audio_config,
    datasets=[dataset_config],
    output_path=OUTPUT_PATH,
    run_name="vits_mn_optimized",
    project_name="vits_mn",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    epochs=3000,
    save_step=500,
    log_model_step=200,
    test_delay_epochs=1,
    lr_gen=1e-4,
    lr_disc=1e-4,
    optimizer="AdamW",
    optimizer_params={"betas":[0.8,0.99],"eps":1e-9,"weight_decay":1e-2},
    lr_scheduler_gen_params={"gamma":0.9999},
    lr_scheduler_disc_params={"gamma":0.9999},
    scheduler_after_epoch=True,
    mel_loss_alpha=35.0,
    add_blank=False,
    use_phonemes=False,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        characters="абвгдежзийклмнопрстуфхцчшщъыьэюяёүө ",
        punctuations="!\"'(),-.:;?[]{}–—" 
    ),
    text_cleaner="multilingual_cleaners",
    use_speaker_embedding=False,
    num_speakers=1,
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
speaker_manager.set_ids_from_data(train_samples+eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

model = Vits(config, ap, tokenizer, speaker_manager)
ids = tokenizer.text_to_sequence(config.test_sentences[0], config.text_cleaner)
assert max(ids) < model.text_encoder.emb.num_embeddings

trainer = Trainer(
    args=TrainerArgs(),
    config=config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
