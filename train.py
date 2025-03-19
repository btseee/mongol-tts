import os
from TTS.TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
from TTS.TTS.tts.configs.vits_config import VitsConfig
from TTS.TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs

# Define base paths for better organization
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, "models", "mongol-tts")
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
AUDIO_PATH = os.path.join(DATASET_PATH, "wavs")

# Define Mongolian Cyrillic character set
mongolian_cyrillic = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
punctuations = "!'(),-.:;? “”\""
characters = mongolian_cyrillic + punctuations

# Configure character set for the model
characters_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="_",
    eos="&",
    bos="*",
    blank=None,
    characters=characters,
    punctuations=punctuations,
    phonemes=None,  # No phonemes since we're using characters directly
    is_unique=True,
    is_sorted=True,
)

# Audio configuration matching the dataset
audio_config = VitsAudioConfig(
    sample_rate=22050,  # Matches your audio files
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Vits model configuration
config = VitsConfig(
    output_path=OUTPUT_PATH,
    run_name="mongolian_tts_run",
    audio=audio_config,
    characters=characters_config,
    batch_size=8,  # Suitable for smaller datasets or limited GPU memory
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    epochs=1000,
    test_delay_epochs=100,
    use_phonemes=False,  # Explicitly character-based
    text_cleaner="multilingual_cleaners",  # Suitable for Mongolian Cyrillic
    print_step=50,  # Print training progress every 50 steps
    save_step=500,  # Save checkpoint every 500 steps
    log_model_step=100,  # Log model details every 100 steps
)

# Dataset configuration
mongolian_config = BaseDatasetConfig(
    formatter="ljspeech",  # Assumes metadata.csv is in wav|transcription format
    dataset_name="mongolian_tts",
    meta_file_train="metadata.csv",
    meta_file_val="",  # Add a validation file if available
    path=DATASET_PATH,
    language="mn",
)

# Load training and evaluation samples
train_samples, eval_samples = load_tts_samples(
    [mongolian_config],
    eval_split=True,  # Automatically split dataset for evaluation
    eval_split_max_size=256,  # Maximum size of evaluation set
    eval_split_size=0.1,  # 10% of dataset for evaluation
)

# Initialize the Vits model
model = Vits.init_from_config(config)

# Set up and run the trainer
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()