import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import CharactersConfig, BaseAudioConfig

# Set the output path to the current directory
output_path = os.path.dirname(os.path.abspath(__file__))

# Define the dataset configuration
dataset_config = BaseDatasetConfig(
    meta_file_train="metadata.csv",
    path=os.path.join(output_path, "data/")
)

# Define a custom formatter for the Mongolian dataset
def mongolian_formatter(root_path, meta_file, **kwargs):
    items = []
    meta_file_path = os.path.join(root_path, meta_file)
    speaker_name = "mongolian_speaker"
    with open(meta_file_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            if len(cols) != 2:
                raise ValueError(f"Expected 2 columns, got {len(cols)} in line: {line}")
            audio_file = os.path.join(root_path, cols[0])  # Full path to audio
            text = cols[1].strip()  # Clean text
            if not os.path.exists(audio_file):
                print(f"Warning: {audio_file} not found, skipping")
                continue
            items.append({
                "audio_file": audio_file,
                "text": text,
                "speaker_name": speaker_name,
                "root_path": root_path
            })
    return items

# Define the Mongolian character set (Cyrillic alphabet + common punctuation)
mongolian_characters = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
mongolian_punctuation = " .,!?-;()“”"

# Initialize the training configuration with custom characters
config = GlowTTSConfig(
    batch_size=16,  # Reduced for stability on smaller systems
    eval_batch_size=8,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=10,  # Start evaluation after 10 epochs
    epochs=1000,
    text_cleaner="basic_cleaners",  # Simple cleaner since no phonemes
    use_phonemes=False,  # Mongolian phoneme support is not standard
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,  # Show evaluation progress
    mixed_precision=True,  # Faster training on GPU
    output_path=os.path.join(output_path, "output"),
    datasets=[dataset_config],
    characters=CharactersConfig(
        pad="_",
        eos="~",
        bos="^",
        characters=mongolian_characters,
        punctuations=mongolian_punctuation
    ),
    audio=BaseAudioConfig( 
        sample_rate= 22050,
        win_length= 1024,
        hop_length= 256,
        num_mels= 80,
        mel_fmin= 0.0,
        mel_fmax= 8000.0,
    )
)

# Initialize the audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize the tokenizer with the updated config
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load training and evaluation samples using the custom formatter
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=mongolian_formatter,
    eval_split=True,
    eval_split_size=0.1,
)

# Initialize the GlowTTS model
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# Initialize the trainer
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path=os.path.join(output_path, "output"),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# Start training
trainer.fit()