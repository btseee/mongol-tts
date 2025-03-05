import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import CharactersConfig

# Set the output path to the current directory
output_path = os.path.dirname(os.path.abspath(__file__))

# Define the dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # Overridden by custom formatter
    meta_file_train="metadata.csv",
    path=os.path.join(output_path, "data/")
)

# Define a custom formatter for the Mongolian dataset
def mongolian_formatter(root_path, meta_file, **kwargs):
    """Custom formatter for a two-column metadata file: audio_file|transcription"""
    items = []
    meta_file_path = os.path.join(root_path, meta_file)
    speaker_name = "mongolian_speaker"
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('|')
            if len(cols) != 2:
                raise ValueError(f"Expected 2 columns, got {len(cols)} in line: {line}")
            audio_file = os.path.join(root_path, cols[0])  # e.g., data/audio_segments/segment_1.wav
            text = cols[1]  # Transcription in Mongolian
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
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="mn",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters=CharactersConfig(
        pad="_",
        eos="~",
        bos="^",
        characters=mongolian_characters,
        punctuations=mongolian_punctuation
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

# Debug: Print sample information
print(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")
for i, sample in enumerate(train_samples[:5]):  # Check first 5 samples
    print(f"Sample {i}:")
    print(f"  Audio File: {sample['audio_file']}")
    print(f"  Text: {sample['text']}")
    print(f"  Speaker: {sample['speaker_name']}")
    print(f"  Exists: {os.path.exists(sample['audio_file'])}")
    try:
        audio = ap.load_wav(sample['audio_file'])
        print(f"  Audio Length: {len(audio)} samples")
    except Exception as e:
        print(f"  Error loading audio: {e}")

# Initialize the GlowTTS model
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# Initialize the trainer
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# Start training
trainer.fit()