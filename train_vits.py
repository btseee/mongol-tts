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

# Assuming your custom formatter is in a file named formatter.py
from formatter import formatter as mbspeech

# --- PATHS ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- CONFIGURATIONS ---

dataset_config = BaseDatasetConfig(
    dataset_name="mbspeech_mn",
    meta_file_train="metadata.csv",
    path=DATASET_PATH,
    language="mn",
)

# Use the standard 22050Hz sample rate for best model performance.
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
    run_name="vits_mn_run_fixed",
    project_name="vits_mn",
    
    # Using a slightly more conservative batch size to start, can be increased later.
    batch_size=48,
    eval_batch_size=24,
    
    num_loader_workers=16,
    num_eval_loader_workers=8,
    
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
    
    # FIX 2: Use a more robust text cleaner.
    use_phonemes=False,
    text_cleaner="multilingual_cleaners",
    
    use_speaker_embedding=False,
    
    compute_f0=True,
    compute_energy=True,
    
    # Pass the auto-generated characters config to the model
    characters = CharactersConfig(
        characters="абвгдежзийклмнопрстуфхцчшщъыьэюяёүө",
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

# --- INITIALIZATION ---

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    datasets_config=dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=mbspeech,
)

# ===================================================================
# FIX 3: Reinstate SpeakerManager for robust speaker handling.
# This correctly sets config.num_speakers from your data.
# ===================================================================
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

# --- TRAINING ---
trainer.fit()

