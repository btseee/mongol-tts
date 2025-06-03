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

try:
    from src.dataset import prepare_dataset, ACCEPTED_CHARS 
    from src.utils.formatter import common_voice
except ImportError:
    logging.warning("Could not import from src. Ensure your project structure and PYTHONPATH are correct.")
    logging.warning("Attempting to import assuming files are in the same directory or accessible.")
    from dataset import prepare_dataset, ACCEPTED_CHARS
    from src.utils.formatter import common_voice


# Configure logging for the training script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Determine device and set parameters accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

if device.type == "cuda":
    batch_size = 32  
    eval_batch_size = 16
    num_loader_workers = 4
    num_eval_loader_workers = 2
    mixed_precision = True 
else:
    batch_size = 4 
    eval_batch_size = 2 
    num_loader_workers = 0
    num_eval_loader_workers = 0
    mixed_precision = False

# Define paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
OUTPUT_PATH = os.path.join(BASE_PATH, "output")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
logging.info(f"Dataset will be stored in: {DATASET_PATH}")
logging.info(f"Training output will be stored in: {OUTPUT_PATH}")

# --- 1. DATASET PREPARATION ---
logging.info("Starting dataset preparation...")
prepare_dataset(DATASET_PATH)
logging.info("Dataset preparation finished.")

# --- 2. DEFINE CONFIGURATIONS ---
dataset_config = BaseDatasetConfig(
    formatter="common_voice",
    meta_file_train="train.csv",
    meta_file_val="test.csv",
    path=DATASET_PATH,
    language="mn",
)

# Audio Configuration (Matches TARGET_SR from dataset.py)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=False,
    pitch_fmax=600.0, 
    pitch_fmin=70.0, 
)

# Characters Configuration
MONGOLIAN_ALPHABET = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
PUNCTUATIONS_FOR_CONFIG = ".,!?-:" 

temp_accepted_chars_set = set(ACCEPTED_CHARS)
temp_config_chars_set = set(MONGOLIAN_ALPHABET + PUNCTUATIONS_FOR_CONFIG + " ")
if temp_accepted_chars_set != temp_config_chars_set:
    logging.warning(f"Potential mismatch between ACCEPTED_CHARS in dataset.py and characters/punctuations in config.")
    logging.warning(f"Dataset ACCEPTED_CHARS: {''.join(sorted(list(temp_accepted_chars_set)))}")
    logging.warning(f"Config Chars+Punct:    {''.join(sorted(list(temp_config_chars_set)))}")

characters_config = CharactersConfig(
    characters=MONGOLIAN_ALPHABET + " ",
    punctuations=PUNCTUATIONS_FOR_CONFIG,
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    phonemes=None,
)

# Full Training Configuration for FastSpeech2
config = Fastspeech2Config(
    run_name="fastspeech2_mn",
    project_name="mongolian_tts",
    # Training params
    epochs=1000,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    lr=1e-4,
    grad_clip=1.0,
    
    # Data loading
    num_loader_workers=num_loader_workers,
    num_eval_loader_workers=num_eval_loader_workers,
    
    # Model architecture
    num_speakers=3, # male, female, unknown
    use_speaker_embedding=True,
    
    # Text processing
    use_phonemes=False,
    text_cleaner="basic_cleaners",
    characters=characters_config,
    
    # Audio processing
    audio=audio_config,
    f0_cache_path=os.path.join(OUTPUT_PATH, "cache", "f0"),
    energy_cache_path=os.path.join(OUTPUT_PATH, "cache", "energy"),
    
    # Logging and Saving
    output_path=OUTPUT_PATH,
    print_step=50,    
    plot_step=100,      
    log_model_step=500,   
    save_step=1000,       
    save_n_checkpoints=5, 
    save_best_after=1000, 
    
    # Evaluation
    run_eval=True,
    eval_split_size=0.05,
    test_delay_epochs=-1,
    print_eval=True,    
    
    # Test sentences for synthesis during evaluation
    test_sentences=[
        "Сайн байна уу?",
        "Та хэрхэн байна?",
        "Би сайн байна.",
        "Монгол хэлээр ярьж сурцгаая.",
        "Энэ бол туршилтын өгүүлбэр юм.",
    ],
    
    mixed_precision=mixed_precision,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

# --- 3. INITIALIZE TOKENIZER AND AUDIO PROCESSOR ---
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

# --- 4. LOAD DATA SAMPLES ---
logging.info("Loading training and evaluation samples...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=common_voice,
)

if not train_samples:
    logging.error("No training samples loaded. Check dataset paths, meta files, and formatter.")
    exit()
if not eval_samples:
    logging.warning("No evaluation samples loaded. Check dataset paths, meta files, and formatter for validation.")

logging.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

# --- 5. INITIALIZE SPEAKER MANAGER (for multi-speaker models) ---
if config.num_speakers > 1 and config.use_speaker_embedding:
    logging.info("Initializing Speaker Manager...")
    speaker_manager = SpeakerManager(use_cuda=(device.type == "cuda"), speaker_id_file_path=None)
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.num_speakers = speaker_manager.num_speakers 
    logging.info(f"Actual number of unique speakers found: {speaker_manager.num_speakers}")
    if speaker_manager.num_speakers == 0:
        logging.error("Speaker manager found 0 speakers. Check 'speaker_name' in your data and formatter.")
        exit()
    if speaker_manager.num_speakers == 1 and config.use_speaker_embedding:
        logging.warning("Only 1 speaker found, but use_speaker_embedding is True. Consider setting it to False or check data.")
else:
    speaker_manager = None
    config.num_speakers = 0
    config.use_speaker_embedding = False

# --- 6. INITIALIZE THE MODEL ---
logging.info("Initializing Fastspeech2 model...")
model = ForwardTTS(config, ap, tokenizer, speaker_manager)

# --- 7. INITIALIZE THE TRAINER ---
logging.info("Initializing Trainer...")
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# --- 8. START TRAINING ---
logging.info("Starting training...")
try:
    trainer.fit()
except Exception as e:
    logging.error(f"Training stopped due to an error: {e}", exc_info=True)

logging.info("Training finished or stopped.")
