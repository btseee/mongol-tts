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
from TTS.tts.models.forward_tts import ForwardTTS # Correct import for FastSpeech2 via ForwardTTS

# Assuming dataset.py and formatter.py are in a 'src' directory relative to train.py
# If they are in the same directory as train.py, adjust paths accordingly.
# For this example, let's assume train.py is at the root, and src/dataset.py, src/utils/formatter.py
try:
    from src.dataset import prepare_dataset, ACCEPTED_CHARS # Import ACCEPTED_CHARS for validation
    from src.utils.formatter import common_voice
except ImportError:
    # Fallback if running from a different structure or if src is not a package
    # This might require files to be in the same directory or PYTHONPATH to be set
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
    batch_size = 32  # Consider lowering if OOM occurs
    eval_batch_size = 16 # Consider lowering if OOM occurs
    num_loader_workers = 4
    num_eval_loader_workers = 2
    mixed_precision = True # Can be disabled for debugging
else:
    batch_size = 4 # Reduced for CPU
    eval_batch_size = 2 # Reduced for CPU
    num_loader_workers = 0 # Often better for CPU
    num_eval_loader_workers = 0
    mixed_precision = False

# Define paths
# Assuming train.py is in the root of your project
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "tts_dataset_mongolian") # Renamed for clarity
OUTPUT_PATH = os.path.join(BASE_PATH, "tts_output_mongolian") # Renamed for clarity

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
logging.info(f"Dataset will be stored in: {DATASET_PATH}")
logging.info(f"Training output will be stored in: {OUTPUT_PATH}")

# --- 1. DATASET PREPARATION ---
# This will download and process the dataset if it hasn't been done already.
logging.info("Starting dataset preparation...")
prepare_dataset(DATASET_PATH)
logging.info("Dataset preparation finished.")

# --- 2. DEFINE CONFIGURATIONS ---

# Dataset Configuration
dataset_config = BaseDatasetConfig(
    formatter="common_voice", # Specify the formatter name
    meta_file_train="train.csv",
    meta_file_val="test.csv", # Using 'test.csv' as validation
    path=DATASET_PATH,
    language="mn",
)

# Audio Configuration (Matches TARGET_SR from dataset.py)
audio_config = BaseAudioConfig(
    sample_rate=22050, # from dataset.py
    resample=False, # Resampling is done in dataset.py
    pitch_fmax=600.0, # Adjusted, can be tuned
    pitch_fmin=70.0,  # Adjusted, can be tuned
    # Default values for other params like hop_length, win_length are usually fine
)

# Characters Configuration
# Ensure this exactly matches the characters used in `normalize_text` in dataset.py
# plus any characters you want the model to explicitly know as punctuations.
# The `ACCEPTED_CHARS` from dataset.py includes letters, space, and some punctuation.
# `CharactersConfig` separates `characters` (phonetic alphabet) and `punctuations`.
# The space should be in `characters`.
MONGOLIAN_ALPHABET = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
PUNCTUATIONS_FOR_CONFIG = ".,!?-:" # Punctuation marks from ACCEPTED_CHARS

# Validate that all characters in ACCEPTED_CHARS (from dataset.py) are covered
# This is a sanity check.
temp_accepted_chars_set = set(ACCEPTED_CHARS)
temp_config_chars_set = set(MONGOLIAN_ALPHABET + PUNCTUATIONS_FOR_CONFIG + " ") # Add space
if temp_accepted_chars_set != temp_config_chars_set:
    logging.warning(f"Potential mismatch between ACCEPTED_CHARS in dataset.py and characters/punctuations in config.")
    logging.warning(f"Dataset ACCEPTED_CHARS: {''.join(sorted(list(temp_accepted_chars_set)))}")
    logging.warning(f"Config Chars+Punct:    {''.join(sorted(list(temp_config_chars_set)))}")
    # This might not be an error if dataset.py's ACCEPTED_CHARS is a superset
    # and text normalization correctly filters to what the model expects.
    # However, it's good to keep them closely aligned.

characters_config = CharactersConfig(
    characters=MONGOLIAN_ALPHABET + " ", # Explicitly include space with characters
    punctuations=PUNCTUATIONS_FOR_CONFIG,
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    phonemes=None, # Not using phonemes
)

# Full Training Configuration for FastSpeech2
config = Fastspeech2Config(
    run_name="fastspeech2_mn_cyrillic_v1",
    project_name="mongolian_tts",
    # Training params
    epochs=1000,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    lr=1e-4, # Common starting LR for FastSpeech2, might need AdamW optimizer and scheduler
    # Optimizer and Scheduler (CoquiTTS often handles this internally, but can be specified)
    # e.g. optimizer="AdamW", optimizer_params={"betas": [0.9, 0.98], "eps": 1e-9},
    # lr_scheduler="NoamLR", lr_scheduler_params={"warmup_steps": 4000},
    grad_clip=1.0, # Gradient clipping
    
    # Data loading
    num_loader_workers=num_loader_workers,
    num_eval_loader_workers=num_eval_loader_workers,
    
    # Model architecture
    num_speakers=3, # male, female, unknown
    use_speaker_embedding=True,
    
    # Text processing
    use_phonemes=False, # Using characters
    text_cleaner="basic_cleaners", # Or "multilingual_cleaners" or None if your normalization is sufficient
                                 # If "None", ensure your `normalize_text` is perfect.
                                 # "basic_cleaners" is often safer.
    characters=characters_config,
    
    # Audio processing
    audio=audio_config,
    # Precomputed features paths (ensure OUTPUT_PATH is writable)
    f0_cache_path=os.path.join(OUTPUT_PATH, "cache", "f0"),
    energy_cache_path=os.path.join(OUTPUT_PATH, "cache", "energy"),
    # Ensure these cache paths are distinct if you run multiple experiments in the same OUTPUT_PATH
    # or clear them if you change audio processing parameters.
    
    # Logging and Saving
    output_path=OUTPUT_PATH,
    print_step=50,        # Log training loss every 50 steps
    plot_step=100,        # Generate alignment plots every 100 steps (if applicable)
    log_model_step=500,   # Log model parameters/gradients to TensorBoard (if enabled)
    save_step=1000,       # Save checkpoint every 1000 steps
    save_n_checkpoints=5, # Number of recent checkpoints to keep
    save_best_after=1000, # Start saving the best model after 1000 steps
    
    # Evaluation
    run_eval=True,
    eval_split_size=0.05, # Use 5% of the validation set for quick eval during training
                          # Or set a fixed number like config.eval_split_max_size
    # eval_split_max_size=256, # Max number of samples for eval during training
    test_delay_epochs=-1, # Start evaluation from the first epoch
    print_eval=True,      # Print evaluation results
    
    # Test sentences for synthesis during evaluation
    test_sentences=[
        "Сайн байна уу?",
        "Та хэрхэн байна?",
        "Би сайн байна.",
        "Монгол хэлээр ярьж сурцгаая.",
        "Энэ бол туршилтын өгүүлбэр юм.",
    ],
    
    # Other
    mixed_precision=mixed_precision,
    datasets=[dataset_config], # List of dataset configs
    cudnn_benchmark=False, # Set to True if input sizes are constant, can speed up but uses more memory initially.
                           # For TTS, input sizes vary, so False is often safer.
)

# --- 3. INITIALIZE TOKENIZER AND AUDIO PROCESSOR ---
# These are initialized based on the configuration
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config) # Tokenizer might update the config (e.g., with char mapping)

# --- 4. LOAD DATA SAMPLES ---
# `load_tts_samples` uses the formatter specified in `dataset_config`
logging.info("Loading training and evaluation samples...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True, # Create an evaluation split from the training metafile if meta_file_val is not specific enough
    # If meta_file_val ("test.csv") is intended as the full validation set,
    # you might not need eval_split=True here, or ensure it uses the test data.
    # CoquiTTS logic: if meta_file_val is present, it's used. `eval_split` then refers to splitting *that* val set further if needed.
    # For clarity, often people use distinct train.csv, val.csv (dev), test.csv.
    # Here, your test.csv is acting as the validation set.
    formatter=common_voice, # Pass the actual function
)

if not train_samples:
    logging.error("No training samples loaded. Check dataset paths, meta files, and formatter.")
    exit()
if not eval_samples:
    logging.warning("No evaluation samples loaded. Check dataset paths, meta files, and formatter for validation.")
    # You might choose to exit or continue without run_eval if this is critical

logging.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

# --- 5. INITIALIZE SPEAKER MANAGER (for multi-speaker models) ---
if config.num_speakers > 1 and config.use_speaker_embedding:
    logging.info("Initializing Speaker Manager...")
    speaker_manager = SpeakerManager(speaker_id_file=None, use_cuda=(device.type == "cuda"))
    # Infer speaker IDs from the data
    # Ensure 'speaker_name' in your `common_voice` formatter's output matches this key.
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.num_speakers = speaker_manager.num_speakers # Update config with actual number of found speakers
    logging.info(f"Actual number of unique speakers found: {speaker_manager.num_speakers}")
    if speaker_manager.num_speakers == 0:
        logging.error("Speaker manager found 0 speakers. Check 'speaker_name' in your data and formatter.")
        exit()
    if speaker_manager.num_speakers == 1 and config.use_speaker_embedding:
        logging.warning("Only 1 speaker found, but use_speaker_embedding is True. Consider setting it to False or check data.")
        # config.use_speaker_embedding = False # Optionally disable if only one speaker
else:
    speaker_manager = None
    config.num_speakers = 0 # Or 1 if it's a single speaker model without embeddings
    config.use_speaker_embedding = False

# --- 6. INITIALIZE THE MODEL ---
logging.info("Initializing Fastspeech2 model...")
model = ForwardTTS(config, ap, tokenizer, speaker_manager)

# --- 7. INITIALIZE THE TRAINER ---
logging.info("Initializing Trainer...")
trainer = Trainer(
    TrainerArgs(), # Using default trainer arguments. Can be customized.
    config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    device=device, # Pass the device to the trainer
)

# --- 8. START TRAINING ---
logging.info("Starting training...")
try:
    trainer.fit()
except Exception as e:
    logging.error(f"Training stopped due to an error: {e}", exc_info=True)
    # This will print the full traceback
    # Consider adding more robust error handling or saving state if needed

logging.info("Training finished or stopped.")
