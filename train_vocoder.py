import torch
from pathlib import Path
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.bin.extract_tts_spectrograms import setup_loader, extract_spectrograms
from utils.formatter import common_voices_mn

# === USER SETTINGS (no CLI args) ===
CONFIG_PATH      = "output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/config.json"
CHECKPOINT_PATH  = "output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/best_model.pth"
DATASET_PATH     = "dataset/commonvoice"
OUTPUT_PATH      = "output/vocoder_data"
USE_CUDA         = torch.cuda.is_available()
EVAL_SPLIT       = True

# 1) Load TTS config
config = load_config(CONFIG_PATH)
config.audio.trim_silence = False
ap = AudioProcessor(**config.audio)

# 2) Load metadata with custom formatter
train_meta, eval_meta = load_tts_samples(
    datasets=config.datasets,
    eval_split=EVAL_SPLIT,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=common_voices_mn,
)
samples = train_meta + eval_meta

# 3) Speaker manager
if config.use_speaker_embedding:
    speaker_manager = SpeakerManager(data_items=samples)
else:
    speaker_manager = None

# 4) DataLoader for mel extraction
r = 1 if config.model.lower() == "glow_tts" else None
loader = setup_loader(
    config=config,
    ap=ap,
    r=r,
    speaker_manager=speaker_manager,
    samples=samples,
)

# 5) Build and load model
from TTS.tts.models import setup_model
model = setup_model(config)
model.load_checkpoint(config, CHECKPOINT_PATH, eval=True)
if USE_CUDA:
    model.cuda()

# 6) Extract spectrograms
extract_spectrograms(
    model_name=config.model.lower(),
    data_loader=loader,
    model=model,
    ap=ap,
    output_path=Path(OUTPUT_PATH),
    quantize_bits=0,
    save_audio=False,
    debug=False,
    metadata_name="metadata.txt",
)

print("Mel extraction complete. Files in:", OUTPUT_PATH)
