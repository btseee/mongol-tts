#!/usr/bin/env python3
import os
import torch
from TTS.utils.synthesizer import Synthesizer

TTS_CONFIG = "output/fastspeech2_mn_run/config.json"
TTS_CHECKPOINT = "output/fastspeech2_mn_run/best_model.pth"
VOCODER_CONFIG = "output/hifigan_mn_run/config.json"
VOCODER_CHECKPOINT = "output/hifigan_mn_run/best_model.pth"
TEXT = "Сайн байна уу? Энэ бол туршилтын үг юм."
SPEAKER_ID = "male_masculine"  # Use speaker name or ID as defined in your dataset 
OUTPUT_PATH = "output/test.wav"
USE_CUDA = True

device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

synth = Synthesizer(
    tts_checkpoint=TTS_CHECKPOINT,
    tts_config_path=TTS_CONFIG,
    use_cuda=(device == "cuda"),
)

wav = synth.tts(TEXT, speaker_name=SPEAKER_ID, language=None)
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
synth.save_wav(wav, OUTPUT_PATH)

print(f"Saved: {OUTPUT_PATH}")
