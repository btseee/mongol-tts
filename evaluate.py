from TTS.TTS.utils.synthesizer import Synthesizer
import os

# Load the synthesizer with a model checkpoint
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, "models", "mongol-tts",'mongolian_tts_run-March-17-2025_07+08PM-0000000')

synthesizer = Synthesizer(
    model_dir=OUTPUT_PATH,
    tts_checkpoint=os.path.join(OUTPUT_PATH, "checkpoint_19500.pth"),
    tts_config_path=os.path.join(OUTPUT_PATH, "config.json"),
    use_cuda=False ,    
)

wav = synthesizer.tts("Сайн уу, энэ бол миний TTS загвар.")

# Save the audio file
synthesizer.save_wav(wav, "output/output.wav")
