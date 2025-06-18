from TTS.api import TTS

tts = TTS(
    model_path="output/vits_mn_run-June-15-2025_12+02PM-14b9d9c/best_model.pth",
    config_path="output/vits_mn_run-June-15-2025_12+02PM-14b9d9c/config.json",
    progress_bar=True,
)
tts.to("cuda")
tts.tts_to_file(
    text="Сайн байна уу танд ямар тусламж хэрэгтэй вэ",
    file_path="output/test.wav"
)