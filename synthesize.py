from TTS.api import TTS

tts = TTS(
    model_path="output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/best_model.pth",
    config_path="output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/config.json",
    vocoder_path="output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/best_model.pth",
    vocoder_config_path="output/fastspeech2_mn-May-10-2025_05+02PM-e55c4e2/config.json",
    progress_bar=True,
)

tts.tts_to_file(
    text="Сайн байна уу? Танд ямар тусламж хэрэгтэй вэ?",
    speaker="CV_Speaker_0004",
    file_path="output/test.wav"
)
