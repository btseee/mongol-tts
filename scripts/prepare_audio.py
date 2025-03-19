import torchaudio

audio_path = "data/raw/100jil.wav"
waveform, sample_rate = torchaudio.load(audio_path)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
torchaudio.save("data/processed/audio.wav", waveform, 16000)