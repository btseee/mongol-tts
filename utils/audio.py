"""
Audio processing utilities.

These methods are adapted from:
https://github.com/Kyubyong/dc_tts/
"""

import os
import copy
import librosa
import scipy.io.wavfile
import numpy as np
from typing import Tuple

from tqdm import tqdm
from scipy import signal
from models.hparams import HParams as hp


def spectrogram2wav(mag: np.ndarray) -> np.ndarray:
    """
    Convert a linear magnitude spectrogram to a waveform.
    
    Args:
        mag: A numpy array of shape (T, 1+n_fft//2) with values in [0, 1].
    
    Returns:
        A 1-D numpy array representing the reconstructed waveform.
    """
    # Transpose spectrogram to shape (1+n_fft//2, T)
    mag = mag.T

    # De-normalize magnitude
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # Convert from dB to amplitude
    mag = np.power(10.0, mag * 0.05)

    # Reconstruct waveform using Griffin-Lim algorithm
    wav = griffin_lim(mag ** hp.power)

    # Apply de-preemphasis filter
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # Trim silence
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram: np.ndarray) -> np.ndarray:
    """
    Reconstruct a waveform from a magnitude spectrogram using the Griffin-Lim algorithm.
    
    Args:
        spectrogram: A numpy array representing the magnitude spectrogram.
    
    Returns:
        The reconstructed waveform as a numpy array.
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Invert a spectrogram to a waveform using the inverse STFT.
    
    Args:
        spectrogram: A numpy array of shape (1+n_fft//2, t).
    
    Returns:
        A waveform as a numpy array.
    """
    return librosa.istft(spectrogram, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")


def get_spectrograms(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a WAV file and compute its normalized mel and linear spectrograms.
    
    Args:
        fpath: Full path to a sound file.
    
    Returns:
        A tuple (mel, mag) where:
          - mel is a 2D numpy array of shape (T, n_mels) with type float32.
          - mag is a 2D numpy array of shape (T, 1+n_fft//2) with type float32.
    """
    # Load sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Apply preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # Compute short-time Fourier transform
    linear = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

    # Compute magnitude spectrogram
    mag = np.abs(linear)

    # Compute mel spectrogram using mel filter bank
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    mel = np.dot(mel_basis, mag)

    # Convert to decibels
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Normalize spectrograms to [0, 1]
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose and convert to float32
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    return mel, mag


def save_to_wav(mag: np.ndarray, filename: str) -> None:
    """
    Generate a waveform from a linear spectrogram and save it as a WAV file.
    
    Args:
        mag: Linear spectrogram as a numpy array.
        filename: Destination file path for the WAV file.
    """
    wav = spectrogram2wav(mag)
    scipy.io.wavfile.write(filename, hp.sr, wav)


def preprocess(dataset_path: str, speech_dataset) -> None:
    """
    Preprocess a speech dataset: compute and save mel and linear spectrograms for each sample.
    
    Args:
        dataset_path: Base path for the dataset containing a 'wavs' folder.
        speech_dataset: An object with an attribute 'fnames' containing file identifiers.
    """
    wavs_path = os.path.join(dataset_path, 'wavs')
    mels_path = os.path.join(dataset_path, 'mels')
    mags_path = os.path.join(dataset_path, 'mags')

    os.makedirs(mels_path, exist_ok=True)
    os.makedirs(mags_path, exist_ok=True)

    for fname in tqdm(speech_dataset.fnames, desc="Preprocessing audio"):
        mel, mag = get_spectrograms(os.path.join(wavs_path, f'{fname}.wav'))
        t = mel.shape[0]

        # Compute padding to align frames with reduction rate
        num_paddings = (hp.reduction_rate - (t % hp.reduction_rate)) % hp.reduction_rate
        mel = np.pad(mel, ((0, num_paddings), (0, 0)), mode="constant")
        mag = np.pad(mag, ((0, num_paddings), (0, 0)), mode="constant")

        # Apply reduction by taking every nth frame
        mel = mel[::hp.reduction_rate, :]

        np.save(os.path.join(mels_path, f'{fname}.npy'), mel)
        np.save(os.path.join(mags_path, f'{fname}.npy'), mag)
