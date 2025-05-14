# Mongolian Text-To-Speech (mongol-tts)

A complete Mongolian TTS pipeline built with FastSpeech2 for acoustic modeling and Coqui TTS for neural vocoding. This repository leverages:

- [FastSpeech2](https://arxiv.org/abs/2006.04558): Non-autoregressive transformer-based acoustic model.

- [CoquiTTS](https://github.com/idiap/coqui-ai-TTS): High-quality neural vocoder and inference engine.

- [num2words](https://github.com/savoirfairelinux/num2words.git): For normalizing numeric tokens in Mongolian text.

## 🔍 Repository Structure

```sh
├── .gitmodules              # Defines coqui-ai-TTS & num2words submodules
├── preprocess.py            # Text and audio preprocessing
├── train_fastspeech2.py     # FastSpeech2 acoustic model training
├── train_vocoder.py         # Neural vocoder training (CoquiTTS)
├── synthesize.py            # Inference: generate audio from raw text
├── utils/
│   └── num2words/           # Submodule for numeric normalization
└── coqui-ai-TTS/            # Submodule: Coqui TTS engine & vocoder code
```

## ⚙️ Installation

Note: Requires Python ≥ 3.8

1. Clone with submodules

    ```sh
    git clone <https://github.com/btseee/mongol-tts.git> --recursive
    cd mongol-tts
    ```

2. Install the package and submodules

    ```sh
    pip install .
    ```

This single command will:

- Initialize and update the coqui-ai-TTS and num2words submodules

- Install coqui-ai-TTS in editable mode (-e)

- Install num2words via its own setup.py

- Install the core mongol-tts package and dependencies (torch, etc.)

## 🛠️ Usage

1. Data Preparation

    Prepare your parallel dataset of Mongolian text and corresponding waveforms:

    ```sh
    python preprocess.py \
    --input_dir path/to/wavs \
    --output_dir data/processed \
    --sampling_rate 22050
    ```

    This script will:

    - Normalize punctuation and numbers (using num2words)

    - Extract mel-spectrogram features

    - Save metadata for training

2. Train FastSpeech2 Acoustic Model

    ```sh
    python train_fastspeech2.py \
    --config config/fastspeech2.yaml \
    --data_dir data/processed \
    --output_dir checkpoints/fastspeech2
    ```

    Adjust the YAML config for model hyperparameters, learning rate, batch size, etc.

3. Train Neural Vocoder (Coqui TTS)

    ```sh
    python train_vocoder.py \
    --config config/vocoder.yaml \
    --data_dir data/processed \
    --output_dir checkpoints/vocoder
    ```

    Customize the vocoder config for architecture (e.g., HiFi-GAN, MelGAN), training steps, and sample rate.

4. Synthesize Speech

    Once both models are trained, generate audio from raw Mongolian text:

    ```sh
    python synthesize.py \
    --text "Сайн байна уу, энэ бол туршилтын текст юм." \
    --acoustic_model checkpoints/fastspeech2/best_model.pth \
    --vocoder_model checkpoints/vocoder/best_model.pth \
    --output_path samples/example.wav
    ```

    **Output**: samples/example.wav

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

Improved Mongolian text normalization

New model architectures or training recipes

Bug fixes and documentation enhancements

## 📜 License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0).
