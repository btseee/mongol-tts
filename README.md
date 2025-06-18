# Mongolian Text-To-Speech (TTS) with VITS & CoquiTTS

A simple pipeline for training Mongolian TTS models using [VITS](https://github.com/jaywalnut310/vits) and [CoquiTTS](https://github.com/coqui-ai/TTS).

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/btseee/mongol-tts.git --recursive
cd mongol-tts
```

### 2. Set Up the Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the Dataset

This project uses the [btsee/mbspeech_mn](https://huggingface.co/datasets/btsee/mbspeech_mn) dataset.

```bash
python dataset.py
```

### 4. Start Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_vits.py
```

---

## 📄 Notes

- Make sure you have a compatible GPU and CUDA installed.
- For more details, see the [CoquiTTS documentation](https://tts.readthedocs.io/).
- Contributions and issues are welcome!
