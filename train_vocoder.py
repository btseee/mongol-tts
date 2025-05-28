import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    voc_batch = 16
else:
    voc_batch = 4

from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

from src.dataset import prepare_dataset

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH , "dataset", "wavs")
OUTPUT_PATH = os.path.join(BASE_PATH, "output", "hifigan_run")
DATASET_PATH = os.path.join(BASE_PATH , "dataset")

os.makedirs(OUTPUT_PATH, exist_ok=True)

prepare_dataset(DATASET_PATH)

config = HifiganConfig(
    run_name="hifigan_mn_run",
    output_path=OUTPUT_PATH,
    batch_size=voc_batch,
    eval_batch_size=voc_batch//2,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    epochs=1000,
    run_eval=True,
    test_delay_epochs=5,
    print_step=50,
    print_eval=False,
    sample_rate=22050,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=DATA_PATH
)

ap = AudioProcessor(**config.audio.to_dict())

eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

model = GAN(config=config, ap=ap)

trainer = Trainer(
    TrainerArgs(), config, OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

trainer.fit()
