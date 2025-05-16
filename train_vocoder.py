import os
import torch
from trainer import Trainer, TrainerArgs

from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
from TTS.utils.audio import AudioProcessor

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, "output", "vocoder")
dataset_path = os.path.join(base_path, "dataset", "commonvoice")

config_path = os.path.join(base_path, "output", "fastspeech2_mn-May-10-2025_05+02PM-e55c4e2", "config.json")
checkpoint_path = os.path.join(base_path, "output", "fastspeech2_mn-May-10-2025_05+02PM-e55c4e2", "checkpoint_510000.pth")

use_cuda = torch.cuda.is_available()

config = HifiganConfig(
    run_name="hifigan_mn",
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=25,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    mixed_precision=False,
    lr_gen=2e-4,
    lr_disc=2e-4,
    data_path=os.path.join(dataset_path, "wavs"),
    output_path=output_path,
)

audio_proc = AudioProcessor(**config.audio.to_dict())

eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

model = GAN(config, audio_proc)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()