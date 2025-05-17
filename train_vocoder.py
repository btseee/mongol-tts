import os
from trainer import Trainer, TrainerArgs

from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
from TTS.utils.audio import AudioProcessor

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, "output")
dataset_path = os.path.join(base_path, "dataset", "commonvoice")

config = HifiganConfig(
    run_name="hifigan_mn",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    mixed_precision=True,
    run_eval=True,
    test_delay_epochs=5,
    epochs=100,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    lr_gen=1e-4,
    lr_disc=1e-4,
    wd=1e-6,
    use_cache=True,
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