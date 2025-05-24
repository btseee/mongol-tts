import os

from trainer import Trainer, TrainerArgs

from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
from TTS.utils.audio import AudioProcessor


base_path = os.path.dirname(os.path.abspath(__file__))
output_path_hifigan = os.path.join(base_path, "output_hifigan_mn_v1")
dataset_wav_path = os.path.join(base_path, "dataset", "commonvoice", "wavs")

os.makedirs(output_path_hifigan, exist_ok=True)

config = HifiganConfig(
    run_name="hifigan_mn_cyrillic_run1",
    project_name="hifigan_mongolian",
    epochs=2500,
    test_delay_epochs=50,
    print_step=100,
    run_eval=True,
    print_eval=False,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    mixed_precision=True,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    lr_gen=2e-4,
    lr_disc=2e-4,
    optimizer_gen="AdamW",
    optimizer_disc="AdamW",
    optimizer_params={"betas": [0.8, 0.99], "weight_decay": 0.0},
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_gen_params={"gamma": 0.9999, "last_epoch": -1},
    lr_scheduler_disc="ExponentialLR",
    lr_scheduler_disc_params={"gamma": 0.9999, "last_epoch": -1},
    use_cache=True,
    data_path=dataset_wav_path,
    output_path=output_path_hifigan,
    audio={
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024,
        "fft_size": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "num_mels": 80,
    },
    eval_split_size=256,
    speakers_file=None, 
    num_speakers=0, 
)

audio_proc = AudioProcessor(**config.audio)

eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

model = GAN(config, audio_proc)

trainer_args = TrainerArgs()

trainer = Trainer(
    trainer_args,
    config,
    output_path_hifigan,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()