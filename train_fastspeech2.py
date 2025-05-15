import os
import torch
from torch.utils.checkpoint import checkpoint
from trainer import Trainer, TrainerArgs
from trainer.model import TrainerModel
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from utils.formatter import common_voices_mn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, "output")
dataset_path = os.path.join(base_path, "dataset", "commonvoice")

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)

dataset_config = BaseDatasetConfig(
    meta_file_train="metadata.csv",
    path=dataset_path,
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=30,
    win_length=1024,
    hop_length=512,
    pitch_fmin=50.0,
    pitch_fmax=500.0,
)

model_args = ForwardTTS.args_type(use_pitch=True, use_energy=False, use_aligner=False)

config = Fastspeech2Config(
    project_name="fastspeech2_mn",
    run_description="FastSpeech2 Mongolian",
    run_name="fastspeech2_mn",
    epochs=1000,
    audio=audio_config,
    num_speakers=1,
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    text_cleaner="basic_cleaners",
    characters=CharactersConfig(
        characters="абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя ",
        punctuations=".,-:;!?()[]{}'\"",
    ),
    output_path=output_path,
    datasets=[dataset_config],
    mixed_precision=True,
    print_step=50,
    print_eval=False,
    run_eval=True,
    use_speaker_embedding=False,
    test_delay_epochs=-1,
    use_phonemes=False,
    phoneme_cache_path=None,
    f0_cache_path=None,
    energy_cache_path=None,
    model_args=model_args,
    max_seq_len=128,
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=common_voices_mn,
)

class MyFastSpeech2(ForwardTTS, TrainerModel):
    def __init__(self, config, ap, tokenizer, speaker_manager=None):
        super().__init__(config, ap, tokenizer, speaker_manager)
        for layer in getattr(self.encoder, 'layers', []):
            orig = layer.forward
            layer.forward = lambda *args, orig=orig: checkpoint(orig, *args, use_reentrant=False)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr, **self.config.optimizer_params)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def optimize(self, batch, trainer):
        batch = self.format_batch_on_device(batch)
        with torch.amp.autocast('cuda', enabled=trainer.use_amp_scaler):
            outputs, loss_dict = self.train_step(batch, trainer.criterion)
            loss = sum(loss_dict.values())
        self.scaled_backward(loss, trainer)
        trainer.optimizer.step()
        if trainer.scheduler:
            trainer.scheduler.step()
        trainer.optimizer.zero_grad()
        return outputs, loss_dict

model = MyFastSpeech2(config, ap, tokenizer)
trainer = Trainer(
    TrainerArgs(grad_accum_steps=4),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
