import os
import torch
from torch import nn
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
phoneme_cache_path = os.path.join(output_path, "phoneme_cache")
f0_cache_path = os.path.join(output_path, "f0_cache")
energy_cache_path = os.path.join(output_path, "energy_cache")

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

config = Fastspeech2Config(
    project_name="fastspeech2_mn",
    run_description="FastSpeech2 training for Mongolian language",
    run_name="fastspeech2_mn",
    epochs=1000,
    audio=audio_config,
    num_speakers=1,
    batch_size=64,
    eval_batch_size=32,
    num_loader_workers=12,
    num_eval_loader_workers=6,
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
    phoneme_cache_path=phoneme_cache_path,
    f0_cache_path=f0_cache_path,
    energy_cache_path=energy_cache_path,
    test_sentences=[
        "Сайн байна уу?",
        "Монгол хэл бол гайхамшигтай.",
        "Өнөөдөр цаг агаар сайхан байна.",
        "Би ном унших дуртай.",
        "Таны нэр хэн бэ?",
        "Бид хамтдаа ажиллах болно.",
        "Энэ бол миний гэр бүл.",
        "Та хаанаас ирсэн бэ?",
        "Би кофе уухыг хүсч байна.",
        "Амжилт хүсье!",
    ],
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
        from torch.utils.checkpoint import checkpoint
        for layer in getattr(self.encoder, 'layers', []):
            orig_forward = layer.forward
            layer.forward = lambda *args, orig=orig_forward, checkpoint=checkpoint: checkpoint(orig, *args)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr, **self.config.optimizer_params)

    def get_scheduler(self, optimizer):
        torch.cuda.set_per_process_memory_fraction(0.9)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def optimize(self, batch, trainer):
        batch = self.format_batch_on_device(batch)
        with torch.amp.autocast("cuda", enabled=trainer.use_amp_scaler):
            outputs, loss_dict = self.train_step(batch, trainer.criterion)
            loss = sum(loss for loss in loss_dict.values())
        self.scaled_backward(loss, trainer)
        trainer.optimizer.step()
        if trainer.scheduler:
            trainer.scheduler.step()
        trainer.optimizer.zero_grad()
        return outputs, loss_dict

model = MyFastSpeech2(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(grad_accum_steps=8),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
