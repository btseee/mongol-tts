import os
from trainer import Trainer, TrainerArgs
from TTS.TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.TTS.tts.datasets import load_tts_samples
from TTS.TTS.tts.models.glow_tts import GlowTTS
from TTS.TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.TTS.utils.audio import AudioProcessor
from TTS.TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig, CharactersConfig
from utils.get_batch_size import get_auto_batch_size, get_auto_num_workers

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, "models")
DATASET_PATH = os.path.join(BASE_PATH, "dataset")

os.makedirs(OUTPUT_PATH, exist_ok=True)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="Common-Voice-by-Mozilla",
    language="mn",
    path=DATASET_PATH,
    meta_file_train="metadata_train.csv",
    meta_file_val="metadata_test.csv",    
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=23,
)

config = GlowTTSConfig(
    batch_size = get_auto_batch_size(is_training=True, max_batch=64),
    eval_batch_size = get_auto_batch_size(is_training=False, max_batch=128),
    num_loader_workers = get_auto_num_workers(is_training=True),
    num_eval_loader_workers = get_auto_num_workers(is_training=False),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False, 
    phoneme_language="mn-cyrl", 
    print_step=25,
    mixed_precision=True,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    audio=audio_config,
    test_sentences=[
        "Сайн байна уу?",   
        "Би монгол хүн.",          
        "Өнөөдөр цаг агаар сайхан байна."
    ],
    characters=CharactersConfig(
        characters = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя",
        punctuations=" !\"'(),-.:;?[]{}«»“”‘’",
        blank="",
        space=" ",
        bos="[BOS]",
        eos="[EOS]",
        pad="[PAD]",
        is_sorted=False,
        use_phonemes=False,
        is_unique=False,
        phonemes="",
        vocab_dict=None,
    ),
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

config.print_step = max(1, len(train_samples) // (config.batch_size * 10))
iterations_per_epoch = max(1, len(train_samples)  // config.batch_size)
config.epochs = max(1, 50000 // iterations_per_epoch)

model = GlowTTS(config, ap, tokenizer)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path = OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

trainer.fit()
