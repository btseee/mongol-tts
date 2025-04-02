"""Hyperparameters module.

Author: Erdene-Ochir Tuguldur
"""

class HParams:
    """Hyper parameters for the TTS models."""

    # General settings
    disable_progress_bar = False  # Set True to disable the console progress bar
    logdir = "logdir"  # Directory where checkpoints and tensorboard logs are saved

    # Audio processing options (from https://github.com/Kyubyong/dc_tts/blob/master/hyperparams.py)
    reduction_rate = 4             # Mel spectrogram reduction rate (SSRN uses this rate)
    n_fft = 2048                   # Number of FFT points (samples)
    n_mels = 80                    # Number of Mel filter banks
    power = 1.5                    # Exponent for amplifying the predicted magnitude
    n_iter = 50                    # Number of iterations for waveform inversion
    preemphasis = 0.97
    max_db = 100
    ref_db = 20
    sr = 22050                     # Sampling rate
    frame_shift = 0.0125           # Frame shift in seconds
    frame_length = 0.05            # Frame length in seconds
    hop_length = int(sr * frame_shift)  # Hop length in samples (e.g., 276)
    win_length = int(sr * frame_length) # Window length in samples (e.g., 1102)
    max_N = 180                    # Maximum number of characters
    max_T = 210                    # Maximum number of mel frames

    # Model dimensions
    e = 128      # Embedding dimension
    d = 256      # Text2Mel hidden unit dimension
    c = 512 + 128  # SSRN hidden unit dimension

    dropout_rate = 0.05  # Dropout rate

    # Text2Mel network options
    text2mel_lr = 0.005                  # Learning rate
    text2mel_max_iteration = 300000      # Maximum training iterations
    text2mel_weight_init = 'none'        # Options: 'kaiming', 'xavier', 'none'
    text2mel_normalization = 'layer'     # Options: 'layer', 'weight', 'none'
    text2mel_basic_block = 'gated_conv'    # Options: 'highway', 'gated_conv', 'residual'

    # SSRN network options
    ssrn_lr = 0.0005                     # Learning rate
    ssrn_max_iteration = 150000          # Maximum training iterations
    ssrn_weight_init = 'kaiming'         # Options: 'kaiming', 'xavier', 'none'
    ssrn_normalization = 'weight'        # Options: 'layer', 'weight', 'none'
    ssrn_basic_block = 'residual'        # Options: 'highway', 'gated_conv', 'residual'
