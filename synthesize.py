#!/usr/bin/env python
"""
Synthesize sentences into speech.

Author: Erdene-Ochir Tuguldur
"""

import os
import sys
import argparse
from tqdm import tqdm

import numpy as np
import torch

from models import Text2Mel, SSRN
from models.hparams import HParams as hp
from utils.audio import save_to_wav
from utils.utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech'], help='dataset name')
    args = parser.parse_args()

    if args.dataset == 'ljspeech':
        from datasets.lj_speech import vocab, get_test_data
        SENTENCES = [
            "The birch canoe slid on the smooth planks.",
            "Glue the sheet to the dark blue background.",
        ]
    else:
        from datasets.mb_speech import vocab, get_test_data
        SENTENCES = [
            "Нийслэлийн прокурорын газраас төрийн өндөр албан тушаалтнуудад холбогдох зарим эрүүгийн хэргүүдийг шүүхэд шилжүүлэв.",
        ]

    torch.set_grad_enabled(False)
    text2mel = Text2Mel(vocab).eval().cuda()
    ssrn = SSRN().eval().cuda()

    t2m_ckpt = get_last_checkpoint_file_name(os.path.join(hp.logdir, f'{args.dataset}-text2mel'))
    if t2m_ckpt:
        print(f"Loading text2mel checkpoint '{t2m_ckpt}'...")
        load_checkpoint(t2m_ckpt, text2mel, None)
    else:
        print("Text2Mel checkpoint does not exist")
        sys.exit(1)

    ssrn_ckpt = get_last_checkpoint_file_name(os.path.join(hp.logdir, f'{args.dataset}-ssrn'))
    if ssrn_ckpt:
        print(f"Loading ssrn checkpoint '{ssrn_ckpt}'...")
        load_checkpoint(ssrn_ckpt, ssrn, None)
    else:
        print("SSRN checkpoint does not exist")
        sys.exit(1)

    os.makedirs("samples", exist_ok=True)
    for i, sentence in enumerate(SENTENCES, 1):
        with torch.no_grad():
            sentences = [sentence]
            max_N = len(sentence)
            L = torch.from_numpy(get_test_data(sentences, max_N)).cuda()
            zeros = torch.zeros((1, hp.n_mels, 1), dtype=torch.float32).cuda()
            Y = zeros
            A = None

            for t in tqdm(range(hp.max_T), desc="Synthesizing"):
                _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
                Y = torch.cat((zeros, Y_t), dim=-1)
                attention = A[0, :, -1].argmax().item()
                if L[0, attention] == vocab.index('E'):  # EOS condition
                    break

            _, Z = ssrn(Y)

            # Convert tensors to numpy arrays
            Y_np = Y.cpu().detach().numpy()
            A_np = A.cpu().detach().numpy()
            Z_np = Z.cpu().detach().numpy()

            save_to_png(f'samples/{i}-att.png', A_np[0])
            save_to_png(f'samples/{i}-mel.png', Y_np[0])
            save_to_png(f'samples/{i}-mag.png', Z_np[0])
            save_to_wav(Z_np[0].T, f'samples/{i}-wav.wav')

if __name__ == "__main__":
    main()
