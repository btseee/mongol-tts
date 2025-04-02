#!/usr/bin/env python
"""
Train the Text2Mel network.
Based on: https://arxiv.org/abs/1710.08969

Author: Erdene-Ochir Tuguldur
"""

import sys
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from models import Text2Mel
from models.hparams import HParams as hp
from utils.utils import get_last_checkpoint_file_name, load_checkpoint, save_checkpoint
from datasets.data_loader import Text2MelDataLoader
from utils.logger import Logger

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'commonvoice'], help='dataset name')
    args = parser.parse_args()

    if args.dataset == 'ljspeech':
        from datasets.lj_speech import vocab, LJSpeech as SpeechDataset
    elif args.dataset == 'mbspeech':
        from datasets.mb_speech import vocab, MBSpeech as SpeechDataset
    elif args.dataset == 'commonvoice':
        from datasets.common_voice import vocab, CommonVoiceMongolian as SpeechDataset

    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    train_data_loader = Text2MelDataLoader(
        text2mel_dataset=SpeechDataset(['texts', 'mels', 'mel_gates']),
        batch_size=64,
        mode='train'
    )
    valid_data_loader = Text2MelDataLoader(
        text2mel_dataset=SpeechDataset(['texts', 'mels', 'mel_gates']),
        batch_size=64,
        mode='valid'
    )

    text2mel = Text2Mel(vocab).cuda()
    optimizer = torch.optim.Adam(text2mel.parameters(), lr=hp.text2mel_lr)

    start_epoch = 0
    global_step = 0
    logger = Logger(args.dataset, 'text2mel')

    ckpt = get_last_checkpoint_file_name(logger.logdir)
    if ckpt:
        print(f"Loading last checkpoint: {ckpt}")
        start_epoch, global_step = load_checkpoint(ckpt, text2mel, optimizer)

    def get_lr() -> float:
        return optimizer.param_groups[0]['lr']

    def lr_decay(step: int, warmup_steps: int = 4000) -> None:
        new_lr = hp.text2mel_lr * (warmup_steps ** 0.5) * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
        optimizer.param_groups[0]['lr'] = new_lr

    def train_epoch(epoch_num: int, phase: str = 'train') -> float:
        nonlocal global_step
        lr_decay(global_step)
        print(f"Epoch {epoch_num:3d} with lr={get_lr():.2e}")

        text2mel.train() if phase == 'train' else text2mel.eval()
        torch.set_grad_enabled(phase == 'train')
        data_loader = train_data_loader if phase == 'train' else valid_data_loader

        it = 0
        running_loss = 0.0
        running_l1_loss = 0.0
        running_att_loss = 0.0

        pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size,
                    disable=hp.disable_progress_bar)
        for batch in pbar:
            L, S, gates = batch['texts'], batch['mels'], batch['mel_gates']
            S = S.permute(0, 2, 1).cuda()  # Adjust dimensions as needed
            B, N = L.size()
            _, n_mels, T = S.size()

            S_shifted = torch.cat((S[:, :, 1:], torch.zeros(B, n_mels, 1, device=S.device)), dim=2)

            L = L.cuda()
            S = S.cuda()
            S_shifted = S_shifted.cuda()
            gates = gates.cuda()

            # Create attention weight mask W (shape: [B, N, T])
            def W_nt(_, n, t, g=0.2):
                return 1.0 - np.exp(-((n / float(N) - t / float(T)) ** 2) / (2 * g ** 2))
            W = np.fromfunction(W_nt, (B, N, T), dtype=np.float32)
            W = torch.from_numpy(W).cuda()

            Y_logit, Y, A = text2mel(L, S)

            l1_loss = F.l1_loss(Y, S_shifted)
            masks = gates.reshape(B, 1, T).float()
            att_loss = (A * W * masks).mean()
            loss = l1_loss + att_loss

            if phase == 'train':
                lr_decay(global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

            it += 1
            running_loss += loss.item()
            running_l1_loss += l1_loss.item()
            running_att_loss += att_loss.item()

            if phase == 'train':
                pbar.set_postfix({
                    'l1': f"{running_l1_loss / it:.05f}",
                    'att': f"{running_att_loss / it:.05f}"
                })
                logger.log_step(phase, global_step,
                                {'loss_l1': l1_loss.item(), 'loss_att': att_loss.item()},
                                {'mels-true': S[:1].cpu(), 'mels-pred': Y[:1].cpu(), 'attention': A[:1].cpu()})
                if global_step % 5000 == 0:
                    save_checkpoint(logger.logdir, epoch_num, global_step, text2mel, optimizer)

        epoch_loss = running_loss / it
        logger.log_epoch(phase, global_step, {
            'loss_l1': running_l1_loss / it,
            'loss_att': running_att_loss / it
        })
        return epoch_loss

    since = time.time()
    epoch = start_epoch
    while True:
        train_loss = train_epoch(epoch, phase='train')
        elapsed = time.time() - since
        time_str = f"total time elapsed: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"
        print(f"Train epoch loss {train_loss:.6f}, step={global_step}, {time_str}")

        valid_loss = train_epoch(epoch, phase='valid')
        print(f"Validation epoch loss {valid_loss:.6f}")

        epoch += 1
        if global_step >= hp.text2mel_max_iteration:
            print(f"Max step {hp.text2mel_max_iteration} reached (current step {global_step}), exiting...")
            sys.exit(0)

if __name__ == "__main__":
    main()
