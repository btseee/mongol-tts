import torch
from torch import nn
from trainer.model import TrainerModel
from TTS.tts.models.forward_tts import ForwardTTS

class MyFastSpeech2(ForwardTTS, TrainerModel):
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def get_criterion(self):
        # assume your ForwardTTS.forward returns a dict with 'loss'
        return None  # you’ll extract the loss directly in optimize()

    def optimize(self, batch, trainer):
        # 1) Format the batch on device
        batch = self.format_batch_on_device(batch)

        # 2) Forward + loss
        with torch.cuda.amp.autocast(enabled=trainer.use_amp_scaler):
            outputs = self(**batch)                # calls ForwardTTS.forward()
            loss = outputs["loss"]               # adapt key as needed

        # 3) Backward + step
        self.scaled_backward(loss, trainer)     # handles AMP if enabled
        trainer.optimizer.step()
        if trainer.scheduler:
            trainer.scheduler.step()
        trainer.optimizer.zero_grad()

        # 4) Return outputs and a loss dict for logging
        loss_dict = {"total_loss": loss.detach()}
        return outputs, loss_dict
