import torch

from trainer.model import TrainerModel
from TTS.tts.models.forward_tts import ForwardTTS

class MyFastSpeech2(ForwardTTS, TrainerModel):
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr, **self.config.optimizer_params)

    def get_scheduler(self, optimizer):
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