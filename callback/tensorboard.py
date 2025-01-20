import time

from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    def __init__(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"runs/{timestamp}")

    def log_metrics(self, epoch, train_loss, val_loss, lr):

        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("Learning_Rate", lr, epoch)

    def close(self):
        self.writer.close()
