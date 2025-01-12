import time

from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    def __init__(self, log_dir=None):
        """
        Initialize TensorBoard writer.

        Args:
            log_dir (str, optional): Directory where TensorBoard logs will be written.
                                   If None, defaults to 'runs/timestamp'
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=log_dir or f'runs/{timestamp}')
        self.current_epoch = 0

    def on_epoch_end(self, epoch, loss_dict, lr):

        self.current_epoch = epoch

        if 'train_loss' in loss_dict:
            self.writer.add_scalar('Loss/train', loss_dict['train_loss'], epoch)
        if 'val_loss' in loss_dict:
            self.writer.add_scalar('Loss/val', loss_dict['val_loss'], epoch)
        if 'loss' in loss_dict:
            self.writer.add_scalar('Loss/total', loss_dict['loss'], epoch)

        self.writer.add_scalar('Learning_Rate', lr, epoch)

    def log_batch(self, epoch, batch_idx, loss_dict, lr):

        global_step = epoch * 1000 + batch_idx

        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, (int, float)):  # Skip non-numeric values
                self.writer.add_scalar(f'Batch/{loss_name}', loss_value, global_step)

        self.writer.add_scalar('Batch/Learning_Rate', lr, global_step)

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()