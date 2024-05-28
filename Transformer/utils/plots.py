import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class Plotter:
    def __init__(self, output_dir) -> None:
        self.output_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_loss_per_epoch(self, train_loss: list, test_loss: list, lr_epoch_dict: Optional[dict] = None) -> None:
        x_axis = np.arange(1, len(train_loss) + 1)
        plt.title('Loss per epoch')
        plt.plot(x_axis,train_loss, label='train')
        plt.plot(x_axis,test_loss, label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if lr_epoch_dict is not None:
            for lr, epoch in lr_epoch_dict.items():
                plt.axvline(x=epoch, linestyle='--', label=lr)
                plt.text(epoch, 1, lr, rotation=90, verticalalignment='bottom')
        plt.savefig(os.path.join(self.output_dir, 'loss_per_epoch.png'))