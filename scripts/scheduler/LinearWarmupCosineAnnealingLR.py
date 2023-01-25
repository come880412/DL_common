'''
"LinearWarmupCosineAnnealing" Learning rate scheduler trick decaying by epoch.
python scripts/scheduler/LinearWarmupCosineAnnealingLR.py <warmup_epochs> <max_epochs> <warmup_start_lr> <eta_min>
'''

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import warnings
from typing import List
import math
import argparse

class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of epochs for linear warmup
            max_epochs (int): Maximum number of epochs
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(8, 8)

def main(args):
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.warmup_epochs, 
                                              max_epochs=args.max_epochs, 
                                              warmup_start_lr=args.warmup_start_lr,
                                              eta_min=args.eta_min)
    
    epoch_list, lr_list = [], []
    for i in range(args.max_epochs):
        lr = optimizer.param_groups[0]['lr']
        epoch_list.append(i)
        lr_list.append(lr)
        scheduler.step()
    
    plt.plot(epoch_list, lr_list, color='blue')
    plt.savefig("./lr_decay.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Maximum number of epochs for linear warmup')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--warmup_start_lr', type=int, default=0, help='Learning rate to start the linear warmup. Default: 0.')
    parser.add_argument('--eta_min', type=int, default=0, help='Minimum learning rate. Default: 0.')
    args = parser.parse_args()

    main(args)
