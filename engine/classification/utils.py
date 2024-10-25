import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score

import math
import warnings
from typing import List
import json

from torch.optim.lr_scheduler import _LRScheduler
from torch import nn as nn
from torch.optim import Optimizer

class Record(object):
    def __init__(self, args):
        self.num_data = 0
        self.y_true, self.y_pred, self.image_name = np.array([]), np.array([]), np.array([])
        with open(args.category_file, newline='') as jsonfile:
            self.category_dict = json.load(jsonfile)
        
        self.label_to_category = {}
        for key, value in self.category_dict.items():
            self.label_to_category[value] = key
        
    def update(self, pred, label, image_name):
        """
        pred: the predicted score of each class (Batch_size, num_classes)
        label: the ground truth labels (Batch_size, )
        image_name: Image name lists (Batch_size, )
        """
        pred, label = pred.cpu().detach().numpy(), label.cpu().detach().numpy()
        image_name = np.asarray(image_name)

        pred_label = np.argmax(pred, axis=-1) # (Batch_size, )
        self.num_data += len(pred)
        self.y_pred = np.append(self.y_pred, pred_label)
        self.y_true = np.append(self.y_true, label)
        self.image_name = np.append(self.image_name, image_name)

    def reset(self):
        self.num_data = 0
        self.y_true, self.y_pred, self.image_name = np.array([]), np.array([]), np.array([])

class Metric(object):
    def __init__(self, args):
        with open(args.category_file, newline='') as jsonfile:
            self.category_dict = json.load(jsonfile)

    def compute_metric(self, y_true, y_pred):
        metric_result = {"precision": {}, "recall": {}}
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)

        for idx, key in enumerate(self.category_dict.keys()):
            metric_result["precision"][key] = round(precision[idx], 3)
            metric_result["recall"][key] = round(recall[idx], 3)

        avg_acc = accuracy_score(y_true, y_pred) * 100
        return metric_result, avg_acc
        

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

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
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
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
        