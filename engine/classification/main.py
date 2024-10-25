import os

import torch.nn as nn
import torch

from utils import fixed_seed, LinearWarmupCosineAnnealingLR
from optimizer.optim_factory import create_optimizer_v2
from dataset.Dataset import get_dataloader
from config.config import load_config
from models.build_model import build_model
from train_engine import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    train_loader, val_loader, test_loader, n_classes = get_dataloader(args)
    model = build_model(args, device, n_classes)
    optimizer = create_optimizer_v2(model.parameters(), opt=args.opt, lr=args.lr, weight_decay = args.weight_decay, momentum=args.momentum)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=args.warmup_lr, warmup_epochs=args.warmup_epochs, max_epochs=args.n_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    trainer = Trainer(args, model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, device)
    trainer.train()


if __name__ == '__main__':
    args = load_config()

    fixed_seed(args.seed)
    os.makedirs(os.path.join(args.model_dir, args.model_name), exist_ok=True)
    print("Model_name: %s  |  Optimizer: %s" % (args.model_name, args.opt))
    main(args)


