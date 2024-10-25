import tqdm
import os
import shutil
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch

from utils import Record, Metric

class Trainer(object):
    def __init__(self, args, model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, device):
        # Initialize parameters
        self.args = args
        self.model = model
        self.model_dir, self.model_name = args.model_dir, args.model_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.start_epoch = 0
        self.best_valid_acc, self.best_test_acc = 0., 0.
        self.record = Record(args)
        self.metric = Metric(args)
        
        # Writer setting
        if args.writer_overwrite:
            if os.path.isdir(os.path.join(args.writer_path, args.model_name)):
                shutil.rmtree(os.path.join(args.writer_path, args.model_name))
        self.writer = SummaryWriter(os.path.join(args.writer_path, args.model_name))

        # Reload checkpoint
        if args.resume:
            checkpoints = torch.load(args.resume)
            self.model.load_state_dict(checkpoints["model_state_dict"])
            self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoints["scheduler_state_dict"])
            self.start_epoch = checkpoints["epoch"]
            self.best_valid_acc = checkpoints["best_valid_acc"]
            self.best_test_acc = checkpoints["best_test_acc"]
    
    def train_one_epoch(self, epoch):
        self.model.train()

        loss_total = 0.
        num_data = 0
        lr = self.optimizer.param_groups[0]['lr']
        pbar = tqdm.tqdm(total=len(self.train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, self.args.n_epochs), unit="step")
        for image, label, image_name in self.train_loader:
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.criterion(pred, label)
            loss_total += loss * len(image)
            num_data += len(image)
            
            loss.backward()
            self.optimizer.step()

            pbar.update()
            pbar.set_postfix(
                loss=f"{(loss_total / num_data):.4f}",
                lr = f"{lr:.7f}"
            )
            self.record.update(pred, label, image_name)
        _, train_acc = self.metric.compute_metric(self.record.y_pred, self.record.y_true)
        self.record.reset()

        pbar.set_postfix(
            loss=f"{(loss_total / num_data):.4f}",
            train_acc=f"{train_acc:.3f}",
            lr = f"{lr:.7f}"
        )
        pbar.close()

        self.writer.add_scalar('Training loss', loss_total / num_data, epoch)
        self.writer.add_scalar('Training accuracy', train_acc, epoch)
        self.scheduler.step()

        return train_acc

    def Inference(self, epoch, mode="Valid"):
        self.model.eval()
        loader = self.val_loader if mode == "Valid" else self.test_loader

        pbar = tqdm.tqdm(total=len(loader), ncols=0, desc=f"{mode}", unit="step")
        loss_total = 0.
        num_data = 0
        with torch.no_grad():
            for image, label, image_name in loader:
                image, label = image.to(self.device), label.to(self.device)

                pred = self.model(image)
                loss = self.criterion(pred, label)
                loss_total += loss * len(image)
                num_data += len(image)

                pbar.update()
                pbar.set_postfix(
                    loss=f"{(loss_total / num_data):.4f}",
                )
                self.record.update(pred, label, image_name)
        _, acc = self.metric.compute_metric(self.record.y_pred, self.record.y_true)
        self.record.reset()
        pbar.set_postfix(
            loss=f"{(loss_total / num_data):.4f}",
            acc=f"{acc:.3f}"
        )
        pbar.close()
        self.writer.add_scalar(f'{mode} loss', loss_total / num_data, epoch)
        self.writer.add_scalar(f'{mode} accuracy', acc, epoch)

        return acc

    def train(self):    
        """training"""
        print('------- Start training! -------')
        for epoch in range(self.start_epoch+1, self.args.n_epochs):
            train_acc = self.train_one_epoch(epoch)
            valid_acc = self.Inference(epoch, mode = "Valid")
            test_acc = self.Inference(epoch, mode = "Test")

            if self.best_valid_acc <= valid_acc:
                print('Save model!!')
                self.best_valid_acc = valid_acc
                self.best_test_acc = test_acc
                checkpoints = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                    "best_test_acc": self.best_test_acc,
                }
                torch.save(checkpoints, os.path.join(self.model_dir, self.model_name, f"model_best_epoch{epoch+1}_acc{self.best_valid_acc:.3f}.pth"))
            
            checkpoints = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                    "best_test_acc": self.best_test_acc
                }
            torch.save(checkpoints, os.path.join(self.model_dir, self.model_name, "model_last.pth"))

            print(f'Best validation acc: {self.best_valid_acc:.3f}  |  Best testing acc: {self.best_test_acc:.3f}')