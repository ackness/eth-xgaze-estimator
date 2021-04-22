import importlib
import os

import numpy as np
import torch
import tqdm
import wandb
from accelerate import Accelerator
from torch.optim.lr_scheduler import StepLR

from datasets import get_dataloader
from models import get_model
from utils.metrics import angular_error, AverageMeter

accelerator = Accelerator()


class Solver:
    def __init__(self, config):
        self.config = config
        self.mode = self.config.mode
        self.model = get_model(self.config)
        self.model_name = self.config.model_name
        self.load_checkpoint()
        self.use_val = self.config.use_val
        self.lr = self.config.learning_rate
        self.epochs = self.config.epochs
        self.start_epoch = self.config.start_epoch
        self.batch_size = self.config.batch_size
        self.checkpoint_path = self.config.checkpoint_path
        self.pose_mode = self.config.is_load_pose

        if self.mode == 'train':
            if self.use_val:
                self.train_loader, self.val_loader = get_dataloader(self.config)
                self.n_batch_train = len(self.train_loader)
                self.n_batch_val = len(self.val_loader)
            else:
                self.train_loader = get_dataloader(self.config)
                self.n_batch_train = len(self.train_loader)
        elif self.mode == 'test':
            self.test_loader = get_dataloader(self.config)
            self.n_batch_test = len(self.test_loader)

        self.criterion = torch.nn.L1Loss()
        # self.criterion = CosineLoss()
        optimizer_cls = getattr(
            importlib.import_module('torch.optim'), self.config.optimizer
        )
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)
        # self.scheduler = Lookahead(self.optimizer, k=5, alpha=0.5)
        self.scheduler = StepLR(self.optimizer, step_size=self)

        self.best_ae = float('inf')
        self.best_loss = float('inf')

        if self.config.wandb:
            print("Using wandb to log results")
            name = self.config.prefix + "_" + self.model_name + "_" + self.config.solver
            wandb.init(project='gaze-estimation', entity=self.config.wandb_entity, name=name)
            wandb.config.update(config)

        # use accelerator to run on different devices easily
        self.model, self.optimizer, self.train_loader = accelerator.prepare(
            self.model, self.optimizer, self.train_loader)

    def load_checkpoint(self):
        if self.config.resume:
            ckpt = torch.load(self.config.pre_trained_model_path)
            self.model.load_state_dict(ckpt)

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '_ckpt.pth.tar'
        else:
            filename = 'ckpt.pth.tar'
        ckpt_path = os.path.join(self.checkpoint_path, filename)
        torch.save(state, ckpt_path)

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()

    def train_one_epoch(self, epoch):
        train_errors = AverageMeter()
        train_losses = AverageMeter()
        train_iter = tqdm.tqdm(self.train_loader, desc='Train Epoch', total=self.n_batch_train, leave=False)
        self.model.train()
        for i, batch in enumerate(train_iter):
            image = batch['image']
            gaze = batch['gaze']
            if self.pose_mode:
                pose = batch['pose']
                out = self.model(image, pose)
            else:
                out = self.model(image)
            num = image.size()[0]
            gaze_error_batch = np.mean(angular_error(out.cpu().data.numpy(), gaze.cpu().data.numpy()))
            train_errors.update(gaze_error_batch.item(), num)

            loss_gaze = self.criterion(out, gaze)
            self.optimizer.zero_grad()
            # loss_gaze.backward()
            accelerator.backward(loss_gaze)
            self.optimizer.step()
            train_losses.update(loss_gaze.item(), num)

            if i % self.config.log_freq == 0:
                if self.config.wandb:
                    wandb.log({'epoch': epoch, "batch": i, "Train Errors": train_errors.avg,
                               "Train Losses": train_losses.avg})

                postfix = {
                    'Error': train_errors.avg,
                    'Loss': train_losses.avg
                }
                train_iter.set_postfix(postfix)
                train_errors.reset()
                train_losses.reset()

        if self.use_val:
            self.model.eval()
            val_errors = AverageMeter()
            val_losses = AverageMeter()
            val_iter = tqdm.tqdm(self.val_loader, desc='Val', total=self.n_batch_val, leave=False)
            for i, batch in enumerate(val_iter):
                image = batch['image']
                gaze = batch['gaze']
                if self.pose_mode:
                    pose = batch['pose']
                    out = self.model(image, pose)
                else:
                    out = self.model(image)
                num = image.size()[0]
                gaze_error_batch = np.mean(angular_error(out.cpu().data.numpy(), gaze.cpu().data.numpy()))
                val_errors.update(gaze_error_batch.item(), num)
                loss_gaze = self.criterion(out, gaze)
                val_losses.update(loss_gaze.item(), num)

                if i % self.config.log_freq == 0:
                    postfix = {
                        'Error': val_errors.avg,
                        'Loss': val_losses.avg
                    }
                    val_iter.set_postfix(postfix)

            if self.config.wandb:
                wandb.log({'epoch': epoch, "Val Errors": val_errors.avg, "Val Losses": val_losses.avg})

        return train_errors.avg, train_losses.avg

    def train(self):
        epoch_iter = tqdm.tqdm(range(self.start_epoch, self.epochs), desc='Train')
        for epoch in epoch_iter:
            train_errors_avg, train_losses_avg = self.train_one_epoch(epoch)
            self.scheduler.step()

            if train_errors_avg < self.best_ae:
                self.best_ae = train_errors_avg
            if train_losses_avg < self.best_loss:
                self.best_loss = train_losses_avg

            postfix = {
                'Best-mAE': self.best_ae,
                'Best-LOSS': self.best_loss,
            }
            epoch_iter.set_postfix(postfix)

            add_file_name = os.path.join(
                self.checkpoint_path,
                self.config.prefix + "_" + self.model_name,
                "solver_" + self.config.solver,
                f'Epoch_{epoch}'
            )
            self.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'scheule_state': self.scheduler.state_dict(),
                }, add=add_file_name
            )

    def test(self):
        pass
