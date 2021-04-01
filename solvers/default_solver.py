import importlib

import numpy as np
import torch
import tqdm

from datasets import get_dataloader
from models import get_model
from utils.lookahead import Lookahead
from utils.metrics import angular_error, AverageMeter


class Solver:
    def __init__(self, config):
        self.config = config
        self.mode = self.config.mode
        self.model = get_model(self.config)

        self.use_gpu = torch.cuda.is_available()
        self.use_val = self.config.use_val
        self.lr = self.config.learning_rate
        self.epochs = self.config.epochs
        self.start_epoch = self.config.start_epoch
        self.batch_size = self.config.batch_size
        self.print_freq = self.config.print_freq
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

        optimizer_cls = getattr(
            importlib.import_module('torch.optim'), self.config.optimizer
         )
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)
        self.scheduler = Lookahead(self.optimizer, k=5, alpha=0.5)

    def load_checkpoint(self):
        pass

    @staticmethod
    def send_dict_to_gpu(d):
        for key, value in d.items():
            d[key] = value.cuda()
        return d

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()

    def train_one_epoch(self, epoch):
        train_errors = AverageMeter()
        train_losses = AverageMeter()
        train_iter = tqdm.tqdm(self.train_loader, desc='Train Epoch', ncols=10, total=self.n_batch_train, leave=False)
        self.model.train()
        for batch in train_iter:
            if self.use_gpu:
                batch = self.send_dict_to_gpu(batch)
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
            loss_gaze.backward()
            self.optimizer.step()
            train_losses.update(loss_gaze.item(), num)

            postfix = {
                'Error': train_errors.avg,
                'Loss': train_losses.avg
            }
            train_iter.postfix(postfix)

        if self.use_val:
            self.model.eval()
            val_errors = AverageMeter()
            val_losses = AverageMeter()
            val_iter = tqdm.tqdm(self.val_loader, desc='Val', ncols=10, total=self.n_batch_val, leave=False)
            for batch in val_iter:
                if self.use_gpu:
                    batch = self.send_dict_to_gpu(batch)
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

                postfix = {
                    'Error': val_errors.avg,
                    'Loss': val_losses.avg
                }
                val_iter.postfix(postfix)

    def train(self):
        epoch_iter = tqdm.tqdm(range(self.start_epoch, self.epochs), desc='Train')
        for epoch in epoch_iter:
            pass

    def test(self):
        pass

