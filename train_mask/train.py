import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger

# arguments
from opt import get_opts
# dataset
from datasets import dataset_dict
# optimizer, scheduler, visualization
from utils import *
# losses
from losses import loss_dict
# metrics
from metrics import *

from net import (RenderNet)

# Fix numpy's duplicated RNG issue
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class RenderSystem(LightningModule):
    def __init__(self, hparams):
        super(RenderSystem, self).__init__()
        self.hparams.update(vars(hparams))

        self.loss = loss_dict[hparams.loss_type]()

        self.models = RenderNet(hparams)

    def decode_batch(self, batch):
        target = dict(tgt_rgb=batch['tgt_rgb'].squeeze(1),
                tgt_mask=batch['tgt_mask'].squeeze(1))
        return batch, target

    def forward(self, inputs):
        return self.models(inputs)

    def prepare_data(self):
        # TODO separate train and validation dataset
        dataset = dataset_dict[self.hparams.dataset_name]
        self.train_dataset = dataset(self.hparams.root_dir, img_hw=self.hparams.img_hw,
                        num_frames=1, min_stride=25, max_stride=25)
        self.val_dataset = dataset(self.hparams.val_dir, img_hw=self.hparams.img_hw,
                        num_frames=1, min_stride=25, max_stride=25)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          drop_last=False,
                          pin_memory=True,
                          worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one datum at a time
                          pin_memory=True,
                          worker_init_fn=seed_worker)

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        inputs, target = self.decode_batch(batch)
        results = self(inputs)
        loss = self.loss(results, target)

        with torch.no_grad():
            psnr_ = psnr(results['rgb'], target['tgt_rgb'])
            log['train/psnr'] = psnr_

        self.log_dict(
            {
                'lr': get_learning_rate(self.optimizer),
                'train/loss': loss
            }
        )

        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        inputs, target = self.decode_batch(batch)
        results = self(inputs)

        log = {'val_loss': self.loss(results, target)}

        if batch_nb == 0 and not self.hparams.debug:
            img = torch.clip(results['rgb'].squeeze().cpu(), 0.0, 1.0)
            img_gt = target['tgt_rgb'].squeeze().cpu()
            alpha = torch.clip(results['alpha'].squeeze().cpu(), 0.0, 1.0)[None, :]
            alpha_gt = target['tgt_mask'].squeeze().cpu()[None, :]

            self.logger.experiment.add_image('val/pred', img)        
            self.logger.experiment.add_image('val/GT', img_gt)
            self.logger.experiment.add_image('val/alpha', alpha)
            self.logger.experiment.add_image('val/alpha_GT', alpha_gt)

        log['val_psnr'] = psnr(results['rgb'], target['tgt_rgb'])
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log_dict({'val/loss': mean_loss,
                       'val/psnr': mean_psnr}, prog_bar=True)

if __name__ == '__main__':
    hparams = get_opts()

    seed_everything(hparams.seed)

    timestamp = time.strftime("%m_%d_%H%M", time.localtime())
    
    system = RenderSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(hparams.ckpt_dir,
                                                               f'{hparams.exp_name}_{timestamp}'),
                                          filename='{epoch:d}',
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir=hparams.log_dir,
        name=f'{hparams.exp_name}_{timestamp}',
        debug=hparams.debug,
        create_git_tag=True
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      max_steps=hparams.num_steps,
                      callbacks=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      fast_dev_run=hparams.debug,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch=10,
                      benchmark=True)

    trainer.fit(system)