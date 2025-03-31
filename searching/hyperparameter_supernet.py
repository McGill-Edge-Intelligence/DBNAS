import math
import torch.nn as nn
import pytorch_lightning as pl
from ntpath import join
from datetime import datetime
import os
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.cloud_io import load as pl_load
from data_preprocess import get_dataloader, read_data, cfg, getSSTAccuracy
from DNA_teacherbert import BERT
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from best_student_block_baseline import BestStudentBlock
# from best_student_wo_att import BestStudentBlock

from ray import tune
from ray.tune import CLIReporter, progress_reporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

os.environ["SLURM_JOB_NAME"] = "bash"
fine_tune = True

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim  # can be added as hyperparameter to search in the future
num_labels = cfg.data.num_labels

batch_size = cfg.train.batch_size
max_epochs = cfg.train.max_epochs
train_df = read_data(cfg.data.train_path)

steps_per_epoch=len(train_df) // batch_size
total_training_steps = steps_per_epoch * max_epochs
warmup_steps = total_training_steps // 5

block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_1/time=23/07/2021 00:06:40/epoch=151-val_loss=0.03.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_2/time=23/07/2021 07:40:17/epoch=111-val_loss=0.04.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_3/time=23/07/2021 08:07:17/epoch=151-val_loss=0.07.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_4/time=23/07/2021 08:33:33/epoch=154-val_loss=0.11.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_5/time=23/07/2021 08:49:31/epoch=146-val_loss=0.11.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_6/time=23/07/2021 09:08:36/epoch=149-val_loss=0.04.ckpt'

teacher = BERT()

class BestStudentSupernet(pl.LightningModule):
    def __init__(self, config):
        super(BestStudentSupernet, self).__init__()
        self.lr = config['lr']
        # self.batch_size = config['batch_size']

        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        self.teacher = BERT()
        self.block1 = BestStudentBlock.load_from_checkpoint(block1_ckpt, block_num=1, teacher=teacher)
        self.block2 = BestStudentBlock.load_from_checkpoint(block2_ckpt, block_num=2, teacher=teacher)
        self.block3 = BestStudentBlock.load_from_checkpoint(block3_ckpt, block_num=3, teacher=teacher)
        self.block4 = BestStudentBlock.load_from_checkpoint(block4_ckpt, block_num=4, teacher=teacher)
        self.block5 = BestStudentBlock.load_from_checkpoint(block5_ckpt, block_num=5, teacher=teacher)
        self.block6 = BestStudentBlock.load_from_checkpoint(block6_ckpt, block_num=6, teacher=teacher)
        
        if fine_tune == False:
            self.block1.requires_grad_(False)
            self.block2.requires_grad_(False)
            self.block3.requires_grad_(False)
            self.block4.requires_grad_(False)
            self.block5.requires_grad_(False)
            self.block6.requires_grad_(False)

        self.classifier = nn.Linear(embed_dim, num_labels)
        self.dropout = nn.Dropout(0.5)
        self.criterion = BCEWithLogitsLoss()

    def forward(self,input):
        input = input.type(torch.FloatTensor).to(self.device)
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)    # bs, seq_len, embed_dim
        output = output[:, 0, :]  # bs, embed_dim
        output = self.dropout(output)
        logits = self.classifier(output) # bs, num_labels
        return logits.to(self.device)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        # word embedding
        input = teacher(input_ids, attention_mask, token_type_ids).hidden_states[0]
        output = self(input)
        loss = self.criterion(output, labels.view(-1, 1))
        accuracy = getSSTAccuracy(output, labels.view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
     
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        # word embedding
        input = teacher(input_ids, attention_mask, token_type_ids).hidden_states[0]
        output = self(input)
        loss = self.criterion(output, labels.view(-1, 1))
        accuracy = getSSTAccuracy(output, labels.view(-1, 1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss
    '''
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
    '''
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer


def train_hyper_search(config, num_epochs=10, num_gpus=1):
    train_dataloader = get_dataloader('train', cfg.data.train_path, batch_size)
    val_dataloader = get_dataloader('val', cfg.data.val_path, batch_size)

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    pl.seed_everything(1234)
    model = BestStudentSupernet(config)
    metrics = {'loss': 'val_loss', 'acc': 'val_accuracy'}

    # saves checkpoint at the minimum validation loss
    # checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_hyper/best_parameters/time={}'.format(now), filename='{epoch}-{val_loss:.2f}')

    hyper_finetune_callback = TuneReportCallback(metrics, on='validation_end')
    # hyper_ckpt_callback = TuneReportCheckpointCallback(metrics, filename="checkpoint_hyper/best_parameters", on="validation_end")
    
    # logger
    # wandb_logger = WandbLogger(name='fine_tune:{}-{}'.format(fine_tune, now), project='BERT_NAS_hyperparameter')
    wandb_logger = WandbLogger(name='{}'.format(now), project='BERT_NAS_hyper_2layers')

    # trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback, hyper_finetune_callback], gpus=num_gpus, num_nodes=1, max_epochs=num_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer = Trainer(callbacks=hyper_finetune_callback, gpus=num_gpus, num_nodes=1, max_epochs=num_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)

    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.finish()


def hyper_search_asha(num_samples=10, num_epochs=10, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-5, 1e-1)
        # "batch_size": tune.choice([32, 64, 128])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=['lr'],
        metric_columns=['loss', 'acc', 'training_iteration']
    )

    analysis = tune.run(
        tune.with_parameters(
            train_hyper_search,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            'cpu': 6,
            'gpu': gpus_per_trial
        },
        fail_fast=True,
        metric='loss',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='hyper_search_asha'
    )

    '''
        analysis = tune.run(
        tune.with_parameters(
            train_hyper_search,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            'cpu': 6,
            'gpu': gpus_per_trial
        },
        fail_fast=True,
        metric='loss',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        checkpoint_score_attr="acc",
        checkpoint_freq=1,
        name='hyper_search_asha'
    )
    '''

    print('Best hyperparameters found were:', analysis.best_config)
    best_trial = analysis.get_best_trial(metric="acc", mode="max", scope="all")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="acc")
    print(best_trial)


if __name__ == "__main__":
    hyper_search_asha()