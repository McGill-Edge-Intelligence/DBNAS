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
from torch.optim import Adam
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_preprocess import get_dataloader, read_data, cfg, getSSTAccuracy
from DNA_teacherbert import BERT
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from best_student_block_baseline import BestStudentBlock
# from best_student_wo_att import BestStudentBlock

fine_tune = True

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim
num_labels = cfg.data.num_labels

batch_size = cfg.train.batch_size
max_epochs = cfg.train.max_epochs
train_df = read_data(cfg.data.train_path)

steps_per_epoch=len(train_df) // batch_size
total_training_steps = steps_per_epoch * max_epochs
warmup_steps = total_training_steps // 5

'''
# 2 layers in each block
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_1/time=23/07/2021 00:06:40/epoch=151-val_loss=0.03.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_2/time=23/07/2021 07:40:17/epoch=111-val_loss=0.04.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_3/time=23/07/2021 08:07:17/epoch=151-val_loss=0.07.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_4/time=23/07/2021 08:33:33/epoch=154-val_loss=0.11.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_5/time=23/07/2021 08:49:31/epoch=146-val_loss=0.11.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_6/time=23/07/2021 09:08:36/epoch=149-val_loss=0.04.ckpt'
'''
'''
# 3 layers in each block
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_1/time=23/07/2021 11:38:34/epoch=149-val_loss=0.02.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_2/time=23/07/2021 11:55:00/epoch=102-val_loss=0.06.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_3/time=23/07/2021 12:40:20/epoch=143-val_loss=0.08.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_4/time=23/07/2021 13:09:28/epoch=138-val_loss=0.10.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_5/time=23/07/2021 13:29:26/epoch=153-val_loss=0.13.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_6/time=23/07/2021 14:12:05/epoch=152-val_loss=0.04.ckpt'
'''

# 4 layers in each block
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_1/time=24/07/2021 08:42:01/epoch=144-val_loss=0.02.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_2/time=24/07/2021 10:26:08/epoch=151-val_loss=0.05.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_3/time=24/07/2021 10:59:53/epoch=151-val_loss=0.07.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_4/time=24/07/2021 11:16:56/epoch=147-val_loss=0.11.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_5/time=24/07/2021 11:49:47/epoch=146-val_loss=0.12.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_block/block_6/time=24/07/2021 12:10:41/epoch=151-val_loss=0.04.ckpt'


teacher = BERT()

class BestStudentSupernet(pl.LightningModule):
    def __init__(self):
        super().__init__()
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

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps
        )
        return ([optimizer], [scheduler])


if __name__ == "__main__":
    batch_size = cfg.train.batch_size

    # early stopping: if val_loss doesn't decrease for 10 epochs, stop the training
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )
    # log lr every epoch
    lr_monitor = LearningRateMonitor(logging_interval='step')

    train_dataloader = get_dataloader('train', cfg.data.train_path, batch_size)
    val_dataloader = get_dataloader('val', cfg.data.val_path, batch_size)
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    current_time = datetime.now().strftime("%H:%M:%S")

    # set the seed for reproducibility
    pl.seed_everything(1234)
    model = BestStudentSupernet()

    # logger
    wandb_logger = WandbLogger(name='fine_tune:{}-{}'.format(fine_tune, now), project='BERT_NAS_Supernet_Baseline')
    
    # saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_supernet/best_supernet/time={}'.format(now), filename='{epoch}-{val_loss:.2f}')
    
    trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.finish()