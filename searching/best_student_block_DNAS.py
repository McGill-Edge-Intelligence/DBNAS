import torch.nn as nn
from ntpath import join
from datetime import datetime
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from transformers import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import utils
from operations import OPS
from data_preprocess import get_dataloader, cfg
from DNA_teacherbert import BERT


seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim

# best ops in each block
# cell 1 - 2 layers in each cell
block_1 = ['conv_std_7', 'multihead_attention_8']
block_2 = ['pool_avg_3', 'multihead_attention_8']
block_3 = ['pool_avg_3', 'multihead_attention_8']
block_4 = ['pool_avg_3', 'multihead_attention_8']
block_5 = ['pool_avg_3', 'multihead_attention_8']
block_6 = ['conv_std_1', 'conv_dila_5']
# block_6 = ['conv_std_7', 'multihead_attention_8']
# block_6 = ['pool_avg_3', 'multihead_attention_8']
'''
# cell 2 - 3 layers in each cell
block_1 = ['conv_dila_7', 'conv_std_5', 'conv_std_5']
block_2 = ['conv_dila_5', 'pool_avg_3', 'multihead_attention_4']
block_3 = ['conv_dila_5', 'pool_avg_3', 'multihead_attention_4']
block_4 = ['conv_dila_7', 'pool_avg_3', 'multihead_attention_4']
block_5 = ['conv_dila_5', 'pool_avg_3', 'multihead_attention_4']
block_6 = ['conv_dila_1', 'conv_dila_1', 'conv_dila_3']

# cell 3 - 4 layers in each cell
block_1 = ['conv_std_7', 'conv_std_3', 'conv_dila_3', 'conv_std_5']
block_2 = ['conv_dila_7', 'conv_std_3', 'pool_avg_3', 'multihead_attention_12']
block_3 = ['conv_std_1', 'conv_std_3', 'pool_avg_3', 'multihead_attention_12']
block_4 = ['conv_dila_1', 'conv_std_3', 'pool_avg_3', 'multihead_attention_12']
block_5 = ['conv_std_1', 'conv_std_3', 'pool_avg_3', 'multihead_attention_12']
block_6 = ['conv_std_1', 'conv_dila_1', 'conv_dila_3', 'conv_std_3']
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = BERT().to(device)

class BestStudentBlock(pl.LightningModule):
    def __init__(self, block_num):
        super().__init__()
        self.block_num = block_num
        self.teacher = BERT()
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(seq_len, affine=True)
        
        # select the block
        if block_num == 1:
            self.ops = block_1
        elif block_num == 2: 
            self.ops = block_2
        elif block_num == 3: 
            self.ops = block_3
        elif block_num == 4:
            self.ops = block_4
        elif block_num == 5:
            self.ops = block_5
        elif block_num == 6:
            self.ops = block_6
        
        self.model = nn.ModuleList()
        for op in self.ops:
            self.model.append(OPS[op](embed_dim, seq_len))

    # input has shape [bs, seq_len, embed_dim]
    def forward(self, input):
        input = input.type(torch.FloatTensor).to(self.device)
        output = input
        
        for idx, layer in enumerate(self.model):
            if idx == len(self.model)-1:
               output = output + input 
            if layer.__class__.__name__ == 'MultiheadAttention':
                # transpose the input so that it becomes [bs, seq_len, embed_dim]
                output = torch.transpose(output, 0, 1)
                output = torch.transpose(layer(output, output, output)[0], 0, 1)
            elif layer.__class__.__name__ == 'Conv1d':
                output = layer(output)
                output = self.relu(output)
                output = self.batchnorm(output)
            else:
                output = layer(output)

        return output.to(self.device)
    
    def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

            # 2 layers of teacher is one block, used to guide one stage training of the student
            teacher_output = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
            student_input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
            student_output = self(student_input)

            loss = F.mse_loss(student_output, teacher_output)
            self.log('train_loss', loss, on_step= True, on_epoch=True, prog_bar=True, logger=True)
            return loss
     
    def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']

            # 2 layers of teacher is one block, used to guide one stage training of the student
            teacher_output = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
            student_input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
            student_output = self(student_input)

            loss = F.mse_loss(student_output, teacher_output)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        lr_scheduler = {'scheduler':ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=3, verbose=True), 'monitor':'val_loss'}
        # optimizer = AdamW(self.parameters(), lr=1e-5)
        # lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return ([optimizer], [lr_scheduler])


if __name__ == "__main__":
    block_num = 1

    batch_size = cfg.train.batch_size
    max_epochs = cfg.train.max_epochs

    # early stopping: if val_loss doesn't decrease for 3 epochs, stop the training
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    #log lr every epoch
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    train_dataloader = get_dataloader('train', cfg.data.train_path, batch_size)
    val_dataloader = get_dataloader('val', cfg.data.val_path, batch_size)
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    current_time = datetime.now().strftime("%H:%M:%S")

    #set the seed for reproducibility
    pl.seed_everything(1234)
    model = BestStudentBlock(block_num=block_num)

    #logger
    wandb_logger = WandbLogger(name='{}-block-{}'.format(now, model.block_num), project='DNAS_block_{}'.format(block_num))
    # wandb_logger = WandbLogger(name='{}-block-{}'.format(now, model.block_num), project='DNAS_AdamW_block')
    
    #saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_DNAS_block/DNAS_block_{}/time={}'.format(model.block_num, now), filename='{epoch}-{val_loss:.2f}')
    # checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_DNAS_block/DNAS_AdamW_block_{}/time={}'.format(model.block_num, now), filename='{epoch}-{val_loss:.2f}')
    
    # trainer = Trainer(accelerator='ddp', callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.finish()