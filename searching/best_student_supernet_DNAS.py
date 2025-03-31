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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from data_preprocess import get_dataloader, read_data, cfg, getSSTAccuracy
from DNA_teacherbert import BERT
from best_student_block_DNAS import BestStudentBlock

fine_tune = True

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim
num_labels = cfg.data.num_labels

batch_size = cfg.train.batch_size
max_epochs = 40
train_df = read_data(cfg.data.train_path)

steps_per_epoch=len(train_df) // batch_size
total_training_steps = steps_per_epoch * max_epochs
warmup_steps = total_training_steps // 5

'''
# fine-tuning using AdamW optimizer, lr = 1e-5, StepLR scheduler
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_1/time=28/10/2021 14:43:25/epoch=78-val_loss=0.07.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_2/time=28/10/2021 15:59:15/epoch=82-val_loss=0.06.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_3/time=28/10/2021 17:09:04/epoch=81-val_loss=0.09.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_4/time=28/10/2021 19:15:26/epoch=82-val_loss=0.14.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_5/time=28/10/2021 19:31:25/epoch=84-val_loss=0.14.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_6/time=28/10/2021 19:41:45/epoch=1-val_loss=0.68.ckpt'
# block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_AdamW_block_6/time=28/10/2021 20:17:18/epoch=81-val_loss=0.05.ckpt'
'''

# Cell 1 - 2 layers, lr = 1e-3
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_1/time=12/09/2021 09:35:32/epoch=28-val_loss=0.03.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_2/time=12/09/2021 09:42:13/epoch=28-val_loss=0.04.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_3/time=12/09/2021 09:48:06/epoch=27-val_loss=0.07.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_4/time=12/09/2021 09:54:20/epoch=28-val_loss=0.10.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_5/time=12/09/2021 10:01:30/epoch=28-val_loss=0.11.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=12/09/2021 10:06:32/epoch=1-val_loss=0.93.ckpt'
# block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=23/10/2021 21:58:46/epoch=43-val_loss=0.04.ckpt'

'''
# Cell 2 - 3 layers
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_1/time=12/09/2021 10:21:18/epoch=2-val_loss=2.04.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_2/time=12/09/2021 10:26:17/epoch=28-val_loss=0.04.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_3/time=12/09/2021 10:31:14/epoch=28-val_loss=0.07.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_4/time=12/09/2021 10:33:31/epoch=29-val_loss=0.11.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_5/time=12/09/2021 10:37:54/epoch=28-val_loss=0.12.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=12/09/2021 10:43:14/epoch=4-val_loss=1.84.ckpt'

# Cell 3 - 4 layers
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_1/time=12/09/2021 10:49:23/epoch=3-val_loss=6.78.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_2/time=12/09/2021 10:55:18/epoch=2-val_loss=0.05.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_3/time=12/09/2021 11:00:22/epoch=1-val_loss=0.10.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_4/time=12/09/2021 11:13:43/epoch=2-val_loss=0.14.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_5/time=12/09/2021 11:19:16/epoch=2-val_loss=0.14.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=12/09/2021 11:23:28/epoch=1-val_loss=8.31.ckpt'

# Cell 1 - 2 layers with augmented data
block1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_1/time=12/09/2021 19:10:35/epoch=14-val_loss=0.03.ckpt'
block2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_2/time=12/09/2021 19:16:00/epoch=11-val_loss=0.04.ckpt'
block3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_3/time=12/09/2021 19:36:31/epoch=11-val_loss=0.06.ckpt'
block4_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_4/time=12/09/2021 20:13:26/epoch=11-val_loss=0.11.ckpt'
block5_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_5/time=12/09/2021 20:38:58/epoch=12-val_loss=0.11.ckpt'
# block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=12/09/2021 21:10:08/epoch=0-val_loss=3.94.ckpt'
block6_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_block/DNAS_block_6/time=12/09/2021 10:06:32/epoch=1-val_loss=0.93.ckpt'
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = BERT().to(device)

class BestStudentSupernet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.teacher = BERT()
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        self.block1 = BestStudentBlock.load_from_checkpoint(block1_ckpt, block_num=1)
        self.block2 = BestStudentBlock.load_from_checkpoint(block2_ckpt, block_num=2)
        self.block3 = BestStudentBlock.load_from_checkpoint(block3_ckpt, block_num=3)
        self.block4 = BestStudentBlock.load_from_checkpoint(block4_ckpt, block_num=4)
        self.block5 = BestStudentBlock.load_from_checkpoint(block5_ckpt, block_num=5)
        self.block6 = BestStudentBlock.load_from_checkpoint(block6_ckpt, block_num=6)
        
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
        output = self.block6(output) # bs, seq_len, embed_dim
        output = output[:,0,:] # bs, embed_dim
        output = self.dropout(output)
        logits = self.classifier(output) # bs, num_labels
        return logits.to(self.device)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        # word embedding
        input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[0]
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
        input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[0]
        output = self(input)
        loss = self.criterion(output, labels.view(-1, 1))
        accuracy = getSSTAccuracy(output, labels.view(-1, 1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps
        )
        return ([optimizer], [scheduler])


if __name__ == "__main__":
    # early stopping: if val_loss doesn't decrease for 3 epochs, stop the training
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
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

    #logger
    wandb_logger = WandbLogger(name='fine_tune:{}-{}_finished_KD_version'.format(fine_tune, now), project='DNAS_supernet')
    # wandb_logger = WandbLogger(name='fine_tune:{}-{}_AdamW_supernet'.format(fine_tune, now), project='DNAS_AdamW_supernet')
    
    # saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_DNAS_supernet/time={}'.format(now), filename='{epoch}-{val_loss:.2f}')
    # checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_DNAS_supernet/AdamW_supernet/time={}'.format(now), filename='{epoch}-{val_loss:.2f}')
    
    # trainer = Trainer(accelerator='ddp',callbacks=[checkpoint_callback,early_stop_callback,lr_monitor],gpus=-1,num_nodes=1,max_epochs=max_epochs,progress_bar_refresh_rate=30,logger=wandb_logger)
    trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.finish()