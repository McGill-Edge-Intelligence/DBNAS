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
from operations_sequential import uniform_random_op_encoding, OPS, ops_dict, conv_dict, pool_dict
from data_preprocess import get_dataloader, read_data, cfg
from DNA_teacherbert import BERT

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim


# experiment 1: the operations that give the lowest val loss in each block, some blocks may have multiple optimum choices
block_1 = ['conv_std_7', 'conv_dila_1']
block_2 = ['conv_std_3', 'conv_dila_5']
# block_2 = ['conv_std_5', 'pooling_ave_3', 'conv_dila_1']
block_3 = ['conv_std_1', 'conv_std_1']
block_4 = ['conv_dila_1', 'conv_std_3']
block_5 = ['conv_dila_1', 'conv_std_7']
block_6 = ['conv_std_3', 'conv_dila_5']
# block_6 = ['conv_dila_7', 'conv_std_5']


class BestStudentBlock(pl.LightningModule):
    def __init__(self, block_num, teacher):
        super().__init__()
        self.block_num = block_num
        self.teacher = teacher
        
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
            self.model.append(OPS[op](embed_dim, seq_len, stride=1, affine=False))

     # input has shape [bs, seq_len, embed_dim]
    def forward(self, input):
        input = input.type(torch.FloatTensor).to(self.device)
        outputs = []
        
        for idx,layer in enumerate(self.model):
            if idx == 0:
                output = layer(input)
                outputs.append(output)
            else:
                # set input to previous layer's output 
                input = outputs[idx - 1]
                
                # add skip connection
                if idx >= 2:
                        input = input + outputs[idx - 2]

                output = layer(input)
                outputs.append(output)

        # return output of the last layer
        return outputs[-1].to('cuda')
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        # 2 layers of teacher is one block, used to guide one stage training of the student
        teacher_output = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
        loss = F.mse_loss(student_output, teacher_output)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
     
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        # the counterparts for one block in student are 2 encoder layers in teacher bert
        teacher_output = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
        loss = F.mse_loss(student_output, teacher_output)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-5)
        lr_schedular = {'scheduler':ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=6, verbose=True), 'monitor':'val_loss'}
        return ([optimizer], [lr_schedular])


if __name__ == "__main__":
    block_num = 6
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
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    train_df = read_data(cfg.data.train_path)
    train_dataloader = get_dataloader('train', cfg.data.train_path, batch_size)
    val_dataloader = get_dataloader('val', cfg.data.val_path, batch_size)
    
    teacher = BERT()
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    current_time = datetime.now().strftime("%H:%M:%S")

    # set the seed for reproducibility
    pl.seed_everything(1234)
    model = BestStudentBlock(block_num=block_num, teacher=teacher)

    # logger
    wandb_logger = WandbLogger(name='{}-block-{}'.format(now, model.block_num), project='BERT_NAS_block_{}'.format(block_num))
    
    # saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_block/block_{}/time={}'.format(model.block_num, now), filename='{epoch}-{val_loss:.2f}')
    
    trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], gpus=-1, num_nodes=1, max_epochs=200, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.finish()

    # model = BestStudentBlock.load_from_checkpoint(checkpoint_path, block_num=block_num, teacher=teacher)