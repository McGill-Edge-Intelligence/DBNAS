from ntpath import join
import hydra
from datetime import datetime
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import trainer
import torch
# from operations import uniform_random_op_encoding, OPS, ops_dict, conv_dict, pool_dict
from operations_base import uniform_random_op_encoding, OPS, ops_dict, conv_dict, pool_dict
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.experimental import compose, initialize
import torch.nn as nn
from data_preprocess import get_dataloader, read_data, cfg
from torch.optim import Adam
from pytorch_lightning import Trainer
from DNA_teacherbert import BERT
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger


seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim

'''
Operations: one single operation per cell, selection among convolution, pooling and attention.
There are 6 blocks in total and each block has three cells. The first cell has maximum number of layers as 2, 
second layer has max layers of 3 and the thrid cell has max layers of 4.
''' 
block_config = {
    0: [2, 3, 4],
    1: [2, 3, 4],
    2: [2, 3, 4],
    3: [2, 3, 4],
    4: [2, 3, 4],
    5: [2, 3, 4],
}


class StudentCell(pl.LightningModule):
    def __init__(self, cell, block_num, cell_num, teacher):
        super().__init__()
        
        self.block_num = block_num
        self.cell_num = cell_num
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(seq_len, affine=True)
        self.teacher = teacher
        
        #randomly sampled operations
        self.cell = cell

        #construct the model as a module list
        self.model = nn.ModuleList()
        for layer in self.cell:
            self.model.append(OPS[layer](embed_dim, seq_len))


    # input has shape [bs, seq_len, embed_dim]
    def forward(self, input):
        input = input.type(torch.FloatTensor).to(self.device)
        outputs = []
        
        for idx,layer in enumerate(self.model):
            if idx == 0:  
                if layer.__class__.__name__ == 'MultiheadAttention':
                    # transpose the input so that it becomes batchsize, seq_len,embed_dim
                    input = torch.transpose(input, 0, 1)
                    output = torch.transpose(layer(input, input, input)[0], 0, 1)
                    outputs.append(output)
                # sum the output of each direction
                elif layer.__class__.__name__ == 'GRU':
                    output = layer(input)[0][:, :, :embed_dim] + layer(input)[0][:, :, embed_dim:]
                    outputs.append(output)
                # add relu before conv and batch norm after conv
                elif layer.__class__.__name__ == 'Conv1d':
                    output = self.relu(input)
                    output = layer(output)
                    output = self.batchnorm(output)
                    outputs.append(output)
                else:
                    output = layer(input)
                    outputs.append(output)
            else:
                # set input to previous layer's output 
                input = outputs[idx - 1]
                
                #add skip connection
                if idx >= 2:
                    input = input + outputs[idx-2]
                
                if layer.__class__.__name__ == 'MultiheadAttention':
                    #transpose the input so that it becomes batchsize,seq_len,embed_dim
                    input = torch.transpose(input, 0, 1)
                    output = torch.transpose(layer(input, input, input)[0], 0, 1)
                    outputs.append(output)
                #sum the output of each direction
                elif layer.__class__.__name__ == 'GRU':
                    output = layer(input)[0][:, :, :embed_dim] + layer(input)[0][:, :, embed_dim:]
                    outputs.append(output)
                #add relu before conv and bath norm after conv
                elif layer.__class__.__name__ == 'Conv1d':
                    output = self.relu(input)
                    output = layer(output)
                    output = self.batchnorm(output)
                    outputs.append(output)
                else:
                    output = layer(input)
                    outputs.append(output)
        
        #return output of the last layer
        return outputs[-1].to('cuda')
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        '''
        if cfg.train.use_label:
            labels = batch['labels']
        else:
            labels = None
        '''
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
        '''
        if cfg.train.use_label:
            labels = batch['labels']
        else:
            labels = None
        '''
        # the counterparts for one block in student are 2 encoder layers in teacher bert
        teacher_output = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = self.teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
        loss = F.mse_loss(student_output, teacher_output)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
    
'''
generate and save the operations encodings for each cell
'''
def generate_cell_encodings(num_blocks, num_cells, num_ops):
    for b in range(num_blocks):
        for c in range(num_cells):
            cell_encoding = uniform_random_op_encoding(num_ops, block_config[b][c])
            saved_dir = os.path.join("searching/cell_encodings", "block_{}_cell_{}.txt".format(b + 1, c + 1))

            with open(saved_dir, 'w') as f:
                for encoding in cell_encoding:
                    op_decode = ops_dict[encoding]
                    if op_decode == 'pool':
                        op_decode = pool_dict[uniform_random_op_encoding(len(pool_dict))[0]]
                    elif op_decode == 'conv':
                        op_decode = conv_dict[uniform_random_op_encoding(len(conv_dict))[0]]
                    f.write('%s\n' % op_decode)


'''
returns a list of operations that are randomly selected from the search space
'''
def sample_ops(block_num,cell_num):
    max_layers = block_config[block_num - 1][cell_num - 1]
    op_encodings = uniform_random_op_encoding(len(ops_dict), max_layers)
    op_decodes = []
    for encoding in op_encodings:
        op_decode = ops_dict[encoding]
        if op_decode == 'pool':
            op_decode = pool_dict[uniform_random_op_encoding(len(pool_dict))[0]]
        elif op_decode == 'conv':
            op_decode = conv_dict[uniform_random_op_encoding(len(conv_dict))[0]]
        op_decodes.append(op_decode)
    return op_decodes


if __name__ == "__main__":
    train_df = read_data(cfg.data.train_path)
    train_dataloader = get_dataloader('train', cfg.data.train_path, 64)
    val_dataloader = get_dataloader('val', cfg.data.val_path, 64)

    batch_size = cfg.train.batch_size
    max_epochs = cfg.train.max_epochs
    
    # block and cell to be run
    block_num = cfg.train.block_num
    cell_num = cfg.train.cell_num
    
    teacher = BERT()
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    current_time = datetime.now().strftime("%H:%M:%S")
    cell = sample_ops(block_num, cell_num)
    print(cell)
    # set the seed for reproducibility
    pl.seed_everything(1234)
    model = StudentCell(cell, block_num, cell_num, teacher)
    saved_dir = os.path.join("searching/cell_encodings", "{}_block_{}_cell_{}.txt".format(current_time, block_num, cell_num))
        
    # save the randomly generated operations in a txt file
    with open(saved_dir, 'w') as f:
        for op in model.cell:
            f.write('%s\n' % op)
    
    # logger
    
    wandb_logger = WandbLogger(name='{}-block-{}-cell-{}'.format(now, model.block_num, model.cell_num), project='BERT_NAS_block_{}_cell_{}'.format(block_num, cell_num))
    wandb_logger.experiment.save(saved_dir)
    
    #saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint/block_{}_cell_{}/time={}'.format(model.block_num, model.cell_num, now), filename='{epoch}-{val_loss:.2f}')
    
    trainer = Trainer(callbacks=[checkpoint_callback], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)