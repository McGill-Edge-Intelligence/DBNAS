import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import os
import numpy as np
from ntpath import join
import hydra
from datetime import datetime
from omegaconf import DictConfig
from sklearn.utils import shuffle
from hydra.experimental import compose, initialize
from scipy.special import softmax
from thop import profile
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb

import pytorch_lightning as pl
from pytorch_lightning import trainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from operations import OPS, MixedOpsDNAS, one_op_model
from data_preprocess import get_dataloader, read_data, cfg, SSTDataset
from DNA_teacherbert import BERT
# from flops_counter import get_model_complexity_info

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = BERT().to(device)


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
    5: [2, 3, 4]
}


'''
return FLOPS for attention layer
'''
def multihead_attention_flops(multihead_attention_module, input):
    flops = 0
    q, k, v = input, input, input
    batch_size = q.shape[1]

    num_heads = multihead_attention_module.num_heads
    embed_dim = multihead_attention_module.embed_dim
    kdim = multihead_attention_module.kdim
    vdim = multihead_attention_module.vdim
    if kdim is None:
        kdim = embed_dim
    if vdim is None:
        vdim = embed_dim

    # initial projections
    flops = q.shape[0] * q.shape[2] * embed_dim + \
        k.shape[0] * k.shape[2] * kdim + \
        v.shape[0] * v.shape[2] * vdim
    if multihead_attention_module.in_proj_bias is not None:
        flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

    # attention heads: scale, matmul, softmax, matmul
    head_dim = embed_dim // num_heads
    head_flops = q.shape[0] * head_dim + \
        head_dim * q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] * head_dim

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += q.shape[0] * embed_dim * (embed_dim + 1)

    flops *= batch_size
    return flops


'''
return the number of parameters of a given model
'''
def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


'''
ptflops launches a given model on a random tensor and estimates amount of computations during inference.
Complicated models can have several inputs, some of them could be optional.
To construct non-trivial input one can use the input_constructor argument of the get_model_complexity_info.
input_constructor is a function that takes the input spatial resolution as a tuple and returns a dict with named input arguments of the model.
Next this dict would be passed to the model as keyworded arguments.
'''
def pool_input(input_res):
    return {'x': torch.ones(()).new_empty((1, *input_res), device=device)}


'''
return a dictionary that contains the FLOPS and number of parameters of all the operations in the search space
'''
def get_FLOPS_params():
    input = torch.randn(1, seq_len, embed_dim)
    op_flops = []
    op_params = []
    op_cost = {}
    for op in OPS.keys():
        model = one_op_model(op)
        if 'attention' in op:
            flops = multihead_attention_flops(OPS[op](embed_dim, seq_len), input)
            # flops = multihead_attention_flops(OPS[op](embed_dim, seq_len, stride=1, affine=False), input)
            params = get_num_params(model)
        else:
            flops, params = profile(model, inputs=(input, ))
        op_flops.append(flops/100000000)
        op_params.append(params/1000000)

    op_flops = torch.softmax(torch.DoubleTensor(op_flops), dim=0)
    op_params = torch.softmax(torch.DoubleTensor(op_params), dim=0)
    
    for flop, param, op in zip(op_flops, op_params, list(OPS.keys())):
        op_cost[op] = (flop.item() + param.item())/2

    return op_cost


op_cost = get_FLOPS_params()


class StudentCell(pl.LightningModule):
    def __init__(self, block_num, cell_num, beta=0.5):
        super().__init__()
        
        self.block_num = block_num
        self.cell_num = cell_num
        # self.teacher = teacher
        self.beta = beta
        self.num_layers = block_config[block_num-1][cell_num-1]
        self.cell_supernet = nn.ModuleList()
        
        # construct the model as a module list
        for layer_num in range(self.num_layers):
            layer = MixedOpsDNAS()
            self.cell_supernet.append(layer)

        # activate manual optimization
        self.automatic_optimization = False

     # input has shape [bs, seq_len, embed_dim]
    def forward(self, input):
        input = input.type(torch.FloatTensor).to(self.device)
        # input = input.type(torch.FloatTensor)
        output = input
    
        for idx, layer in enumerate(self.cell_supernet):
            # add skip connection from the input to the last layer
            if idx == len(self.cell_supernet) - 1: 
                output = layer(output + input)
            else:
                output = layer(output)
                        
        return output.to(self.device)

    def configure_optimizers(self):
        thetas_params = [param for name, param in self.named_parameters() if 'thetas' in name]
        w_params = [param for name, param in self.named_parameters() if 'thetas' not in name]
        w_optimizer = Adam(w_params, lr=1e-4)
        theta_optimizer = Adam(thetas_params, lr=1e-4)
        return w_optimizer, theta_optimizer

    def training_step(self, batch, batch_idx):
        w_optimizer, theta_optimizer = self.optimizers()
        
        ##########
        #update w#
        ##########
        input_ids = batch['w']['input_ids']
        attention_mask = batch['w']['attention_mask']
        token_type_ids = batch['w']['token_type_ids']
        
        # the counterparts for one block in student are 2 encoder layers in teacher bert
        teacher_output = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
            
        kd_loss = F.mse_loss(student_output, teacher_output)
        efficiency_cost = EfficiencyLoss()
        e_loss = 0.0
        for layer in self.cell_supernet:
            e_loss += efficiency_cost(layer.thetas)
        loss = self.beta * kd_loss + (1 - self.beta) * e_loss
        w_optimizer.zero_grad()
        self.manual_backward(loss)
        w_optimizer.step()

        ##############
        #update theta#
        ##############
        input_ids = batch['theta']['input_ids']
        attention_mask = batch['theta']['attention_mask']
        token_type_ids = batch['theta']['token_type_ids']
        
        # the counterparts for one block in student are 2 encoder layers in teacher bert
        teacher_output = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
            
        kd_loss = F.mse_loss(student_output, teacher_output)
        efficiency_cost = EfficiencyLoss()
        e_loss = 0.0
        for layer in self.cell_supernet:
            e_loss += efficiency_cost(layer.thetas)
        loss = self.beta * kd_loss + (1 - self.beta) * e_loss
        theta_optimizer.zero_grad()
        self.manual_backward(loss)
        theta_optimizer.step()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
       
        # the counterparts for one block in student are 2 encoder layers in teacher bert
        teacher_output = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2]
        student_input = teacher(input_ids, attention_mask, token_type_ids).hidden_states[self.block_num * 2 - 2]
        student_output = self(student_input)
        
        kd_loss = F.mse_loss(student_output, teacher_output)
        efficiency_cost = EfficiencyLoss()
        e_loss = 0.0
        for layer in self.cell_supernet:
            e_loss += efficiency_cost(layer.thetas)
        loss = self.beta * kd_loss + (1 - self.beta) * e_loss

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


class EfficiencyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, thetas):
        alphas = nn.functional.gumbel_softmax(thetas, tau=0.2)
        e_loss = torch.sum(torch.dot(alphas, torch.FloatTensor(list(op_cost.values())).to(device)))
        # e_loss = torch.sum(torch.dot(alphas, torch.FloatTensor(list(op_cost.values()))))
        return e_loss


if __name__ == "__main__":
    batch_size = cfg.train.batch_size
    max_epochs = cfg.train.max_epochs
    
    train_df = read_data(cfg.data.train_path)
    train_w_df, train_theta_df = train_test_split(train_df, test_size=0.2, shuffle=True)
    train_w_dataset = SSTDataset(train_w_df)
    train_theta_dataset = SSTDataset(train_theta_df)
    train_w_dl = DataLoader(train_w_dataset, batch_size, shuffle=True, num_workers=2)
    train_theta_dl = DataLoader(train_theta_dataset, batch_size, shuffle=True, num_workers=2)
    val_dataloader = get_dataloader('val', cfg.data.val_path, batch_size)
    train_dataloader = {'w':train_w_dl, 'theta':train_theta_dl}
    
    #block and cell to be run
    block_num = cfg.train.block_num
    cell_num = cfg.train.cell_num
    
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    current_time = datetime.now().strftime("%H:%M:%S")

    # set the seed for reproducibility
    pl.seed_everything(1234)
    model = StudentCell(block_num, cell_num)
           
    # logger
    wandb_logger = WandbLogger(name='{}-block-{}-cell-{}'.format(now, model.block_num, model.cell_num), project='DNAS_block_{}_cell_{}_right'.format(block_num, cell_num))
    
    #saves checkpoint at the minimum validation loss
    checkpoint_callback = ModelCheckpoint(mode='min', monitor='val_loss', dirpath='checkpoint_DNAS_right/DNAS_block_{}_cell_{}/time={}'.format(model.block_num, model.cell_num, now), filename='{epoch}-{val_loss:.2f}')
    
    # trainer = Trainer(accelerator='ddp', callbacks=[checkpoint_callback], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer = Trainer(callbacks=[checkpoint_callback], gpus=-1, num_nodes=1, max_epochs=max_epochs, progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    wandb_logger.experiment.finish()