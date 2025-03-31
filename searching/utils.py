import torch
import torch.nn as nn
from torch.nn import parameter
import pytorch_lightning as pl
import numpy as np
import time
from scipy.special import softmax
from data_preprocess import cfg, get_dataloader, read_data
from best_student_supernet_baseline import BestStudentSupernet
from DNA_teacherbert import BERT
from operations import OPS
from DNA_teacherbert import BERT
from DNA_studentbertdiff import StudentCell

random_best_student_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_supernet/best_supernet/time=25/07'

seq_len = cfg.data.seq_len
embed_dim = cfg.data.embed_dim
batch_size = cfg.train.batch_size


'''
return the function to get the time for a model to do inference
'''
def get_inference_time(model, dataloader):
    teacher = BERT()
    model.cuda()
    model.eval()
    ttime_elapsed = 0
    for step, batch in enumerate(dataloader):
        input = teacher(batch['input_ids'], batch['attention_mask'], batch['token_type_ids']).hidden_states[0]
        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        output = model(input)
        torch.cuda.synchronize()
        ttime_elapsed = ttime_elapsed + int(round(time.time() * 1000)) - tsince
    print ('inference time elapsed {}ms'.format(ttime_elapsed))


'''
return the best model in DNAS (the searched model with highest weight is the best)
'''
def get_best_model_from_DNAS(checkpoint_path, block_num, cell_num, hard_sampling=True):
    teacher = BERT()
    best_model = StudentCell.load_from_checkpoint(checkpoint_path, block_num=block_num, cell_num=cell_num, teacher=teacher, beta=0.5)
    best_ops = []
    ops = list(OPS.keys())
    for layer in best_model.cell_supernet:
        thetas = [param for name, param in layer.named_parameters() if 'thetas' in name][0]
        if hard_sampling == True:
            best_ops.append(ops[np.argmax(thetas.detach().cpu().numpy())])
        else: 
            distribution = softmax(thetas.detach().numpy())
            best_ops.append(ops[np.random.choice(np.linspace(0, len(ops)-1, len(ops), dtype=int), p=distribution)])
    return best_ops


if __name__ == "__main__":
    '''
    val_dataloader = get_dataloader('val', cfg.data.val_path, 1)
    model = BestStudentSupernet.load_from_checkpoint(best_student_ckpt)
    get_inference_time(model, val_dataloader)
    
    b1_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_1_cell_1/time=03/09/2021 13:20:30/epoch=4-val_loss=0.23.ckpt'
    b1_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_1_cell_2/time=04/09/2021 10:00:40/epoch=7-val_loss=0.26.ckpt'
    b1_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_1_cell_3/time=05/09/2021 13:32:37/epoch=10-val_loss=0.25.ckpt'
    b2_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_2_cell_1/time=03/09/2021 20:16:50/epoch=3-val_loss=0.26.ckpt'
    b2_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_2_cell_2/time=04/09/2021 15:45:41/epoch=7-val_loss=0.33.ckpt'
    b2_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_2_cell_3/time=05/09/2021 13:39:38/epoch=5-val_loss=0.34.ckpt'
    b3_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_3_cell_1/time=03/09/2021 20:24:27/epoch=3-val_loss=0.33.ckpt'
    b3_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_3_cell_2/time=04/09/2021 15:59:03/epoch=3-val_loss=0.36.ckpt'
    b3_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_3_cell_3/time=05/09/2021 13:45:17/epoch=5-val_loss=0.38.ckpt'
    b4_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_4_cell_1/time=03/09/2021 20:31:14/epoch=3-val_loss=0.38.ckpt'
    b4_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_4_cell_2/time=04/09/2021 17:43:05/epoch=3-val_loss=0.40.ckpt'
    b4_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_4_cell_3/time=05/09/2021 13:50:18/epoch=5-val_loss=0.45.ckpt'
    b5_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_5_cell_1/time=04/09/2021 08:34:29/epoch=3-val_loss=0.40.ckpt'
    b5_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_5_cell_2/time=04/09/2021 20:14:07/epoch=3-val_loss=0.44.ckpt'
    b5_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_5_cell_3/time=05/09/2021 14:08:45/epoch=4-val_loss=0.47.ckpt'
    b6_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_6_cell_1/time=04/09/2021 08:24:53/epoch=45-val_loss=0.14.ckpt'
    b6_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_6_cell_2/time=04/09/2021 21:34:34/epoch=16-val_loss=0.16.ckpt'
    b6_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS/DNAS_block_6_cell_3/time=05/09/2021 14:29:32/epoch=24-val_loss=0.20.ckpt'
    '''

    b1_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_1_cell_1/time=10/09/2021 12:18:26/epoch=6-val_loss=0.22.ckpt'
    b1_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_1_cell_2/time=10/09/2021 13:51:56/epoch=16-val_loss=0.24.ckpt'
    b1_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_1_cell_3/time=11/09/2021 07:58:45/epoch=19-val_loss=0.23.ckpt'
    b2_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_2_cell_1/time=10/09/2021 12:35:53/epoch=4-val_loss=0.23.ckpt'
    b2_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_2_cell_2/time=10/09/2021 14:54:31/epoch=11-val_loss=0.31.ckpt'
    b2_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_2_cell_3/time=11/09/2021 09:01:46/epoch=11-val_loss=0.29.ckpt'
    b3_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_3_cell_1/time=10/09/2021 12:50:07/epoch=4-val_loss=0.26.ckpt'
    b3_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_3_cell_2/time=10/09/2021 16:39:54/epoch=11-val_loss=0.37.ckpt'
    b3_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_3_cell_3/time=11/09/2021 09:21:28/epoch=11-val_loss=0.32.ckpt'
    b4_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_4_cell_1/time=10/09/2021 13:00:09/epoch=4-val_loss=0.30.ckpt'
    b4_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_4_cell_2/time=11/09/2021 04:20:00/epoch=11-val_loss=0.42.ckpt'
    b4_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_4_cell_3/time=11/09/2021 10:02:26/epoch=11-val_loss=0.40.ckpt'
    b5_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_5_cell_1/time=10/09/2021 13:08:21/epoch=4-val_loss=0.34.ckpt'
    b5_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_5_cell_2/time=11/09/2021 07:41:41/epoch=11-val_loss=0.41.ckpt'
    b5_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_5_cell_3/time=11/09/2021 10:24:37/epoch=11-val_loss=0.40.ckpt'
    b6_c1_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_6_cell_1/time=10/09/2021 13:27:42/epoch=14-val_loss=0.14.ckpt'
    b6_c2_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_6_cell_2/time=11/09/2021 07:48:21/epoch=16-val_loss=0.16.ckpt'
    b6_c3_ckpt = '/home/hang/projects/def-wjgross/hang/bert_nas/checkpoint_DNAS_right/DNAS_block_6_cell_3/time=11/09/2021 11:11:18/epoch=22-val_loss=0.19.ckpt'

    b1_ckpt = [b1_c1_ckpt, b1_c2_ckpt, b1_c3_ckpt]
    b2_ckpt = [b2_c1_ckpt, b2_c2_ckpt, b2_c3_ckpt]
    b3_ckpt = [b3_c1_ckpt, b3_c2_ckpt, b3_c3_ckpt]
    b4_ckpt = [b4_c1_ckpt, b4_c2_ckpt, b4_c3_ckpt]
    b5_ckpt = [b5_c1_ckpt, b5_c2_ckpt, b5_c3_ckpt]
    b6_ckpt = [b6_c1_ckpt, b6_c2_ckpt, b6_c3_ckpt]

    supernet_ckpts = [b1_ckpt, b2_ckpt, b3_ckpt, b4_ckpt, b5_ckpt, b6_ckpt]

    for block_num, block_ckpt in enumerate(supernet_ckpts):
        for cell_num, cell_ckpt in enumerate(block_ckpt):
            # print(block_num + 1, cell_num + 1)
            # print(get_best_model_from_DNAS(cell_ckpt, block_num + 1, cell_num + 1))
            print('block {} cell {}'.format(block_num + 1, cell_num + 1), get_best_model_from_DNAS(cell_ckpt, block_num + 1, cell_num + 1))