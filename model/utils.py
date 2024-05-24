from torch import nn
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import distance_matrix

### Metrics 

def ade(y1,y2):
    """
    y: (seq_len,2)
    """

    loss = y1 -y2
    loss = loss**2
    loss = np.sqrt(np.sum(loss,1))

    return np.mean(loss)

def fde(y1,y2):
    loss = (y1[-1,:] - y2[-1,:])**2
    return np.sqrt(np.sum(loss))

def rel_to_abs(obs,rel_pred):

    pred = rel_pred.copy()
    pred[0] += obs[-1]
    for i in range(1,len(pred)):
        pred[i] += pred[i-1]
    
    return pred 

def rmse(y1,y2):
    criterion = nn.MSELoss()

    # return loss
    return torch.sqrt(criterion(y1, y2))

## General utils

def acc_to_abs(acc,obs,delta=1):
    acc = acc.permute(2,1,0)
    pred = torch.empty_like(acc)
    pred[0] = 2*obs[-1] - obs[0] + acc[0]
    pred[1] = 2*pred[0] - obs[-1] + acc[1]
    
    for i in range(2,acc.shape[0]):
        pred[i] = 2*pred[i-1] - pred[i-2] + acc[i]
    return pred.permute(2,1,0)
    

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    context = torch.cat(context_list, dim=0 ).permute(2,0,1)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end
    ]
    return tuple(out)


def loss_func(recon_y,y,mean,log_var):
    traj_loss = rmse(recon_y,y)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return traj_loss + KLD