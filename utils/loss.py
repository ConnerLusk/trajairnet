from torch import nn
import torch

def rmse(y1,y2):
    criterion = nn.MSELoss()
    # return loss
    return torch.sqrt(criterion(y1, y2))

def loss_func(recon_y,y,mean,log_var):
    traj_loss = rmse(recon_y,y)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return traj_loss + KLD