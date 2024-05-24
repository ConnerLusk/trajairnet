import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from model.trajairnet import TrajAirNet
from utils.loss import loss_func

class TrajAirNetLightning(pl.LightningModule):
    def __init__(self, args):
        super(TrajAirNetLightning, self).__init__()
        self.model = TrajAirNet(args)
        self.learning_rate = 1e-3

    def forward(self, x, y, adj, context):
        return self.model(x, y, adj, context)

    def training_step(self, batch, batch_idx):
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
        num_agents = obs_traj.shape[1]
        pred_traj = torch.transpose(pred_traj, 1, 2)
        adj = torch.ones((num_agents, num_agents))

        recon_y, means, log_var = self(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2))

        loss = 0
        for agent in range(num_agents):
            loss += loss_func(recon_y[agent], torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), means[agent], log_var[agent])
        print('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
        num_agents = obs_traj.shape[1]
        pred_traj = torch.transpose(pred_traj, 1, 2)
        adj = torch.ones((num_agents, num_agents))

        recon_y, means, log_var = self(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2))

        val_loss = 0
        for agent in range(num_agents):
            val_loss += loss_func(recon_y[agent], torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), means[agent], log_var[agent])

        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)