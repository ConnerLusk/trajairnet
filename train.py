import argparse
import os 
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl



from model.trajairnet import TrajAirNet
from model.utils import seq_collate, loss_func
from datasets.trajair_dataset import TrajAirDataset
from datamodules.trajair_datamodule import TrajAirDataModule
from model.trajairnet_lightning import TrajAirNetLightning
from test import test



def train():
    ## Dataset params
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/data/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    ## Network params
    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)

    parser.add_argument('--num_context_input_c',type=int,default=2)
    parser.add_argument('--num_context_output_c',type=int,default=7)
    parser.add_argument('--cnn_kernels',type=int,default=2)

    parser.add_argument('--gat_heads',type=int, default=16)
    parser.add_argument('--graph_hidden',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--cvae_hidden',type=int,default=128)
    parser.add_argument('--cvae_channel_size',type=int,default=128)
    parser.add_argument('--cvae_layers',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=32)

    ## Training Params
    parser.add_argument('--lr',type=float,default=0.001)


    parser.add_argument('--total_epochs',type=int, default=50)
    parser.add_argument('--delim',type=str,default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)

    parser.add_argument('--model_pth', type=str , default="/saved_models/")

    args=parser.parse_args()
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    args.root = datapath
    args.train_batch_size = 1
    args.val_batch_size = 1

    if torch.cuda.is_available():
        trainer = pl.Trainer.from_argparse_args(args, accelerator="gpu", devices="auto")
    else:
        trainer = pl.Trainer.from_argparse_args(args)
    model = TrajAirNetLightning(args)
    datamodule = TrajAirDataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)

if __name__=='__main__':

    train()