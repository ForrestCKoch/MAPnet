from typing import Any, Optional, Callable
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

import argparse

from map.defaults import *

# Use of typing inspired by https://github.com/vlukiyanov/pt-sdae
def train(
        train: torch.utils.data.Dataset, 
        test: torch.utils.data.Dataset, 
        model: torch.nn.Module, 
        epochs: Optional[int] = EPOCHS, 
        update_freq: Optional[int] = UPDATE_FREQ,
        batch_size: Optional[int] = BATCH_SIZE, 
        num_workers: Optional[int] = WORKERS, 
        cuda: Optional[bool] = CUDA, 
        loss_func: Optional[Callable[[float,float],None]] = None, 
        optimizer: Optional[Callable[[torch.nn.Module],torch.optim.Optimizer]]=None, 
        scheduler: Optional[Callable[[int, torch.nn.Module],None]] = None
    ) -> None:
    """
    Train the provided MAPnet
    :param train: Dataset object containing the training data.
    :param test: Dataset object containing the test data.
    :param model: `torch.nn.Module` model to be trained.
    :param epoch: The number of epochs to train over.
    :param update_freq: Test set accuracy will be assessed every `update_freq` epochs.
    :param batch_size: batch size for training.
    :param num_workers: how many workers to use for the DataLoader.
    :param cuda: Whether to use cuda device. Defaults to False.
    :param loss_func: Loss function to use.  If `None` (default), MSELoss is used.
    :param optimizer: Optimizer to use. If `None` (default), Adam is used.
    :param scheduler: Scheduler to use. If `None` (default), no scheduler is used. 
    """

    ###########################################################################
    # Some preamble to get everything ready for training
    ###########################################################################
    train_data_loader = DataLoader(train, num_workers=num_workers,
                    pin_memory=cuda, batch_size=batch_size,
                    shuffle=True)

    test_data_loader = DataLoader(test, num_workers=num_workers,
                    pin_memory=cuda, batch_size=batch_size,
                    shuffle=True)
    desc_genr = lambda x,y,z: 'Epoch: {} Test Loss: {:.2f} Train Loss: {:.2f}'.format(x,y,z)
    test_loss = 0.0

    if cuda:
        model = model.cuda()
    else: 
        model = model.float()

    if loss_func is None:
        loss_func = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam

    model_optimizer = optimizer(model.parameters(),lr=0.001)
    model_scheduler = scheduler(model) if scheduler is not None else scheduler
    

    ###########################################################################
    # Start Training Epoch
    ###########################################################################
    for i in range(0, epochs):

        if model_scheduler is not None:
            model_scheduler.step()
        data_iterator = tqdm(train_data_loader,desc=desc_genr(i,test_loss,0))

        train_loss = list()
        #######################################################################
        # Cycle through each batch in the epoch 
        #######################################################################
        for index, batch in enumerate(data_iterator):
            x,label = batch
            if cuda:
                x = x.cuda().float()
                label = label.cuda().float()
            else:
                x = x.float()
                label = label.float()
            y = model(x)
            loss = loss_func(y,label.view(-1,1))
            loss_value = float(loss.item())
            train_loss.append(loss_value)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()  
            data_iterator.set_description(desc_genr(i,test_loss,np.mean(train_loss)))
    
        #######################################################################
        # Update test accuracy every `update_freq` number of epochs
        #######################################################################
        if (i+1)%update_freq==0:
            total_loss = 0.0
            for index, batch in enumerate(test_data_loader):
                x,label = batch
                if cuda:
                    x = x.cuda().float()
                    label = label.cuda().float()
                else:
                    x = x.float()
                    label = label.float()
                y = model(x)
                loss = loss_func(y,label.view(-1,1))
                total_loss += float(loss.item())
            test_loss = total_loss/(index+1)

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type = str,
        default = DATAPATH,
        help = "path to data folder [Default: '{}']".format(DATAPATH)
    )
    parser.add_argument(
        "--conv-layers",
        type = int,
        default = CONV_LAYERS,
        help = "number of Conv3d layers [Default: {}]".format(CONV_LAYERS)
    )
    parser.add_argument(
        "--kernel-size",
        type = int,
        default = KERNEL_SIZE,
        help = "kernel size of each filter [Default: {}]".format(KERNEL_SIZE)
    )
    parser.add_argument(
        "--dilation",
        type = int,
        default = DILATION,
        help = "dilation factor for each filter [Default: {}]".format(DILATION)
    )
    parser.add_argument(
        "--padding",
        type = int,
        default = PADDING,
        help = "zero padding to be used in Conv3d layers [Default: {}]".format(PADDING)
    )
    parser.add_argument(
        "--stride",
        type = int,
        default = STRIDE,
        help = "stride between filter applications [Default: {}]".format(STRIDE)
    )
    parser.add_argument(
        "--filters",
        nargs = '+',
        type = int,
        default = [4,4,4],
        help = "filters to apply to each channel -- one entry per layer [Default: {}]".format(' '.join(FILTERS))
    )
    parser.add_arguments(
        "--batch-size",
        type = int,
        default = 32,
        help = "number of samples per batch [Default: {}]".format(BATCH_SIZE)
    )
    parser.add_arguments(
        "--epochs",
        type = int,
        default = EPOCHS,
        help = "number of epochs to train over [Default: {}]".format(EPOCHS)
    )
    parser.add_arguments(
        "--update-freq",
        type = int,
        default = UPDATE_FREQ,
        help = "how often (in epochs) to asses test set accuracy [Default: {}]".format(UPDATE_FREQ)
    )
    parser.add_arguments(
        "--learning-rate",
        type = float,
        default = 0.001,
        help = "learning rate paramater [Default: {}]".format(LEARNING_RATE)
    )
    parser.add_arguments(
        "--workers",
        type = int,
        default = 8,
        help = "number of workers in DataLoader [Default: {}]".format(WORKERS)
    )
    parser.add_arguments(
        "--cuda",
        action="store_true",
        help = "set flag to use cuda device(s)"
    )

    # not implemented
    parser.add_arguments(
        "--subpooling",
        action="store_true",
        help = "set flag to use subpooling between Conv3d layers"
    )
    parser.add_arguments(
        "--scale-inputs",
        action="store_true",
        help = "set flag to scale input images"
    )
    parser.add_argument(
        "--encode-age",
        action="store_true",
        help = "set flag to encode age in a binary vector"
    )
    parser.add_argument(
        "--savepath",
        type = str,
        default = SAVEPATH,
        help = "folder where model checkpoints should be saved -- if None model will not be saved [Default: {}]".format(str(SAVEPATH))
    )
    parser.add_argument(
        "--save-freq",
        type = str,
        default = SAVE_FREQ,
        help = "how often model checkpoints should be saved (in epochs) [Default: {}]".format(SAVE_FREQ)
    )
    

    return parser
            
if __name__ == '__main__': 
    pass
