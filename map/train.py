from typing import Any, Optional, Callable
import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm


from defaults import *
from data import *
from model import *
from train import *

# Use of typing inspired by https://github.com/vlukiyanov/pt-sdae
def train(
        train: torch.utils.data.Dataset, 
        test: torch.utils.data.Dataset, 
        model: torch.nn.Module, 
        epochs: Optional[int] = EPOCHS, 
        update_freq: Optional[int] = UPDATE_FREQ,
        savepath: Optional[str] = SAVEPATH,
        save_freq: Optional[int] = SAVE_FREQ,
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
    desc_genr = lambda x,y,z: 'Epoch: {} Test Loss: {} Train Loss: {}'.format(
        x,
        np.format_float_scientific(y,precision=3),
        np.format_float_scientific(z,precision=3)
    )
    test_loss = 0.0

    if cuda:
        model = model.cuda().float()
    else: 
        model = model.float()

    if loss_func is None:
        loss_func = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = lambda x: torch.optim.Adam(x.parameters(),lr=0.000001)

    model_optimizer = optimizer(model)
    model_scheduler = scheduler(model) if scheduler is not None else scheduler

    if savepath is not None:
        save_folder = os.path.join(savepath,datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))    
        os.makedirs(save_folder)
    else:
        save_folder = None

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
                x = x.cuda()
                label = label.cuda()

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
                    x = x.cuda()
                    label = label.cuda()
                y = model(x)
                loss = loss_func(y,label.view(-1,1))
                total_loss += float(loss.item())
            test_loss = total_loss/(index+1)
    
        if (i+1)%save_freq==0 and savepath is not None:
            torch.save(model, os.path.join(save_folder,'epoch-{}.dat'.format(i+1)))
            

def _get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--datapath",
        type = str,
        metavar = '[str]',
        default = DATAPATH,
        help = "path to data folder"
    )
    parser.add_argument(
        "--conv-layers",
        type = int,
        metavar = '[int]',
        default = CONV_LAYERS,
        help = "number of Conv3d layers"
    )
    parser.add_argument(
        "--kernel-size",
        type = int,
        metavar = '[int]',
        default = KERNEL_SIZE,
        help = "kernel size of each filter"
    )
    parser.add_argument(
        "--dilation",
        type = int,
        metavar = '[int]',
        default = DILATION,
        help = "dilation factor for each filter"
    )
    parser.add_argument(
        "--padding",
        type = int,
        metavar = '[int]',
        default = PADDING,
        help = "zero padding to be used in Conv3d layers"
    )
    parser.add_argument(
        "--stride",
        type = int,
        metavar = '[int]',
        default = STRIDE,
        help = "stride between filter applications"
    )
    parser.add_argument(
        "--filters",
        nargs = '+',
        type = int,
        metavar = 'int',
        default = [4,4,4],
        help = "filters to apply to each channel -- one entry per layer"
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        metavar = '[int]',
        default = 32,
        help = "number of samples per batch"
    )
    parser.add_argument(
        "--epochs",
        type = int,
        metavar = '[int]',
        default = EPOCHS,
        help = "number of epochs to train over"
    )
    parser.add_argument(
        "--update-freq",
        type = int,
        metavar = '[int]',
        default = UPDATE_FREQ,
        help = "how often (in epochs) to asses test set accuracy"
    )
    parser.add_argument(
        "--lr",
        type = float,
        metavar = '[float]',
        default = LEARNING_RATE,
        help = "learning rate paramater"
    )
    parser.add_argument(
        "--workers",
        type = int,
        metavar = '[int]',
        default = WORKERS,
        help = "number of workers in DataLoader"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help = "set flag to use cuda device(s)"
    )
    parser.add_argument(
        "--savepath",
        type = str,
        metavar = '[str]',
        default = SAVEPATH,
        help = "folder where model checkpoints should be saved -- if None model will not be saved "
    )
    parser.add_argument(
        "--save-freq",
        type = str,
        metavar = '[str]',
        default = SAVE_FREQ,
        help = "how often model checkpoints should be saved (in epochs) "
    )
    parser.add_argument(
        "--scale-inputs",
        action="store_true",
        help = "set flag to scale input images"
    )

    # not implemented
    parser.add_argument(
        "--silent",
        action = "store_true",
        #help = "set flag for quiet training"
        help = "NOT IMPLEMENTED"
    )
    parser.add_argument(
        "--subpooling",
        action="store_true",
        #help = "set flag to use subpooling between Conv3d layers"
        help = "NOT IMPLEMENTED"
    )
    parser.add_argument(
        "--encode-age",
        action="store_true",
        #help = "set flag to encode age in a binary vector"
        help = "NOT IMPLEMENTED"
    )
    

    return parser
            
if __name__ == '__main__': 
    parser = _get_parser()
    args = parser.parse_args()

    if not args.silent:
        print("Fetching training data ...")
    train_dict = get_sample_dict(
        datapath = args.datapath,
        dataset='train'
    )
    train_ages = get_sample_ages(
        ids = train_dict.keys(),
        path_to_csv = os.path.join(args.datapath,'subject_info.csv')
    )
    train_ds = NiftiDataset(
        samples = train_dict,
        labels = train_ages/100, # divide by 100 for faster learning!
        scale_inputs = args.scale_inputs
    )

    if not args.silent:
        print("Fetching test data ...")
    test_dict = get_sample_dict(
        datapath = args.datapath,
        dataset = 'test'
        )
    test_ages = get_sample_ages(
        ids = test_dict.keys(),
        path_to_csv = os.path.join(args.datapath,'subject_info.csv')
    )
    test_ds = NiftiDataset(
        samples = test_dict,
        labels = test_ages/100, # divide by 100 for faster learning!
        scale_inputs = args.scale_inputs,
        cache_images = True
    )

    if not args.silent:
        print("Initializing model ...")
    model = MAPnet(
        input_shape = train_ds.image_shape,
        n_conv_layers = args.conv_layers,
        padding = args.padding,
        dilation = args.dilation,
        kernel = args.kernel_size,
        stride = args.stride,
        filters = args.filters,
        input_channels = train_ds.images_per_subject
    )
    # print out summary of model
    model = model.cuda() if args.cuda else model
    if not args.silent:
        summary(
            model,
            input_size = tuple(np.concatenate(
                [[train_ds.images_per_subject],np.array(train_ds.image_shape)]
            )),
            device = "cuda" if args.cuda else "cpu"
        )    

    train(
        train_ds,
        test_ds,
        model,
        cuda = args.cuda,
        batch_size = args.batch_size,
        num_workers = args.workers,
        epochs = args.epochs,
        update_freq = args.update_freq,
        savepath = args.savepath,
        save_freq = args.save_freq,
        optimizer = lambda x: torch.optim.Adam(x.parameters(),lr=args.lr)
    )
