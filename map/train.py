from typing import Any, Optional, Callable
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

# Use of typing inspired by https://github.com/vlukiyanov/pt-sdae
def train(
        train: torch.utils.data.Dataset, 
        test: torch.utils.data.Dataset, 
        model: torch.nn.Module, 
        epochs: Optional[int] = 10, 
        update_freq: Optional[int] = 5,
        batch_size: Optional[int] = 8, 
        num_workers: Optional[int] = 1, 
        cuda: Optional[bool] = False, 
        loss_func: Optional[Callable[[float,float],None]] = None, 
        optimizer: Optional[Callable[[torch.nn.Module],torch.optim.Optimizer]]=None, 
        scheduler: Optional[Callable[[int, torch.nn.Module],None]] = None
    ) -> None:
    """
    Train the provided MAPnet
    :param train: Dataset object containing the training data.
    :param test: Dataset object containing the test data.
    :param model: torch.nn.Module model to be trained.
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

    model_optimizer = optimizer(model.parameters(),lr=0.01,weight_decay=0.1)
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
            
    
