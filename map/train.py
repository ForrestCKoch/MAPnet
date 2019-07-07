from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

def train(dataset, model, epochs=10, update_freq=1, batch_size=8, num_workers=1, 
            cuda=False, loss_func=None, optimizer=None, scheduler=None):

    data_loader = DataLoader(dataset, num_workers=num_workers,
                    pin_memory=cuda, batch_size=1)
    desc_genr = lambda x,y: 'Epoch: {}\tLoss: {:.2f}'.format(x,y)

    if cuda:
        model = model.cuda()
    else: 
        model = model.float()

    if loss_func is None:
        loss_func = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam

    model_optimizer = optimizer(model.parameters())
    model_scheduler = scheduler(model) if scheduler is not None else scheduler

    for i in range(0, epochs):
        if model_scheduler is not None:
            model_scheduler.step()
        data_iterator = tqdm(data_loader,desc=desc_genr(0,0),unit='batch')
        for index, batch in enumerate(data_iterator):
            x,label = batch
            if cuda:
                x = x.cuda()
                label = label.cuda()
            else:
                x = x.float()
                label = label.float()
            y = model(x)
            loss = loss_func(y,label)
            loss_value = float(loss.item())
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()  
            data_iterator.set_description(desc_genr(i,loss_value))
    
    
