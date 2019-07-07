from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

def train(train, test, model, epochs=10, update_freq=5, batch_size=8, 
    num_workers=1, cuda=False, loss_func=None, optimizer=None, scheduler=None,):

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

    model_optimizer = optimizer(model.parameters())
    model_scheduler = scheduler(model) if scheduler is not None else scheduler
    

    for i in range(0, epochs):

        if model_scheduler is not None:
            model_scheduler.step()
        data_iterator = tqdm(train_data_loader,desc=desc_genr(i,test_loss,0))

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
            loss_value = float(loss.item())/batch_size
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()  
            data_iterator.set_description(desc_genr(i,test_loss,loss_value))
    
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
                total_loss += float(loss.item())/batch_size
            test_loss = total_loss/(index+1)
            
    
