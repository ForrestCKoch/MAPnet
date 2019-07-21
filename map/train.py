from typing import Any, Optional, Callable
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm


from defaults import *
from data import get_sample_dict, get_sample_ages, encode_age_nonordinal, \
        encode_age_ordinal, encode_smooth_age, NiftiDataset 
from model import get_out_dims, init_weights, get_even_padding, MAPnet, \
        winit_funcs, actv_funcs

loss_funcs = {
    'L1':torch.nn.L1Loss,
    'L2':torch.nn.MSELoss,
    'SmoothL1':torch.nn.SmoothL1Loss,
    'BCE':torch.nn.BCELoss,
}

def train_model(
        train_set: torch.utils.data.Dataset, 
        test_set: torch.utils.data.Dataset, 
        model: torch.nn.Module, 
        epochs: Optional[int]=EPOCHS, 
        update_freq: Optional[int]=UPDATE_FREQ,
        save_folder: Optional[str]=SAVEPATH,
        save_freq: Optional[int]=SAVE_FREQ,
        batch_size: Optional[int]=BATCH_SIZE, 
        num_workers: Optional[int]=WORKERS, 
        cuda: Optional[bool]=CUDA, 
        loss_func: Optional[Callable[[float,float],None]]=None, 
        optimizer: Optional[Callable[[torch.nn.Module],torch.optim.Optimizer]]=None, 
        scheduler: Optional[Callable[[int, torch.nn.Module],None]]=None,
        silent: Optional[bool]=False
    ) -> None:
    """
    Train the provided MAPnet model.

    Note: 

    :param train_set: Dataset object containing the training data.
    :param test_set: Dataset object containing the test data.
    :param model: `torch.nn.Module` model to be trained.
    :param epoch: The number of epochs to train over.
    :param update_freq: Test set accuracy will be assessed every `update_freq` epochs.
    :param save_folder: Path for checkpoints to be stored in.
    :param save_freq: How often checkpoints should be saved.
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
    train_data_loader = DataLoader(train_set, num_workers=num_workers,
            pin_memory=cuda, batch_size=batch_size,
            shuffle=True)

    test_data_loader = DataLoader(test_set, num_workers=num_workers,
            pin_memory=cuda, batch_size=batch_size,
            shuffle=True)

    desc_genr = lambda x,y,z,lr: 'Epoch: {0:3} Test Loss: {1:9} Train Loss: {2:9} LR: {3:9}'.format(
        x,
        np.format_float_scientific(y, precision=3),
        np.format_float_scientific(z, precision=3),
        np.format_float_scientific(lr,precision=3)
    )
    test_loss = 0.0

    if cuda:
        model = model.cuda().float()
    else: 
        model = model.float()

    if loss_func is None:
        loss_func = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = lambda x: torch.optim.Adam(x.parameters(),lr=0.001)

    model_optimizer = optimizer(model)
    model_scheduler = scheduler(model_optimizer) if scheduler is not None else scheduler


    ###########################################################################
    # Start Training Epoch
    ###########################################################################
    for i in range(0, epochs):

        if model_scheduler is not None:
            model_scheduler.step()
        data_iterator = tqdm(
            train_data_loader,
            desc=desc_genr(i,test_loss,0,model_optimizer.param_groups[0]['lr']),
            disable=silent
        )

        train_loss = list()
        #######################################################################
        # Cycle through each batch in the epoch 
        #######################################################################
        for index, batch in enumerate(data_iterator):
            x,label = batch
            if cuda:
                x = x.cuda()
                label = label.cuda()

            model_optimizer.zero_grad()
            y = model(x)
            loss = loss_func(y,label.view(batch_size,model.output_size))
            loss_value = float(loss.item())
            train_loss.append(loss_value)
            loss.backward()
            model_optimizer.step()  
            data_iterator.set_description(desc_genr(
                i,
                test_loss,
                np.mean(train_loss),
                model_optimizer.param_groups[0]['lr']
            ))

        #######################################################################
        # Update the learning rate if we've been supplied a scheduler
        #######################################################################
        if model_scheduler is not None:
            if isinstance(model_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                model_scheduler.step(float(np.mean(train_loss)))
            else:
                model_scheduler.step()
    
        #######################################################################
        # Save the model when requested
        #######################################################################
        if (i+1)%save_freq==0 and save_folder is not None:
            torch.save(model, os.path.join(save_folder,'epoch-{}.dat'.format(i+1)))

        #######################################################################
        # Update test accuracy every `update_freq` number of epochs
        #######################################################################
        if (i+1)%update_freq==0:
            test_loss = test_model(model,test_data_loader,loss_func,cuda) 

        #######################################################################
        # Record our losses for this epoch
        #######################################################################
        with open(os.path.join(save_folder,'loss.csv'),'a') as fh:
            print('{},{},{},{}'.format( #epoch,LR,train_loss,test_loss
                str(i),
                np.format_float_scientific(
                    model_optimizer.param_groups[0]['lr'],
                    precision=5,
                ),
                np.format_float_scientific(np.mean(train_loss), precision=5),
                np.format_float_scientific(test_loss, precision=5)
                ),
                file=fh
            )
            
def test_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_func: Callable[[float,float],None],
        cuda: Optional[bool]=False,
        show_progress: Optional[bool]=False,
    )->float:
    """
    Return the average loss over the provided dataset.
    :param model: `torch.nn.Module` to be tested
    :param data_loader: DataLoader class for the dataset being tested
    :param loss_func: Loss function to use for measuring error/loss
    :param cuda: whether to use the cuda device (model should already be moved to GPU)
    :param show_progress: whether to use tqdm to show progress of test loop
    """
    total_loss = 0.0
    # disable gradient calculations to avoid wasting memory
    data_iterator = tqdm(data_loader) if show_progress else data_loader
    with torch.no_grad():
        for index, batch in enumerate(data_iterator):
            x,label = batch
            if cuda:
                x = x.cuda()
                label = label.cuda()
            y = model(x)
            loss = loss_func(y,label.view(data_loader.batch_size,model.output_size))
            total_loss += float(loss.item())
        test_loss = total_loss/(index+1)
    return test_loss
    
            

def _get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ###########################################################################
    # Data options
    ###########################################################################
    parser.add_argument(
        "--datapath",
        type=str,
        metavar='[str]',
        default=DATAPATH,
        help="Path to data folder"
    )
    parser.add_argument(
        "--scale-inputs",
        action="store_true",
        help="Set flag to scale input images"
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar='[int]',
        default=WORKERS,
        help="Number of workers in DataLoader"
    )
    ###########################################################################
    # Loading/Saving
    ###########################################################################
    parser.add_argument(
        "--savepath",
        type=str,
        metavar='[str]',
        default=SAVEPATH,
        help="Folder where model checkpoints should be saved -- if None model will not be saved. If savepath is specified, models will be saved in a new folder named according to the date and time it is run.  If mutliple instances are being run in parallel, each instance should have a different savepath to avoid overlap."
    )
    parser.add_argument(
        "--save-freq",
        type=str,
        metavar='[str]',
        default=SAVE_FREQ,
        help="How often model checkpoints should be saved (in epochs) "
    )
    parser.add_argument(
        "--load-model",
        type=str,
        metavar='[str]',
        default=None,
        help="Specify a saved model to load and train.  Other arguments relating to model paremeters (padding, kernel-size, etc..) will be ignored.  Training parameters (learning rate, update frequency, etc ...) may still be specified."
    )
    ###########################################################################
    # Model Architecture Options
    ###########################################################################
    parser.add_argument(
        "--conv-layers",
        type=int,
        metavar='[int]',
        default=CONV_LAYERS,
        help="Number of Conv3d layers"
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        nargs='+',
        metavar='int',
        default=KERNEL_SIZE,
        help="Kernel size of each filter"
    )
    parser.add_argument(
        "--dilation",
        type=int,
        nargs='+',
        metavar='int',
        default=DILATION,
        help="Dilation factor for each filter"
    )
    parser.add_argument(
        "--padding",
        type=int,
        nargs='+',
        metavar='int',
        default=PADDING,
        help="Zero padding to be used in Conv3d layers"
    )
    parser.add_argument(
        "--even-padding",
        action="store_true",
        help="Calculate padding vectors to ensure even perfect overlap with kernel applications.  Layers with stride = 1 will have input dimensions preserved.  The '--padding' argument is ignored when this flag is set"
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs='+',
        metavar='int',
        default=STRIDE,
        help="Stride between filter applications"
    )
    parser.add_argument(
        "--filters",
        nargs='+',
        type=int,
        metavar='int',
        default=[4,4,4],
        help="Filters to apply to each channel -- one entry per layer"
    )
    parser.add_argument(
        "--weight-init",
        type=str,
        metavar='[str]',
        default=WEIGHT_INIT,
        choices=winit_funcs.keys(),
        help="Weight initialization method [{}]".format(', '.join(winit_funcs.keys()))
    )
    parser.add_argument(
        "--conv-actv",
        type=str,
        nargs='+',
        metavar='str',
        default=CONV_ACTV_ARG,
        choices=actv_funcs.keys(),
        help="Activation functions to be used in convolutional layers -- must be 1 or n_conv_layers [{}]".format(', '.join(actv_funcs.keys()))
    )
    parser.add_argument(
        "--fc-actv",
        type=str,
        nargs='+',
        metavar='str',
        default=CONV_ACTV_ARG,
        choices=actv_funcs.keys(),
        help="Activation functions to be used in convolutional layers -- must be 1 or n_conv_layers [{}]".format(', '.join(actv_funcs.keys()))
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=['max','avg'],
        metavar='[str]',
        default=None,
        help="Which pooling method to apply in between convolution layers.  If this argument is not specified, then no pooling will be performed. ['max','avg']"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        choices=['value','scaled-value','single-class','ordinal-class','gaussian'],
        metavar='[str]',
        default='age',
        help="Specify what type of output the model should produce. 'value': model is trained to predict a single value (e.g age). 'scaled-value': same as 'value', but scaled down by a factor of 100. 'single-class': ages are treated as individual classes to be predited. 'ordinal-class': a class should be predicted if it is <= target age. 'gaussian': model is trained to predict a range of outputs centered around the target age. ['value','scaled-value','single-class','ordinal-class','gaussian']"
    )
    ###########################################################################
    #  Training Options
    ###########################################################################
    parser.add_argument(
        "--lr",
        type=float,
        metavar='[float]',
        default=LEARNING_RATE,
        help="Learning rate paramater"
    )
    parser.add_argument(
        "--decay",
        type=float,
        metavar='[float]',
        default=DECAY,
        help="Learning rate decay (multiplicative factor).  Unless '--reduce-on-plateau is set, this decay rate is applied every epoch" 
    )
    parser.add_argument(
        "--reduce-on-plateau",
        action="store_true",
        help="Learning rate will decay after performance on the train set plateaus as opposed to every epoch"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        metavar='[int]',
        default=BATCH_SIZE,
        help="Number of samples per batch"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        metavar='[int]',
        default=EPOCHS,
        help="Number of epochs to train over"
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        metavar='[int]',
        default=UPDATE_FREQ,
        help="How often (in epochs) to asses test set accuracy"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Set flag to use cuda device(s)"
    )
    parser.add_argument(
        "--loss",
        metavar='[str]',
        choices=loss_funcs.keys(),
        default='L2',
        help="Specify a loss function. [{}]".format(', '.join(loss_funcs.keys()))
    )
    ###########################################################################
    # Misc. Options
    ###########################################################################
    parser.add_argument(
        "--debug-size",
        type=int,
        metavar='',
        nargs=4, 
        help="Print out the expected architecture.  4 Integers should be supplied to this argument [channels, dimx, dimy, dimz].  Program execution will terminate afterwards"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        #help="Set flag for quiet training"
        help="NOT IMPLEMENTED"
    )
    parser.add_argument(
        "--test-model",
        type=str,
        metavar='[str]',
        choices=['test','train'],
        default=None,
        help="Instead of training the loaded model, it's performance will be assessed on either the test or train set. ['test','train']"
    )

    ###########################################################################
    # not implemented
    ###########################################################################
    parser.add_argument(
        "--encode-age",
        action="store_true",
        #help="set flag to encode age in a binary vector"
        help="NOT IMPLEMENTED"
    )
    

    return parser

def print_network_size(args:argparse.ArgumentParser):
    padding = args.padding
    dilation = args.dilation
    kernel = args.kernel_size
    stride = args.stride
    conv_layers = args.conv_layers
    filters = args.filters
    t,x,y,z = args.debug_size

    if len(padding) == 1:
        padding = np.repeat(padding,conv_layers)
    if len(dilation) == 1:
        dilation = np.repeat(dilation,conv_layers)
    if len(kernel) == 1:
        kernel = np.repeat(kernel,conv_layers)
    if len(stride) == 1:
        stride = np.repeat(stride,conv_layers)

    conv_layer_sizes = list([np.array([x,y,z])])
    n_channels = list([t])
    for i in range(0,conv_layers):
        dims = get_out_dims(
            conv_layer_sizes[-1],
            np.repeat(padding[i],3),
            np.repeat(dilation[i],3),
            np.repeat(kernel[i],3),
            np.repeat(stride[i],3)
        ).astype(np.int16)
        idims = conv_layer_sizes[-1]
        conv_layer_sizes.append(dims)
        n_channels.append(n_channels[-1]*filters[i])

        print("Conv Layer {}: ({},{},{},{}) -> ({},{},{},{})".format(
            i,
            n_channels[-2],
            idims[0],
            idims[1],
            idims[2],
            n_channels[-1],
            dims[0],
            dims[1],
            dims[2]
        ))

    fc = int(np.prod(conv_layer_sizes[-1]))*n_channels[-1]
    print("FC layers: {} -> {} -> 100 -> 1".format(fc,int(fc/2)))

def convert_targets(targets,option):
    choices=['value','scaled-value','single-class','ordinal-class','gaussian'],
    if option == 'scaled-value':
        return targets/100
    elif option == 'single-class':
        return [encode_age_nonordinal(t,np.array(range(35,75))) for t in targets]
    elif option == 'ordinal-class':
        return [encode_age_ordinal(t,np.array(range(35,75))) for t in targets]
    elif option == 'gaussian':
        return [encode_smooth_age(t,np.array(range(35,75)),0.7) for t in targets]
    else:
        return targets
            
if __name__ == '__main__': 
    parser = _get_parser()
    args = parser.parse_args()

    if args.debug_size:
        print_network_size(args)
        exit()
    ###########################################################################
    # Preamble
    ###########################################################################
    if not args.silent:
        print('')
        print('================================================================')
        print('| MAPnnet -- MRI Age Prediction neural network                 |')
        print('|    Forrest Koch (forrest.koch@unsw.edu.au)                   |')
        print('================================================================')
        print('')
        print('{} was called with arguments:'.format(sys.argv[0]))
        for arg in vars(args):
            print('{0:20}{1}'.format(arg+':',getattr(args,arg)))
        print('')


    ###########################################################################
    # Loading Training Data
    ###########################################################################
    if not args.silent:
        print("Fetching training data ...")
    train_dict = get_sample_dict(
        datapath=args.datapath,
        dataset='train'
    )
    train_ages = get_sample_ages(
        ids=train_dict.keys(),
        path_to_csv=os.path.join(args.datapath,'subject_info.csv')
    )
    conv_train_ages = convert_targets(train_ages,args.model_output)
    train_ds = NiftiDataset(
        samples=train_dict,
        labels=conv_train_ages,
        scale_inputs=args.scale_inputs
    )

    ###########################################################################
    # Loading Testing Data
    ###########################################################################
    if not args.silent:
        print("Fetching test data ...")
    test_dict = get_sample_dict(
        datapath=args.datapath,
        dataset='test'
        )
    test_ages = get_sample_ages(
        ids=test_dict.keys(),
        path_to_csv=os.path.join(args.datapath,'subject_info.csv')
    )
    conv_test_ages = convert_targets(test_ages,args.model_output)
    test_ds = NiftiDataset(
        samples=test_dict,
        labels=conv_test_ages, 
        scale_inputs=args.scale_inputs,
        cache_images=False
    )

    ###########################################################################
    # Initializing Model
    ###########################################################################
    if args.load_model is None:
        if not args.silent:
            print("Initializing model ...")
        model = MAPnet(
            input_shape=train_ds.image_shape,
            n_conv_layers=args.conv_layers,
            padding=args.padding,
            dilation=args.dilation,
            kernel=args.kernel_size,
            stride=args.stride,
            filters=args.filters,
            input_channels=train_ds.images_per_subject,
            conv_actv=[actv_funcs[x] for x in args.conv_actv],
            fc_actv=[actv_funcs[x] for x in args.fc_actv],
            even_padding=args.even_padding,
            pool=args.pooling,
            output_size=len(conv_train_ages[0])
        )
        ###########################################################################
        # Weight Initializaiton
        ###########################################################################
        fn = winit_funcs[args.weight_init]
        model.apply(lambda x: init_weights(x,fn))
    else:
        if not args.silent:
            print("Loading model ...")
        if not os.path.exists(args.load_model):
            raise ValueError("Cannot load model -- {} does not exist".format(args.load_model))
        model = torch.load(args.load_model)
        # a quick hack to allow for backwards compatibility
        if not hasattr(model,'even_padding'):
            model.even_padding = False
        if not hasattr(model,'pool'):
            model.pool = False
        if not hasattr(model,'output_size'):
            model.output_size = 1


    ###########################################################################
    # Move the model to GPU if cuda is requested
    ###########################################################################
    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    ###########################################################################
    # Print out model info ...
    ###########################################################################
    if not args.silent:
        summary(
            model,
            input_size = tuple(np.concatenate(
                [[train_ds.images_per_subject],np.array(train_ds.image_shape)]
            )),
            device = "cuda" if args.cuda else "cpu"
        )    

    if args.test_model is not None:
        ###########################################################################
        # Run a test of the model instead of a training
        ###########################################################################
        run_set = train_ds if args.test_model == 'train' else test_ds
        data_loader = DataLoader(
            run_set, 
            num_workers=args.workers,
            pin_memory=args.cuda, 
            batch_size=args.batch_size,
            shuffle=True
        )
        loss = test_model(
            model,
            data_loader,
            loss_funcs[args.loss](),
            args.cuda,show_progress=True
        )
        print("Total loss for the {} set is {}".format(
                args.test_model,
                np.format_float_scientific(loss,precision=3)
            )
        )
    else:
        ###########################################################################
        # Setup our save location
        ###########################################################################
        if args.savepath is not None:
            save_folder = os.path.join(
                args.savepath,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            )    
            os.makedirs(save_folder)
            # Record the arguments for this program call in the save_folder
            with open(os.path.join(save_folder,'arguments.txt'),'w') as fh:
                for arg in vars(args):
                    print('{0:20}{1}'.format(arg+':',getattr(args,arg)),file=fh)
        else:
            save_folder = None
        ###########################################################################
        # And finally, begin training
        ###########################################################################
        train_model(
            train_ds,
            test_ds,
            model,
            cuda=args.cuda,
            batch_size=args.batch_size,
            num_workers=args.workers,
            epochs=args.epochs,
            update_freq=args.update_freq,
            save_folder=save_folder,
            save_freq=args.save_freq,
            loss_func=loss_funcs[args.loss](),
            optimizer=lambda x: torch.optim.Adam(x.parameters(),lr=args.lr),
            scheduler=lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(
                    x, 
                    factor=args.decay,
                    patience=10,
                    threshold=1e-04
                ) if args.reduce_on_plateau else \
                torch.optim.lr_scheduler.ExponentialLR(
                    x,
                    gamma=args.decay
                )
        )
