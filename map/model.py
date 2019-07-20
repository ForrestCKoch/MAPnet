from typing import Any, Optional, Callable, Tuple, List
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from defaults import *


winit_funcs = {
    'normal':nn.init.normal_,
    'uniform':nn.init.uniform_,
    'xavier-normal':nn.init.xavier_normal_,
    'xavier-uniform':nn.init.xavier_uniform_,
    'kaiming-normal':lambda x :nn.init.kaiming_normal_(x,nonlinearity='relu'),
    'kaiming-uniform':lambda x :nn.init.kaiming_uniform_(x,nonlinearity='relu'),
    'leaky-kaiming-normal':nn.init.kaiming_normal_,
    'leaky-kaiming-uniform':nn.init.kaiming_uniform_
}

actv_funcs = {
    'sigmoid':torch.sigmoid,
    'tanh':torch.tanh,
    'relu':F.relu,
    'elu':F.elu,
    'leaky-relu':F.leaky_relu,
    'rrelu':F.rrelu
}

def init_weights(m, fn):
    if hasattr(m,'weight'):
        fn(m.weight)

def get_out_dims(
        inputs: np.ndarray,
        padding: np.ndarray,
        dilation: np.ndarray,
        kernel: np.ndarray,
        stride: np.ndarray
    )->np.ndarray:
    """
    calculate the output dimensions of a `nn.Conv3d` layer
    :param inputs: 3 element vector for shape of input
    :param padding: 3 element vector for padding parameters
    :param dilation: 3 element vector for dilation parameters
    :param kernel: 3 element vector for kernel parameters
    :param stride: 3 element vector for stride parameters 
    """
    return np.floor(1+(inputs-1+(2*padding)-(dilation*(kernel-1)))/stride).astype(np.int32)

def get_even_padding(
        inputs: np.ndarray,
        dilation: np.ndarray,
        kernel: np.ndarray,
        stride: np.ndarray,
        preserve_inputs: bool = False
    )->np.ndarray:
    """
    Calculate the padding vector necessary to ensure perfect overlap of 
    kernel application with input tensor.
    :param inputs: 3 element vector for shape of input
    :param dilation: 3 element vector for dilation parameters
    :param kernel: 3 element vector for kernel parameters
    :param stride: 3 element vector for stride parameters 
    :param preserve_inputs: require output dimensions to be unchanged
    """
    if preserve_inputs and np.sum(stride == 1) != 3:
        raise ValueError("If input dimension is to be preserved, stride should not be > 1.")

    if preserve_inputs:
        return (dilation*(kernel-1)).astype(np.int32)
    else:
        return ((dilation*(kernel-1)+1-inputs)%stride).astype(np.int32)


class MAPnet(nn.Module):

    def __init__(
            self, 
            input_shape: Tuple[int,int,int],
            n_conv_layers: Optional[int]=CONV_LAYERS,
            padding: Optional[List[int]]=[PADDING],
            dilation: Optional[List[int]]=[DILATION],
            kernel: Optional[List[int]]=[KERNEL_SIZE],
            stride: Optional[List[int]]=[STRIDE],
            filters: Optional[List[int]]=FILTERS,
            input_channels: Optional[int]=1,
            conv_actv: Optional[List[Callable[[nn.Module],nn.Module]]]=[F.relu],
            fc_actv: Optional[List[Callable[[nn.Module],nn.Module]]]=[F.relu,F.tanh,F.relu],
            even_padding: Optional[bool]=False,
            pool: Optional[str]=None
        ):
        """
        Initialize an instance of MAPnet.
        :param input_shape: shape of input image.
        :param n_conv_layers: number of `nn.Conv3d` layers to use.
        :param padding: `padding` parameter for `nn.Conv3d`.
        :param dilation: `dilation` parameter for `nn.Conv3d`.
        :param kernel: `kernel_size` parameter for `nn.Conv3d`.
        :param stride: `stride` parameter for `nn.Conv3d`.
        :param filters: List of filters per layer.
        :param input_channels: Number of input channels to the model. 
        :param conv_actv: List of of activation functions to be used in 
        convolutional layers.  If only one element is supplied, then this
        activation function will be used for all layers.
        :param fc_actv: List of activation functions to be used in
        fully conneced layers.  If only one element is supplied, then this
        activation function will be used for all layers.
        :param even_padding:  setting this to True will result in the padding 
        parameter being ignored. Padding will be added to the input of each
        convolutional layer to ensure convolutions line up exactly with
        the input. Furthermore, layers with stride == 1 will have their
        input dimensions preserved in the output.
        :param pool: which pooling method to apply ('max' or 'avg').  If None,
        no pooling will be applied (which is the default).
        Pooling will be performed with a kernel size and stride of 2,
        and padding will be added to ensure the whole of the input is used.
        If pool = 'avg', padding will not be used to calculate the average.
        """
        #######################################################################
        # Sanitizing input
        #######################################################################
        if len(input_shape) != 3:
            raise ValueError("Expected input_shape to have 3 dimensions not {}".format(len(input_shape)))
        elif not ((len(filters) == n_conv_layers) or (len(filters) == 1)):
            raise ValueError("Length of filters ({}) does not match n_conv_layers ({})".format(len(filters),n_conv_layers))
        elif not ((len(conv_actv) == 1) or (len(conv_actv) == n_conv_layers)):
            raise ValueError("conv_actv arguments has incorrect length")
        elif not ((len(padding) == 1) or (len(padding) == n_conv_layers)):
            raise ValueError("padding arguments has incorrect length")
        elif not ((len(dilation) == 1) or (len(dilation) == n_conv_layers)):
            raise ValueError("dilation arguments has incorrect length")
        elif not ((len(kernel) == 1) or (len(kernel) == n_conv_layers)):
            raise ValueError("kernel arguments has incorrect length")
        elif not ((len(stride) == 1) or (len(stride) == n_conv_layers)):
            raise ValueError("stride arguments has incorrect length")
        elif not ((len(fc_actv) == 1) or (len(fc_actv) == 3)):
            raise ValueError("conv_actv arguments has incorrect length")
        
        super(MAPnet,self).__init__()

        #######################################################################
        # Handle the case where only 1 number is supplied
        #######################################################################
        if len(filters) == 1:
            filters = list(np.repeat(filters,n_conv_layers))
        if len(padding) == 1:
            padding = list(np.repeat(padding,n_conv_layers))
        if len(dilation) == 1:
            dilation = list(np.repeat(dilation,n_conv_layers))
        if len(kernel) == 1:
            kernel = list(np.repeat(kernel,n_conv_layers))
        if len(stride) == 1:
            stride = list(np.repeat(stride,n_conv_layers))

        # note that conv_layer_sizes with have length n_conv_layers + 1
        # because it also holds the input shape
        self.conv_layer_sizes = list([np.array(input_shape)])
        self.even_padding = even_padding # We will need this later
        self.pool = False if pool is None else True
        self.pool_layer_sizes = list()
        #######################################################################
        # Calculate layer sizes and padding if needed
        #######################################################################
        for i in range(0,n_conv_layers):
            ###################################################################
            # *** This bit is a little complicated, so I might leave a detailed
            # note explaining this next little section of code ***
            #
            # When the normal `padding` parameter is used, padding will be 
            # implemented through the Conv3d layers; HOWEVER, when 
            # `even_padding` is set, we must be aple to pad with an odd number
            # of zeros (i.e only on one side).  Thus we will need to use a
            # `torch.nn.ConstPad3d` layer to get the necessary size.
            #
            # Furthermore there is an extra layer of messyness introduced by
            # my attempt to allow for supbooling
            ###################################################################
            if even_padding:
                #preserve = True if stride[i] == 1 else False
                preserve = False
                # TODO:
                # This is really bad... I'm changing types from
                # int to list/array here.  Find a better/more clear way!
                padding[i] = get_even_padding(
                    inputs = self.pool_layer_sizes[-1] if self.pool and i \
                            else self.conv_layer_sizes[-1],
                    dilation = np.repeat(dilation[i],3),
                    kernel = np.repeat(kernel[i],3),
                    stride = np.repeat(stride[i],3),
                    preserve_inputs = preserve
                )
                if (self.pool is None) or (i == 0):
                    self.conv_layer_sizes[-1] += padding[i]
                else:
                    self.pool_layer_sizes[-1] += padding[i]

            self.conv_layer_sizes.append(
                get_out_dims(
                    self.pool_layer_sizes[-1] if self.pool and i \
                        else self.conv_layer_sizes[-1], # input dimensions    
                    np.repeat(0 if even_padding else padding[i],3), # *padding
                    np.repeat(dilation[i],3), # dilation
                    np.repeat(kernel[i],3), # kernel
                    np.repeat(stride[i],3) # stride
                )
            )
            if self.pool is not None:
                self.pool_layer_sizes.append(
                    get_out_dims(
                        self.conv_layer_sizes[-1],
                        self.conv_layer_sizes[-1]%2,
                        np.repeat(1,3),
                        np.repeat(2,3),
                        np.repeat(2,3)
                    )
                )

        #######################################################################
        # Initialize Conv3d & Pooling layers
        #######################################################################
        conv_layers = list()
        pool_layers = list()
        pad_layers = list()
        self.conv_actv = list()
        self.n_channels=list([input_channels])
        for i in range(0,n_conv_layers):
            self.n_channels.append(self.n_channels[-1] * filters[i])

            conv_layers.append(
                nn.Conv3d(
                    in_channels=int(self.n_channels[-2]), 
                    out_channels=int(self.n_channels[-1]), 
                    kernel_size=kernel[i], 
                    stride=stride[i], 
                    padding=0 if even_padding else padding[i], 
                    dilation=dilation[i], 
                    groups=int(self.n_channels[-2]), 
                    bias=True, 
                    padding_mode='zeros'
                )
            )
            if self.pool:
                if pool == 'max':
                    pool_layers.append(
                        nn.MaxPool3d(2,
                            padding=tuple((self.conv_layer_sizes[i+1]%2).astype(np.int32)))
                    )
                else:
                    pool_layers.append(
                        nn.AvgPool3d(2,padding=tuple((self.conv_layer_sizes[+1]%2).astype(np.int32)),
                            count_include_pad=False)
                    )
                
            # Manage our activation functions
            self.conv_actv.append(conv_actv[i] if len(conv_actv) > 1 else conv_actv[0])

            # And if `even_padding` was set, we will need to generate these layers 
            # but the complication here is that we now need to calculate
            # (P_Left,P_Right,P_Up,P_Down,P_Front,P_Back)
            if even_padding:
                d,h,w = padding[i] 
                pad = (int(np.floor(d/2)), int(np.ceil(d/2)), int(np.floor(h/2)), 
                        int(np.ceil(h/2)), int(np.floor(w/2)), int(np.ceil(w/2)))
                pad_layers.append(nn.ConstantPad3d(pad,0))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)
        self.pad_layers = nn.ModuleList(pad_layers) if even_padding else None
        
        #######################################################################
        # Initialize Fully Connected layers
        #######################################################################
        # calculate the size of flattening out the last conv layer
        layer_size = self.pool_layer_sizes[-1] if self.pool \
                else self.conv_layer_sizes[-1]
        self.fc_input_size  = int(np.prod(layer_size))*self.n_channels[-1]

        """
        print(self.conv_layer_sizes)
        print(self.pool_layer_sizes)
        print(self.fc_input_size)
        """

        fc_layers = list()

        fc_layers.append(nn.Linear(self.fc_input_size,int(self.fc_input_size/2)))
        fc_layers.append(nn.Linear(int(self.fc_input_size/2),100))
        fc_layers.append(nn.Linear(100,1))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_actv = fc_actv * 3 if len(fc_actv) == 1 else fc_actv
        
        self.d1 = nn.Dropout()

    def forward(self,x): 
        for i,conv in enumerate(self.conv_layers):
            if self.even_padding:
                x = self.pad_layers[i](x)
            actv = self.conv_actv[i]
            x = actv(conv(x))
            if self.pool:
                x = self.pool_layers[i](x)

        x = x.view(-1,self.fc_input_size)
        x = self.d1(x)
        for i,fc in enumerate(self.fc_layers):
            actv = self.fc_actv[i]
            x = actv(fc(x))

        return x

