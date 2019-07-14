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
    return np.floor(1+(inputs-1+(2*padding)-(dilation*(kernel-1)))/stride)

class MAPnet(nn.Module):

    def __init__(
            self, 
            input_shape: Tuple[int,int,int],
            n_conv_layers: int = CONV_LAYERS,
            padding: List[int] = [PADDING],
            dilation: List[int] = [DILATION],
            kernel: List[int] = [KERNEL_SIZE],
            stride: List[int] = [STRIDE],
            filters: List[int] = FILTERS,
            input_channels: int = 1,
            conv_actv: List[Callable[[nn.Module],nn.Module]] = [F.relu],
            fc_actv: List[Callable[[nn.Module],nn.Module]] = [F.relu,F.tanh,F.relu]
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
        """
        if len(input_shape) != 3:
            raise ValueError("Expected input_shape to have 3 dimensions not {}".format(len(input_shape)))
        elif len(filters) != n_conv_layers:
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

        # Handle the case where only 1 number is supplied
        if len(padding) == 1:
            padding = np.repeat(padding,n_conv_layers)
        if len(dilation) == 1:
            dilation = np.repeat(dilation,n_conv_layers)
        if len(kernel) == 1:
            kernel = np.repeat(kernel,n_conv_layers)
        if len(stride) == 1:
            stride = np.repeat(stride,n_conv_layers)

        self.conv_layer_sizes = list([np.array(input_shape)])
        for i in range(0,n_conv_layers):
            self.conv_layer_sizes.append(
                get_out_dims(
                    self.conv_layer_sizes[-1], # input dimensions    
                    np.repeat(padding[i],3), # padding
                    np.repeat(dilation[i],3), # dilation
                    np.repeat(kernel[i],3), # kernel
                    np.repeat(stride[i],3), # stride
                )
            )

        conv_layers = list()
        self.conv_actv = list()
        self.n_channels=list([input_channels])
        for i in range(0,n_conv_layers):
            self.n_channels.append(self.n_channels[-1] * filters[i])
            conv_layers.append(
                nn.Conv3d(
                    in_channels=self.n_channels[-2], 
                    out_channels=self.n_channels[-1], 
                    kernel_size=kernel[i], 
                    stride=stride[i], 
                    padding=padding[i], 
                    dilation=dilation[i], 
                    groups=self.n_channels[-2], 
                    bias=True, 
                    padding_mode='zeros'
                )
            )
            self.conv_actv.append(conv_actv[i] if len(conv_actv) > 1 else conv_actv[0])
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # calculate the size of flattening out the last conv layer
        layer_size = self.conv_layer_sizes[-1]
        self.fc_input_size  = int(np.prod(layer_size))*self.n_channels[-1]

        print("Conv3d sizes")
        print(self.conv_layer_sizes)
        print("Number of channels")
        print(self.n_channels)
        print("Output nodes of convolutions")
        print(self.fc_input_size)

        fc_layers = list()

        fc_layers.append(nn.Linear(self.fc_input_size,int(self.fc_input_size/2)))
        fc_layers.append(nn.Linear(int(self.fc_input_size/2),100))
        fc_layers.append(nn.Linear(100,1))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_actv = fc_actv * 3 if len(fc_actv) == 1 else fc_actv
        
        self.d1 = nn.Dropout()

    def forward(self,x): 
        for i,conv in enumerate(self.conv_layers):
            actv = self.conv_actv[i]
            x = actv(conv(x))

        x = x.view(-1,self.fc_input_size)
        x = self.d1(x)
        for i,fc in enumerate(self.fc_layers):
            actv = self.fc_actv[i]
            x = actv(fc(x))

        return x
