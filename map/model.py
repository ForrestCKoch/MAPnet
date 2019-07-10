from typing import Any, Optional, Callable
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_out_dims(inputs: np.ndarray,
                 padding: np.ndarray,
                 dilation: np.ndarray,
                 kernel: np.ndarray,
                 stride: np.ndarray)->np.ndarray:
    """
    calculate the output dimensions of a Conv3d layer
    :param inputs: 3 element vector for shape of input
    :param padding: 3 element vector for padding parameters
    :param dilation: 3 element vector for dilation parameters
    :param kernel: 3 element vector for kernel parameters
    :param stride: 3 element vector for stride parameters 
    """
    
    return np.floor(1+(inputs-1+(2*padding)-(dilation*(kernel-1)))/stride)

class MAPnet(nn.Module):

    def __init__(self,input_shape: tuple):
        """
        Initialize an instance of MAPnet
        :param input_shape: shape of input image
        """
        super(MAPnet,self).__init__()
        self.conv_layer_sizes = list([np.array(input_shape)])
        for i in range(0,4):
            self.conv_layer_sizes.append(
                get_out_dims(
                    self.conv_layer_sizes[-1], # input dimensions    
                    np.repeat(0,3), # padding
                    np.repeat(1,3), # dilation
                    np.repeat(3,3), # kernel
                    np.repeat(3,3), # stride
                )
            )

        n_filters = [2,3,4,4]
        conv_layers = list()
        self.n_channels=list([1])
        for i in range(0,4):
            self.n_channels.append(self.n_channels[-1] * n_filters[i])
            conv_layers.append(
                nn.Conv3d(
                    in_channels=self.n_channels[-2], 
                    out_channels=self.n_channels[-1], 
                    kernel_size=3, 
                    stride=3, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros'
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # calculate the size of flattening out the last conv layer
        layer_size = self.conv_layer_sizes[-1]
        self.fc_input_size  = int(np.prod(layer_size))*self.n_channels[-1]
        self.fc1 = nn.Linear(self.fc_input_size,100) 
        self.fc2 = nn.Linear(100,1)

    def forward(self,x): 
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.view(-1,self.fc_input_size)
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
