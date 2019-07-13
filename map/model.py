from typing import Any, Optional, Callable, Tuple, List
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_out_dims(
        inputs: np.ndarray,
        padding: np.ndarray,
        dilation: np.ndarray,
        kernel: np.ndarray,
        stride: np.ndarray
    )->np.ndarray:
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

    def __init__(
            self, 
            input_shape: Tuple[int,int,int],
            n_conv_layers: int = 3,
            padding: int = 2,
            dilation: int = 1,
            kernel: int = 5,
            stride: int = 3,
            filters: List[int] = [4,4,4],
            input_channels: int = 1,
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
        """
        if len(input_shape) != 3:
            raise ValueError("Expected input_shape to have 3 dimensions not {}".format(len(input_shape)))
        elif len(filters) != n_conv_layers:
            raise ValueError("Length of filters ({}) does not match n_conv_layers ({})".format(len(filters),n_conv_layers))
        
        super(MAPnet,self).__init__()
        self.conv_layer_sizes = list([np.array(input_shape)])
        for i in range(0,n_conv_layers):
            self.conv_layer_sizes.append(
                get_out_dims(
                    self.conv_layer_sizes[-1], # input dimensions    
                    np.repeat(padding,3), # padding
                    np.repeat(dilation,3), # dilation
                    np.repeat(kernel,3), # kernel
                    np.repeat(stride,3), # stride
                )
            )

        conv_layers = list()
        self.n_channels=list([input_channels])
        for i in range(0,n_conv_layers):
            self.n_channels.append(self.n_channels[-1] * filters[i])
            conv_layers.append(
                nn.Conv3d(
                    in_channels=self.n_channels[-2], 
                    out_channels=self.n_channels[-1], 
                    kernel_size=kernel, 
                    stride=stride, 
                    padding=padding, 
                    dilation=dilation, 
                    groups=self.n_channels[-2], 
                    bias=True, 
                    padding_mode='zeros'
                )
            )
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
        self.fc1 = nn.Linear(self.fc_input_size,int(self.fc_input_size/2)) 
        self.fc2 = nn.Linear(int(self.fc_input_size/2),100) 
        self.fc3 = nn.Linear(100,1)
        self.d1 = nn.Dropout()
        self.d2 = nn.Dropout()

    def forward(self,x): 
        for conv in self.conv_layers:
            x = F.sigmoid(conv(x))
        x = x.view(-1,self.fc_input_size)
        x = self.d1(x)
        x = F.relu(self.fc1(x))
        #x = self.d2(x)
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
