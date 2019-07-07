import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_out_dims(inputs,padding,dilation,kernel,stride):
    return np.floor(1+(inputs-1+(2*padding)-(dilation*(kernel-1)))/stride)

class MAPnet(nn.Module):

    def __init__(self,input_shape):
        super(MAPnet,self).__init__()
        self.conv_layer_sizes = list([np.array(input_shape)])
        for i in range(0,4):
            self.conv_layer_sizes.append(
                get_out_dims(self.conv_layer_sizes[-1],
                    np.array([0,0,0]),
                    np.array([1,1,1]),
                    np.array([3,3,3]),
                    np.array([3,3,3])
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
        fc_input_size  = int(np.prod(layer_size))*self.n_channels[-1]
        self.fc1 = nn.Linear(fc_input_size,100) 
        self.fc2 = nn.Linear(100,1)

    def forward(self,x): 
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.flatten()
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
