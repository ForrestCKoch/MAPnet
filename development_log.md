# Model Development Log

## 14/07/19

Despite beginning development about 1 week ago, I am just now beginning to
log my progress in a diary. The majority of the work done up to this point
was to lay down the boilerplate/general framework to allow me to experiment
with different models

### Aim:

The aim of this project is to accurately predict age from an MRI
image using convolutional neural networks.

### Brief Explanation of Code "Framework":

* `NiftiDataset` (`torch.utils.data.Dataset`) -- Used to store nifti dataset.
It can be used with `torch.utils.data.DataLoader` for multithreaded loading.
    * There are a few helper functions implemented in data.py that are intended
    to help with constructing the dataset.  They expect a folder structure of
    ```
    data/
    ├── subject_info.csv
    ├── train/
    │   ├── sub1/
    │   │   ├── sub1_image1.nii.gz
    │   │   ├── ...
    │   │   └── sub1_imageN.nii.gz
    │   ├── ...
    │   ├── subM/
    │   │   ├── subM_image1.nii.gz
    │   │   ├── ...
    │   │   └── subM_imageN.nii.gz
    ├── test/
    │   ├── ...
    ```
* `MAPnet` (`torch.nn.Module`) -- A very general/flexible basic Conv3d network.
    * Caller can specify number of Conv3d layers and their parameters
    * The helper function `get_out_dims(...)` can be used to calculate the
    dimensions of the layer output given certain parameters
    * Can input multiple channels/modalities
* `train.py` -- Implements the main training loop. When run as the main script,
it allows the user to specify a variety of options (run `python3 train.py -h` 
for help).

### Datsets I'm currently using:

* For initial testing purposes, I put together a small dataset of T1 scans from
UK Biobank (UKBB) data. There are ~300 scans in the train set and 23 scans in
the test set.  This is only a small sample of the available data and is 
intended for development/testing purposes.  If used for serious training later,
I will encorporate more images and do a proper test/train/validate split.
    * Input images are 182x218x182
* For model development, I am currently using a DWI dataset from UKBB data
consisting of 17915 subjects.
    * Each subject contains four images which are DTI derivatives -- FA, MD, AD
    RD (refer to [link](http://www.diffusion-imaging.com/2013/01/relation-between-neural-microstructure.html))
    for description). Each image is used as a separate channel.
    * Each input image is 104x104x72
    * Data is divided into Train (13915), Test (2000), and Validate (2000) sets.
    The Validate set won't be touched until assessing my final model accuracy.

### Initial development notes/results:

I forgot to note down the very first model architecture used for testing on the
T1 dataset.  I believe it was a 4 convolutional layers, with stride 3, padding 2, 
kernel size of 5, and dilation of 1 follwed by 2 fully connected layers
with the output being a single value.  Relu was used as an activation function
for all layers except for the second to last FC layer which used sigmoid.  
MSE loss with an Adam optimizer was used for training.  I cannot remember learning
rate; however, the model was unimpressive, only learning to predict the mean age.

Ultimately, this is okay as I can't magically expect the model to work prefectly.
I will clearly need to put some more effor into architecture design.  I did run 
into some issues when moving over to the larger DWI dataset.  When the learning
rate was too high, the model failed to reduce error -- it could not even learn 
the mean.  When the learning rate was low enough to reduce error, it was 
incredibly slow.  Even after 80 epochs, it could not predict the mean.

I have two ideas for what was causing this poor learning performance on the DWI data.
1. The value range of the DWI derivatives is much smaller than the T1 images used 
previously.
2. The value range of the DWI derivatives is inconsistent (e.g FA is [0,1.2] 
whereas MD is [-0.006,0.006].
3. Age values range from 40-70 years of age.  Given small weight adjustments, it
may take longer to increase output values?  I'm not too convinced this last point
is an issue, however, as the gradient calculation should scale with the amount of
error ...

My solution was two-fold.  I scale each of the input images as follows:
```
mn = np.min(img)
mx = np.max(img)
sc = ((img-mn)/(-mn+mx))*(img != 0)
```
Note that I am only scaling non-zero voxels here.  This is because I don't
want to affect the background of the image -- just the brain itself.

The second thing I do is to divide the ages by 100, so they are in the range
[0.4,0.7].

This seems to do the trick, and with the following architecture and learning
rate of 0.0001:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 16, 35, 35, 24]           2,016
            Conv3d-2        [-1, 64, 12, 12, 8]           8,064
            Conv3d-3         [-1, 256, 4, 4, 3]          32,256
           Dropout-4                [-1, 12288]               0
            Linear-5                 [-1, 6144]      75,503,616
            Linear-6                  [-1, 100]         614,500
            Linear-7                    [-1, 1]             101
================================================================
Total params: 76,160,553
Trainable params: 76,160,553
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 11.88
Forward/backward pass size (MB): 4.39
Params size (MB): 290.53
Estimated Total Size (MB): 306.80
----------------------------------------------------------------
```

The model quickly converges on predicted the mean:
```
Epoch: 0 Test Loss: 0.e+00 Train Loss: 9.272e-02: 100%|██████████████████████████████████████████████| 218/218 [13:58<00:00,  3.18s/it]
Epoch: 1 Test Loss: 5.409e-03 Train Loss: 5.577e-03: 100%|███████████████████████████████████████████| 218/218 [14:12<00:00,  3.14s/it]
Epoch: 2 Test Loss: 5.411e-03 Train Loss: 5.576e-03: 100%|███████████████████████████████████████████| 218/218 [14:06<00:00,  3.35s/it]
Epoch: 3 Test Loss: 5.359e-03 Train Loss: 5.574e-03: 100%|███████████████████████████████████████████| 218/218 [14:52<00:00,  3.37s/it]
```

### Thoughts moving forward
* See if there are efficiency gains to be made to decrease training time
* Implement subpooling to allow for more fine-grain convolutions and smaller FC
* Weight initialization
* Loss functions / age encoding

## 15/07/19

### Model Experiment:

Last night I had the idea that instead of adding pooling (max or average),
I could instead just use strided convolutions.  A quick google search showed that
this is in fact done 
(see this stackoverflow [discussion](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling))

So, this morning I started training a model with the following call:
```
python3.6 map/train.py \
    --datapath dwi_data/ \
    --savepath models/ \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --workers 32 \
    --scale-inputs \
    --cuda \
    --stride 1 4 1 3 1 4 3\
    --padding 2 2 1 1 2 1 1\
    --conv-layers 7 \
    --filters 4 1 4 1 4 1 2\
    --kernel-size 4 4 3 3 4 4 2 \
    --conv-actv elu \
    --fc-actv elu
```
The parameter sizes were somewhat lazily chosen last night in an attempt to
preserve layer size inbetween non-striding convolutions.  I will try to
implement a "smart" padding method to make this easier to achieve ...

Model Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1     [-1, 16, 105, 105, 73]           1,040
            Conv3d-2       [-1, 16, 27, 27, 19]           1,040
            Conv3d-3       [-1, 64, 27, 27, 19]           1,792
            Conv3d-4          [-1, 64, 9, 9, 7]           1,792
            Conv3d-5       [-1, 256, 10, 10, 8]          16,640
            Conv3d-6         [-1, 256, 3, 3, 2]          16,640
            Conv3d-7         [-1, 512, 2, 2, 1]           4,608
           Dropout-8                 [-1, 2048]               0
            Linear-9                 [-1, 1024]       2,098,176
           Linear-10                  [-1, 100]         102,500
           Linear-11                    [-1, 1]             101
================================================================
Total params: 2,244,329
Trainable params: 2,244,329
Non-trainable params: 0
```

Thirty epochs in, it appears to be making some slight improvements over simply
predicting the mean.
```
Epoch: 30 Test Loss: 4.853e-03 Train Loss: 4.802e-03: 100%|██████████████████████████████████████████| 435/435 [11:24<00:00,  1.49s/it]
...
Epoch: 49 Test Loss: 4.191e-03 Train Loss: 3.921e-03: 100%█████████████████████████████████████████████████████████████████████████| 435/435 [11:10<00:00,  1.46s/it]
```

Note that if it were predicting the mean, loss would be around 5.6e-03.

## 18/07/19

Little progress has been made on training the model.  I extended the code to load in models to keep training them further.
The above mentioned model was trained for a further 100 epochs at a slightly lower learning rate (0.00001), and managed to 
reduce test loss to ~4.01e-03 I believe.  I forgot to save the terminal output, so I will write code to test saved models soon
to re-assess the perfomance.

I have also been working out how to automatically calculate layer sizes as well as incoporate max/avg pooling in an attempt to
make model specification easier from the command line script.  I made some pretty silly mistakes due to programming later at night
so I speant today bug fixing and making sure everything runs smoothly. Will include more information a later time/in the report.

## 19/07/19

So this is more of a continuation of last night's entry to expand on a few thoughts. I haven't yet explicitly stated my 
current approach to this project, so it's probably worth briefly discussing.

My first aim is to develop a generalized framework that will allow me to easily experiment with various Conv3d oriented 
model architectures using a commandline interface.  
This is a fairly open ended issue, and I must be careful not to spend too much time on this task alone.
That said there are a few requirements I feel this must meet.
* Flexible model architecture -- with little effort I would like to be able to specify the model architecture
    * Number of layers [ *implemented* ]
    * Number of filters [ *implemented* ]
    * Layer parameters (stride, dilation, padding , kernel size, ...) [ *implemented* ]
    * Activation Functions [ *implemented* ]
    * Pooling [ *implemented* ]
    * Weight initialization [ *implemented* ]
    * Skip connections? [ **not implemented** ] 
* Training options
    * Number of epochs [ *implemented* ]
    * Batch size [ *implemented* ]
    * Learning rate [ *implemented* ]
    * GPU training [ *implemented* ]
    * LR decay [ *implemented* ] 
    * Choice of loss function [ *implemented* ] 
* Updated:
* And a few other features for ease of use
    * Model saving/loading [ *implemented* ]
    * Keep traing previously saved models [ *implemented* ]
    * Test previouly saved/trained models on specific datasets [ *implemented* ]
    * Test/Train accuracy plots [ **not implemented** ]

## 21/07/19

I've modified the above list to reflect what I finished implemented (loss functions and scheduling). I've also added a few
options regarding the output of the model.
* Value output -- the standard output I've been using up to now.  Model simply predicts a single value (e.g age).
* Class output -- each age is considered a single class.
* Ordinal class output -- classes are ordinal, and so predicting a class should also mean predicting all 'younger' classes.
* Gaussian Smoothed -- A one-hot encoded vector for age is generated the same way as for 'Class output', however the resulting
1d vector is smoothed with a gaussian kernel.  The idea here is to train the network to predict a range of ages.

I've made some changes to the main script to try to make things a little more reproducible.  This involves:
* Recording the parameters the program is called with.
* Recording train/test loss and learning rate at each epoch in a csv.
* General formatting to main script output ...

To the side, I've also been experimenting with various model architectures, so I will make a few informal notes here:
* Max Pooling -- seems to work much better than using 3dConv with stride=kernel.
* Kernel size -- I tried increasing kernel size.  It may give good results; however, training time in painfully slow (~4x longer)
* LR decay -- very helpful, but this is entirely expected.

I am currently training a model (stored in models/2019-07-21_09-59-17/) with program call:
```
datapath:           dwi_data/
scale_inputs:       True
workers:            4   
savepath:           models/
save_freq:          1   
load_model:         None
conv_layers:        5   
kernel_size:        [5, 4, 4, 2, 2]
dilation:           [1] 
padding:            [2] 
even_padding:       True
stride:             [1] 
filters:            [2, 2, 2, 2, 2]
weight_init:        kaiming-uniform
conv_actv:          ['elu']
fc_actv:            ['elu']
pooling:            max 
lr:                 0.001
decay:              0.99
reduce_on_plateau:  False
batch_size:         32  
epochs:             200 
update_freq:        1   
cuda:               True
debug_size:         None
silent:             False
test_model:         None
encode_age:         False
```

Which has managed to achieve the best performance so far on the test set:
```
Epoch: 35 Test Loss: 3.548e-03 Train Loss: 2.958e-03 LR: 4.899e-04: 100%|████████████████████████████| 435/435 [14:00<00:00,  1.91s/it]
```
This corresponds to an 'average' error of around 6 years, which is an imporovement over the 'average' of 7.5 expected by the mean.
I say 'average' because it it mean squared error, meaning that larger discrepancies will be weighted more heavily.

My experimentation so far hasn't been very systematic.  I've been making tweaks to the model based on curiousity/intuition; 
however, my intention is to comprehensively explore the features I've laid out.  I believe the framework for this experimentaiton
is essentially finished, so now I can move into a testing phase.  Using the above model as a starting point, I plan on exploring.
* Initial LR
* LR decay (no decay vs every epoch vs on plateau)
* Batch size (which will likely have interaction effects with LR)
* Weight initlization
* Activation functions used in each layer
* The effect of different loss functions and model outputs
* Number of convolutional layers
* Number of filters (and which layers they should be added)
* Kernel size(s)

## 22/07/19

Quick side note -- I decided I wanted to add one more feature to play around with.  I have implemented a loss function based on the 
Wasserstein distance between two density functions.  The intention is for this to be used in conjuction with the 'gaussian' model 
output in order to learn a distribution function over age as opposed to a single value.  It is available by selecting `-loss Wasserstein`.
Note that the last fully connected layer should be specified to use softmax to produce a discrete probability density as its output.

# 27/07/19

17:40 -- Started running some tests to compare various learning rates and batch sizes (see `scripts/lr_test.sh`)

# 09/08/19

I've been pretty bad at updating this diary over the past few weeks as I have mainly been waiting on results to come through.  I will give an overview of my experiments here:

## Learning Rate & Batch Sizes
As mentioned in the previous log, I began by conducting a grid search over batch sizes (32,64,96,128), and learning rate (0.01,0.001,0.0001,0.00001).  The grid search is important here as the batch size can have effects on the optimal learning rate. Trials were run over 10 epochs.

These results indicated that I should use a batch size of 64 with learning rate of 0.0001.

### Activation Functions
PyTorch provides access to a large range of activation functions:
* relu
* elu
* rrelu
* leaky-relu
* sigmoid
* tanh

I conducted another grid search over each pairwise combination (36 trials).  One activation function was used for each of the convolutional layers, and the other was used for each of the fully connected layers. Trials were run over 20 epochs.

These results indicated that tanh should be used for the convolutional layers, and sigmoid should be used for the 3 fully connected layers.  I then conducted another series of tests allowing for a combination of activation functions in the fully connected layers as this is standard practice.  The fully connected layers took the form `X-X-sigmoid` where `X` is one of: relu,elu,rrelu,tanh (leaky-relu was exclued for poor performance in the previous test).

the `relu-relu-sigmoid` performed best in this second trial, but still not as well as the fully sigmoid option.  The `relu-relu-sigmoid` option seems to be more common in literature, so I will not exclude it as a possibility at this point.

## Learning Rate Decay
For each both the `relu-relu-sigmoid` and fully `sigmoid` options, I tested the effects over various amounds of learning rate decay.  At the end of each epoch, learning rate was reduced by a multiplicitive factor (no_decay,0.995,0.99,0.985,0.98,...,0.90).

No decay still seems to learn the fastest; however, 0.995 was quite close.  Given lr decay has proven benefits in much of the literature, I will consider models with both no decay and 0.995.

## MASSIVE PROBLEM FOUND
I just realized I've been using inaccurate ages for my data.  I have updated the data and will see how it affects accuracy.
