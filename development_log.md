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
        RD (refer to [link](http://www.diffusion-imaging.com/2013/01/relation-between-neural-microstructure.html)
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
    Epoch: 3 Test Loss: 5.359e-03 Train Loss: 5.574e-03: 100%|███████████████████████████████████████████| 218/218 [14:52<00:00,  3.37s/it
    ```

