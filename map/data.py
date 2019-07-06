import os
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np

def get_subject_dict(data_folder_path,dataset='train'):
    """
    Return a dictionary of subject IDs and nifti files
    :param data_folder_path: path to the data folder
    :param dataset: either 'train' or 'test'
    """
    if (dataset != 'train') and (dataset != 'test'):
        raise ValueError('Invalid dataset')
    if not os.isdir(os.path.join(data_folder_path,dataset)):
        raise ValueError('Invalid data path')


    subject_dict = {}
    dataset_path = os.path.join(data_folder_path,dataset)
    for subject in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path,subject)
        # For now we'll just take the first nifti file
        subject_dict[subject] = glob(subject_path,'*.nii*')[0]
    return subject_dict 

def check_subject_folder(path):
    return False

class NiftiDataLoader(Dataset)

    def __init__(self,files,labels=None):

        pass

    def __getitem__(self, index):

    def __len__(self):
