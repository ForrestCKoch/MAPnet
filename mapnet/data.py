from typing import Any, Optional, Tuple, List, Dict
import os
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_filter1d


def get_sample_dict(
        datapath: str,
        dataset: Optional[str] = 'train'
    ) -> Dict[str,List[str]]:
    """
    Return a dictionary of subject IDs and nifti files
    :param datapath: path to the data folder
    :param dataset: either 'train' or 'test'
    """
    if (dataset != 'train') and (dataset != 'test') and (dataset != 'validate'):
        raise ValueError('Invalid dataset')
    if not os.path.isdir(os.path.join(datapath,dataset)):
        raise ValueError('Invalid data path')

    subject_dict = {}
    dataset_path = os.path.join(datapath,dataset)
    for subject in os.listdir(dataset_path):
        glob_pattern = os.path.join(dataset_path,subject,'*.nii*')
        subject_dict[subject] = glob(glob_pattern)
    return subject_dict 

def get_sample_ages(
        ids : List[str],
        path_to_csv: str
    ) -> np.ndarray:
    """
    Return the ages of the requested IDs.
    :param ids: a list of subject ids
    :param path_to_csv: a path to a csv containing ages for each subject.
    This file must have a header reading "id","age" and each successive line
    is a tuple of of the form "id,age" 
    """
    with open(path_to_csv,'r') as fh:
        fh.readline()
        id_to_age = {}
        for line in fh:
            line = line.rstrip('\n')
            sid,age = line.split(',')
            try:
                id_to_age[sid] = float(age)
            except:
                pass
    return np.array([id_to_age[i] for i in ids],dtype=np.float32)
    
def encode_age_ordinal(
        age: float,
        bins: np.ndarray,
    )->torch.Tensor:
    """
    Encode age as an ordinal binary vector according to the provided list.
    Given a List [x_1 < x_2 < ... < x_n] a binary vector of
    length n will be returned where each element is 1
    if it is less than or equal to the give age and 0 otherwise.
    :param age: age to be encoded
    :param bins: List of bin boundaries 
    """

    return torch.from_numpy(np.array(bins <= age,dtype=np.float32))

def encode_age_nonordinal(
        age: float,
        bins: np.ndarray,
    )->torch.Tensor:
    """
    Encode age as a non-ordinal binary vector according to the provided list.
    Given a List [x_1 < x_2 < ... < x_n ], a binary vector of
    length n+1 will be returned where the ith element is equal to 1 if
    it is less than x_i, but greater than x_{i-1}. Thus the first bin
    will capture all values [-inf,x_1], and the n+1th bin will
    capture all values [x_n,inf]
    :param age: age to be encoded
    :param bins: List of bin boundaries 
    """
    upr_bins = np.append(bins,np.inf)
    lwr_bins = np.roll(upr_bins,1)
    lwr_bins[0] = np.NINF
    return torch.Tensor(np.array((age >= lwr_bins)*(age < upr_bins),dtype=np.float32))

def encode_smooth_age(
        age: int,
        bins: np.ndarray,
        sigma: Optional[float] = 0.7
    )->torch.Tensor:
    """
    First uses `encode_age_nonordinal` to construct an embedding, and then applies
    a 1d Gaussian filter with parameter `sigma` to smooth the ages.
    :param age: age to be encoded
    :param bins: List of bin boundaries 
    :param sigma: parameter for gaussian filter
    """
    return torch.Tensor(gaussian_filter1d(
        encode_age_nonordinal(age,bins),
        sigma,
        mode='constant',
        cval=0
    ))
    

def check_subject_folder(path):
    return False

def scale_image(img: np.ndarray)->np.ndarray:
    """
    Return a copy of the array where all non-zero values have been scaled
    between 0 and 1
    :param img: img to be scaled
    """
    mn = np.min(img)
    mx = np.max(img)
    return ((img-mn)/(-mn+mx))*(img != 0)
    

class NiftiDataset(Dataset):

    def __init__(
            self,
            samples: Dict[str,List[str]],
            labels: Any = None,
            scale_inputs: bool = False,
            cache_images: bool = False
        ):
        """
        Generate a Torch-style Dataset from a list of samples and list of labels
        :param samples: a dict of lists -- the list should be a set of
        files for the sample, and each key of the dict represents each sample
        :param labels: If not None, this should be a list of same length as
        samples, with each label being the supervised label of the corresponding
        sample
        :param cache_images: If True, nibabel will be allowed to cache images in
        memory.  Defaults to False (images will be read from disk each time the
        are requested).
        """
        super(NiftiDataset,self).__init__()
        if (len(samples.keys()) != len(labels)) and labels is not None:
            raise ValueError("Number of samples ({}) does not equal number of labels ({}).".format(len(samples.keys()),len(labels)))
        self.scale_inputs = scale_inputs
        self.cache_images = cache_images
        self.labels = labels
        self.samples = list()
        for indv in samples:
            # load an nibabel image object for each file
            self.samples.append([nib.load(fname) for fname in samples[indv]])

        # Perform some sanity checks on the input files
        self.images_per_subject = None
        self.image_shape = None
        for indv in self.samples:
            nimages = len(indv)
            if self.images_per_subject is None:
                self.images_per_subject = nimages
            elif self.images_per_subject != nimages:
                raise ValueError("Inconsistent number of files for each subject")

            for img in indv:
                if self.image_shape is None:
                    self.image_shape = img.shape
                elif self.image_shape != img.shape:
                    raise ValueError("Inconsistent shapes between images")

    def __getitem__(self, index):
        if self.labels is None:
            label = None
        else:
            label = self.labels[index]
        indv = self.samples[index]

        if self.scale_inputs:
            img_array = np.concatenate(
                    [[scale_image(img.get_fdata(dtype=np.float32))] for img in indv],
                    axis=0
            )
        else:
            img_array = np.concatenate(
                    [[img.get_fdata(dtype=np.float32)] for img in indv],
                    axis=0
            )

        ret = (torch.from_numpy(img_array),label)
        # remove the cached image unless the `cache_image` flag is set
        if not self.cache_images:
            for img in indv:
                img.uncache()
        return ret

    def __len__(self):
        return len(self.labels)
