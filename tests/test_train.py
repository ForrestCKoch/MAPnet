from map.data import *
from map.model import *
from map.train import *
from torch.utils.data import DataLoader

sd = get_sample_dict('data',dataset='train')
ages = get_sample_ages(sd.keys(),'data/subject_info.csv')

ds = NiftiDataset(sd,ages)

model = MAPnet(ds.image_shape)

train(ds,model,cuda=True,batch_size=8,n_workers=8)
