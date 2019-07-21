from mapnet.data import *
from tqdm import tqdm
from torch.utils.data import DataLoader

sd = get_sample_dict('data',dataset='train')
ages = get_sample_ages(sd.keys(),'data/subject_info.csv')

ds = NiftiDataset(sd,ages)
dl = DataLoader(ds,num_workers=8,pin_memory=True,batch_size=32)

for i in range(0,10):
    for batch,sample in tqdm(enumerate(dl)):
        pass
