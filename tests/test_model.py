from mapnet.data import *
from mapnet.model import *
from torch.utils.data import DataLoader

sd = get_sample_dict('data',dataset='train')
ages = get_sample_ages(sd.keys(),'data/subject_info.csv')

ds = NiftiDataset(sd,ages)
dl = DataLoader(ds,num_workers=8,pin_memory=False,batch_size=1)

x,l = ds[0]

model = MAPnet(ds.image_shape)
model = model.float()

for batch,labels in dl:
    y = model.forward(batch.float())
    print(y)
