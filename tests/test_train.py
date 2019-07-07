from map.data import *
from map.model import *
from map.train import *
from torch.utils.data import DataLoader

train_dict = get_sample_dict('data',dataset='train')
train_ages = get_sample_ages(train_dict.keys(),'data/subject_info.csv')
train_ds = NiftiDataset(train_dict,train_ages)

test_dict = get_sample_dict('data',dataset='train')
test_ages = get_sample_ages(test_dict.keys(),'data/subject_info.csv')
test_ds = NiftiDataset(test_dict,test_ages)

model = MAPnet(train_ds.image_shape)

train(train_ds,test_ds,model,cuda=True,batch_size=8,num_workers=8,epochs=50,update_freq=10)
