from map.data import *
from map.model import *
from map.train import *
from torch.utils.data import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_dict = get_sample_dict('dwi_data',dataset='train')
train_ages = get_sample_ages(train_dict.keys(),'dwi_data/subject_info.csv')
train_ds = NiftiDataset(train_dict,train_ages)

test_dict = get_sample_dict('dwi_data',dataset='test')
test_ages = get_sample_ages(test_dict.keys(),'dwi_data/subject_info.csv')
test_ds = NiftiDataset(test_dict,test_ages,cache_images=False)

model = MAPnet(train_ds.image_shape,input_channels=4)
#print(count_parameters(model))

train(train_ds,test_ds,model,cuda=True,batch_size=32,num_workers=8,epochs=100,update_freq=1)
