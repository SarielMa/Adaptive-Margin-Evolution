
import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as tv_transforms
#%%
'''
from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../../data/SVHN', split='train', download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../../data/SVHN', split='test', download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=False)
#%%
data_train=train_loader.dataset
X_train=[]
Y_train=[]
for n in range(0, len(data_train)):
    X_train.append(data_train[n][0].view(1, 3, 32, 32))
    Y_train.append(data_train[n][1])
X_train=torch.cat(X_train, dim=0)
Y_train=torch.tensor(Y_train, dtype=torch.int64)
#
data_test=test_loader.dataset
X_test=[]
Y_test=[]
for n in range(0, len(data_test)):
    X_test.append(data_test[n][0].view(1, 3, 32, 32))
    Y_test.append(data_test[n][1])
X_test=torch.cat(X_test, dim=0)
Y_test=torch.tensor(Y_test, dtype=torch.int64)
#%%
rng=np.random.RandomState(0)
idxlist=np.arange(0, X_train.shape[0])
rng.shuffle(idxlist)
idxlist_train=idxlist[0:65931] # 90% for training
idxlist_val=idxlist[65931:] # 10% for val
data={}
data['X_train']=X_train[idxlist_train]
data['Y_train']=Y_train[idxlist_train]
data['X_val']=X_train[idxlist_val]
data['Y_val']=Y_train[idxlist_val]
data['X_test']=X_test
data['Y_test']=Y_test
torch.save(data, '../../data/SVHN/svhn_data.pt')
'''
#%%



#%%
class MyDataset(torch_dataset):
    def __init__(self, X, Y, x_shape, return_idx=False, transform=None):
        self.X=X.detach()
        self.Y=Y.detach()
        self.x_shape=x_shape
        self.return_idx=return_idx
        self.transform = transform
    def __len__(self):
        #return the number of data points
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx].view(self.x_shape)
        if self.transform is not None:
            x=self.transform(x)        
        
        y = self.Y[idx]
        if self.return_idx == False:
            return x, y
        else:
            return x, y, idx

def get_dataloader(batch_size=128, num_workers=2, x_shape=(3,32,32), return_idx=(False, False, False)):
    data = torch.load('../../data/SVHN/svhn_data.pt')
    """
    transform=tv_transforms.Compose([tv_transforms.Pad(4, padding_mode='reflect'),
                                     tv_transforms.RandomCrop(32),
                                     tv_transforms.RandomHorizontalFlip(),
                                     tv_transforms.ToTensor()])
    """
    dataset_train = MyDataset(data['X_train'], data['Y_train'], x_shape, return_idx=return_idx[0], transform = None)
    dataset_val = MyDataset(data['X_val'], data['Y_val'], x_shape, return_idx=return_idx[1],
                             transform=None)
    dataset_test = MyDataset(data['X_test'], data['Y_test'], x_shape, return_idx=return_idx[2],
                              transform=None)
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    loader_val = torch_dataloader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader_train, loader_val, loader_test
#%%

