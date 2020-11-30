import os
import torch
import random

import numpy as np
import torchvision.transforms as transforms

from glob import glob
from torch.utils.data import Dataset, DataLoader

def get_data_from_dir(directory):  
    faultDir = 'fault'
    seisDir = 'seis'
    
    faultDir = os.path.join(directory, faultDir)
    seisDir = os.path.join(directory, seisDir)       
    
    faultList = sorted(glob(os.path.join(faultDir, '*.npy')))[:500]
    seisList = sorted(glob(os.path.join(seisDir, '*.npy')))[:500]
           
    return faultList, seisList

class DatasetFolder(Dataset):
    def __init__(self, root, transform=None, dim=64):
        self.root = root
        self.fault_sample_paths, self.seis_sample_paths = get_data_from_dir(root)        
                
        self.transform = transform
        self.dim = dim
        
        
    def __getitem__(self, index):
        ## Seismic is input and Fault is output
        seis_sample_path = self.seis_sample_paths[index]        
        fault_sample_path = self.fault_sample_paths[index]
        
        seis_sample = self.npy_loader(seis_sample_path, seis=True)   
      
        if self.transform is not None:
            seis_sample = self.transform(seis_sample)
                
        fault_sample = self.npy_loader(fault_sample_path, seis=False)
        
        dataPack = {'seismic': seis_sample, 'fault': fault_sample, 'seismic_path': seis_sample_path, 'fault_sample_path': fault_sample_path}
        
        return dataPack
                
    def npy_loader(self, path, seis=True):
        datData = np.load(path)
        
        assert len(datData.shape) == 3        
        ## Insert channel dimension
        datData = np.expand_dims(datData, 0) # [1, 64, 64, 64], [time slice, inline slice, xline slice]
       
        ## If seis image, normalize
        if seis:
            datData = (datData - np.mean(datData)) / np.std(datData)
        
        return datData
        
    def __len__(self):
        assert len(self.fault_sample_paths) == len(self.seis_sample_paths)
        return len(self.fault_sample_paths)
        
    def __repr__(self):
        str = 'Data root: {}'.format(self.root)
        str += '\t{} samples'.format(len(self))
        return str
    
'''Transformations'''
class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image.copy())
    
class VerticalFlip(object):    
    def __call__(self, image):     
        if np.random.uniform() > 0.5:        
            return image[:,:,::-1]
        else:
            return image
    
    
def dataloader(root, batch_size=5, train_dir='train', val_dir='validation'):
    transformations = transforms.Compose([VerticalFlip(), 
                                          ToTensor()])
    val_transformations = transforms.Compose([ToTensor()])
    
    trainDataset = DatasetFolder(root=os.path.join(root, train_dir),
                                 transform=transformations)
    
    train_dataloader = DataLoader(trainDataset, shuffle=True, num_workers=2, pin_memory=True,
                                  batch_size=batch_size, drop_last=False)
    
    valDataset = DatasetFolder(root=os.path.join(root, val_dir),
                               transform=val_transformations)
    
    val_dataloader = DataLoader(valDataset, shuffle=False, num_workers=2, pin_memory=True,
                                batch_size=batch_size, drop_last=False)
    
    return train_dataloader, val_dataloader