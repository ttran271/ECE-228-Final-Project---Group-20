import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import nibabel as nib
import cv2
import os

START = 52
NUM_SLICE = 50
NUM_CLASSES = 4
main_path = "./MICCAI_BraTS2020_TrainingData"

# Cropping coordinates
SR = 38
ER = 202
SC = 25
EC = 225

# No cropping
SR = 0
ER = 240
SC = 0
EC = 240

class BraTS_Data(Dataset):
    def __init__(self, paths, start=60, num_slice=40, img_size=128, channels=2, sr=0, er=240, sc=0, ec=240):
        self.paths = paths
        self.img_size = img_size
        self.channels = channels
        self.num_slice=num_slice
        self.start = start
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.SR = sr
        self.ER = er
        self.SC = sc
        self.EC = ec
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        curr_path = self.paths[idx]
        img_path = os.path.join(main_path, curr_path, curr_path)

        t1ce = nib.load(img_path + '_t1ce.nii').get_fdata()
        flair = nib.load(img_path + '_flair.nii').get_fdata()
        seg = nib.load(img_path + '_seg.nii').get_fdata()
        
        X1 = self.transforms(cv2.resize(t1ce[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                        (self.img_size, self.img_size)))
        X2 = self.transforms(cv2.resize(flair[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                        (self.img_size, self.img_size)))
        X = torch.stack([X1, X2], dim=1)
        y = self.transforms(cv2.resize(seg[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                       (self.img_size, self.img_size)))

        X = ((X-torch.min(X))/(torch.max(X)-torch.min(X))).float()

        y[y==4] = 3
        y0 = torch.where(y==0, 1, 0)
        y1 = torch.where(y==1, 1, 0)
        y2 = torch.where(y==2, 1, 0)
        y3 = torch.where(y==3, 1, 0)
        y_oh = torch.stack([y0, y1, y2, y3], dim=1)

        return X, y_oh
    
########## TESTING ##########
### Dataset but includes T1CE, T2, FLAIR
class BraTS_Data_Test(Dataset):
    def __init__(self, paths, start=60, num_slice=40, img_size=128, channels=2, sr=0, er=240, sc=0, ec=240):
        self.paths = paths
        self.img_size = img_size
        self.channels = channels
        self.num_slice=num_slice
        self.start = start
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.SR = sr
        self.ER = er
        self.SC = sc
        self.EC = ec

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        curr_path = self.paths[idx]
        img_path = os.path.join(main_path, curr_path, curr_path)

        t1ce = nib.load(img_path + '_t1ce.nii').get_fdata()
        t2 = nib.load(img_path + '_t2.nii').get_fdata()
        flair = nib.load(img_path + '_flair.nii').get_fdata()
        seg = nib.load(img_path + '_seg.nii').get_fdata()
        
        X1 = self.transforms(cv2.resize(t1ce[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                        (self.img_size, self.img_size)))
        X2 = self.transforms(cv2.resize(t2[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                        (self.img_size, self.img_size)))
        X3 = self.transforms(cv2.resize(flair[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                        (self.img_size, self.img_size)))
        X = torch.stack([X1, X2, X3], dim=1)
        y = self.transforms(cv2.resize(seg[self.SR:self.ER,self.SC:self.EC,self.start:self.start+self.num_slice], \
                                       (self.img_size, self.img_size)))

        X = ((X-torch.min(X))/(torch.max(X)-torch.min(X))).float()

        y[y==4] = 3
        y0 = torch.where(y==0, 1, 0)
        y1 = torch.where(y==1, 1, 0)
        y2 = torch.where(y==2, 1, 0)
        y3 = torch.where(y==3, 1, 0)
        y_oh = torch.stack([y0, y1, y2, y3], dim=1)

        return X, y_oh