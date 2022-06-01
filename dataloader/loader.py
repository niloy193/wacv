import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Get_Patch
import os

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, cfg, is_patch = True):
        'Initialization'
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.cfg = cfg

        self.list_IDs_dir = cfg['dataset_params']['images_dir']
        self.labels_dir = cfg['dataset_params']['mask_dir']
            
        self.normalize = transforms.Normalize(cfg['dataset_params']['mean'], cfg['dataset_params']['std'])
        self.is_patch = is_patch
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        im_size = self.cfg['dataset_params']['im_size']
        image = cv2.imread(os.path.join(self.list_IDs_dir,self.list_IDs[index]))
        image = cv2.resize(image, (im_size,im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)
    
        mask = cv2.imread(os.path.join(self.labels_dir, self.labels[index]),0)
        mask = cv2.resize(mask, (im_size,im_size), interpolation = cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        spliced_patch_len = self.cfg['dataset_params']['s_patch_len']
        pristine_patch_len = self.cfg['dataset_params']['p_patch_len']
        patch_size = self.cfg['dataset_params']['patch_size']
        if (self.is_patch == True) and (self.cfg['global_params']['with_con'] == True):
            gp = Get_Patch()
            ps = torch.empty((spliced_patch_len,mask.shape[0],mask.shape[1]))
            for i in range(spliced_patch_len):
                m = gp.get_spliced_patch(mask, patch_size, 5)
                ps[i] = m
            pp = torch.empty((pristine_patch_len,mask.shape[0],mask.shape[1]))
            for i in range(pristine_patch_len):
                m = gp.get_pristine_patch(mask, patch_size, 5)
                pp[i] = m
            return image, mask, ps, pp

        return image, mask


def train_generator(cfg):
    image_IDs = open('dataloader/train_images.txt', 'r').read().split('\n')
    mask_IDs = open('dataloader/train_masks.txt', 'r').read().split('\n')

    batch_size = cfg['dataset_params']['batch_size']
    params = {'batch_size': batch_size,
                'shuffle': True,
                'pin_memory':True,
                'num_workers': 4}
    training_set = Dataset(image_IDs, mask_IDs, cfg, is_patch=True)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator

def val_generator(cfg):
    val_image_IDs = open('dataloader/test_images.txt', 'r').read().split('\n')
    val_mask_IDs = open('dataloader/test_masks.txt', 'r').read().split('\n')

    batch_size = cfg['dataset_params']['batch_size']
    params = {'batch_size': batch_size,
                  'shuffle': False,
                  'pin_memory':True,
                  'num_workers': 4}
    val_set = Dataset(val_image_IDs, val_mask_IDs, cfg, is_patch = False)
    validation_generator = data.DataLoader(val_set, **params)
    return validation_generator

