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
            
        self.normalize = transforms.Normalize(cfg['dataset_params']['mean'], cfg['dataset_params']['std'])
        self.is_patch = is_patch
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        im_size = self.cfg['dataset_params']['im_size']
        image = cv2.imread(self.list_IDs[index],1)
        image = cv2.resize(image, (im_size,im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)
    
        mask = cv2.imread(self.labels[index],0)
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

def get_file_names(cfg):
    train_v1 = cfg['dataset_params']['images_dir_v1']
    mask_v1 = cfg['dataset_params']['mask_dir_v1']
    train_v2 = cfg['dataset_params']['images_dir_v2']
    mask_v2 = cfg['dataset_params']['mask_dir_v2']

    train_IDs = {}
    mask_IDs = {}
    test_IDs = {}
    mask_test_IDs = {}

    train_v1_files = open('dataloader/casiav1_train.txt', 'r').read().split('\n')
    mask_v1_files = open('dataloader/casiav1_mask.txt', 'r').read().split('\n')
    train_v2_files = open('dataloader/casiav2_train.txt', 'r').read().split('\n')
    mask_v2_files = open('dataloader/casiav2_mask.txt', 'r').read().split('\n')


    rand1 = np.random.choice(len(train_v1_files), 50, replace=False).tolist()
    count = 0
    test_count = 0
    for i in range(len(train_v1_files)):
        if i in rand1:
            test_IDs[test_count] = os.path.join(train_v1, train_v1_files[i])
            mask_test_IDs[test_count] = os.path.join(mask_v1, mask_v1_files[i])
            test_count += 1
        else:
            train_IDs[count] = os.path.join(train_v1, train_v1_files[i])
            mask_IDs[count] = os.path.join(mask_v1, mask_v1_files[i])
            count += 1

    rand2 = np.random.choice(len(train_v2_files), 500, replace= False).tolist()
    for i in range(len(train_v2_files)):
        if i in rand2:
            test_IDs[test_count] = os.path.join(train_v2, train_v2_files[i])
            mask_test_IDs[test_count] = os.path.join(mask_v2, mask_v2_files[i])
            test_count += 1
        else:
            train_IDs[count] = os.path.join(train_v2, train_v2_files[i])
            mask_IDs[count] = os.path.join(mask_v2, mask_v2_files[i])
            count += 1
    
    
    return train_IDs, mask_IDs, test_IDs, mask_test_IDs

class generator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.train_IDs, self.mask_IDs, self.test_IDs, self.mask_test_IDs = get_file_names(cfg)

    def get_train_generator(self):
        
        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': True,
                    'pin_memory':True,
                    'num_workers': 4}
        training_set = Dataset(self.train_IDs, self.mask_IDs, self.cfg, is_patch=True)
        training_generator = data.DataLoader(training_set, **params)

        return training_generator

    def get_val_generator(self):

        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size,
                    'shuffle': False,
                    'pin_memory':True,
                    'num_workers': 4}
        val_set = Dataset(self.test_IDs, self.mask_test_IDs, self.cfg, is_patch = False)
        validation_generator = data.DataLoader(val_set, **params)
        return validation_generator

