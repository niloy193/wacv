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
        image = cv2.imread(self.list_IDs[index])
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


def train_generator(cfg):
    dresden_dir = '/home/agency/xai/forgery/dresden_spliced'
    files = os.listdir(dresden_dir)
    train_IDs = {}
    train_masks = {}
    train_count = 0

    for i in range(len(files)):
        if files[i].split('_')[1].split('.')[0] == 'rgb':
            train_file_name = files[i]
            train_IDs[train_count] = os.path.join(dresden_dir, train_file_name)
            mask_file_name = files[i].split('_')[0] + '_mask.png'
            train_masks[train_count] = os.path.join(dresden_dir, mask_file_name)
            train_count += 1
        
    bernard_rgb_dir = '/home/agency/xai/forgery/Bernard/Computer_Vision/spliced_NIST/rgb_imgs'
    rgb_files = os.listdir(bernard_rgb_dir)
    bernard_mask_dir = '/home/agency/xai/forgery/Bernard/Computer_Vision/spliced_NIST/masks'

    for i in range(len(rgb_files)):
        train_IDs[train_count] = os.path.join(bernard_rgb_dir, rgb_files[i])
        mask_file_name = rgb_files[i].split('_')[0] + '_mask.png'
        train_masks[train_count] = os.path.join(bernard_mask_dir, mask_file_name)
        train_count += 1

    batch_size = cfg['dataset_params']['batch_size']
    params = {'batch_size': batch_size,
                'shuffle': True,
                'pin_memory':True,
                'num_workers': 4}
    training_set = Dataset(train_IDs, train_masks, cfg, is_patch=True)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator

def val_generator(cfg):
    test_dir = '/home/agency/xai/CASIA2/Tp'
    test_mask_dir =  '/home/agency/xai/CASIA2/Gt'
    files1 = os.listdir(test_dir)
    files2 = os.listdir(test_mask_dir)

    test_IDs = {}
    test_masks = {}
    test_count = 0

    for i in range(len(files1)):
        if (files1[i].endswith('jpg') or files1[i].endswith('tif')):
            mask_file_name = files1[i].split('.')[0] + '_gt.png'
            if mask_file_name in files2:
                test_IDs[test_count] = os.path.join(test_dir, files1[i])
                test_masks[test_count] = os.path.join(test_mask_dir, mask_file_name)
                test_count += 1

    batch_size = cfg['dataset_params']['batch_size']
    params = {'batch_size': batch_size,
                  'shuffle': False,
                  'pin_memory':True,
                  'num_workers': 4}
    val_set = Dataset(test_IDs, test_masks, cfg, is_patch = False)
    validation_generator = data.DataLoader(val_set, **params)
    return validation_generator

