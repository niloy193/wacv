import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import cv2

def preprocess_dresden(dresden_path):
    # dresden_path = '/home/agency/xai/forgery/dresden_spliced'
    files = os.listdir(dresden_path)
    for i in tqdm(range(len(files))):
        if files[i].split('_')[1].split('.')[0] == 'mask':
            img = cv2.imread(os.path.join(dresden_path,files[i]),0)
            img = torch.from_numpy(img).cuda()
            img[(img>0).int() == (img<255).int()] = 255
            img = img/255
            img = torch.add(torch.negative(img),1)
            img = img*255
            img = img.cpu().numpy()
            cv2.imwrite(os.path.join(dresden_path,files[i]), img)

def preprocess_nist(nist_path):
    # nist_path = '/home/agency/xai/forgery/Bernard/Computer_Vision/spliced_NIST/masks'
    files = os.listdir(nist_path)
    for i in tqdm(range(len(files))):
        img = cv2.imread(os.path.join(nist_path,files[i]),0)
        img = torch.from_numpy(img).cuda()
        img[(img>0).int() == (img<255).int()] = 255
        img = img/255
        img = torch.add(torch.negative(img),1)
        img = img*255
        img = img.cpu().numpy()
        cv2.imwrite(os.path.join(nist_path,files[i]), img)