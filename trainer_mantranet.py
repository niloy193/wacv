import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import utils.utils as utils
import torch.optim as optim
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
import itertools
from sklearn import metrics
import datetime
import timm
import yaml

set_random_seed(1441)

with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

from model.hrnet_con import hrnet48
from model.mantranet import MantraNet

model = MantraNet(device=device).to(device)
from dataloader.loader import generator

gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()




if cfg['model_params']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr = cfg['model_params']['lr'], weight_decay = 1e-4, momentum = 0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr = cfg['model_params']['lr'] )

casia_imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']).to(device)
criterion = nn.CrossEntropyLoss(weight = casia_imbalance_weight)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

max_val_auc = 0

for epoch in range(cfg['model_params']['epoch']):
    train_loss = AverageMeter()
    train_inter = AverageMeter()
    train_union = AverageMeter()
    train_sloss = AverageMeter()
    train_closs = AverageMeter()

    for sample in tqdm(training_generator):
        model.train()
        optimizer.zero_grad()

        img = sample[0].to(device)
        tar = sample[1].to(device)
        # if cfg['global_params']['with_con'] == True:
        #     ps = sample[2].to(device)
        #     pp = sample[3].to(device)

        pred = model(img)
        # pred = pred.squeeze(dim=1)
        
        # pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)

        # feat = F.interpolate(feat, img.shape[2:], mode= 'bilinear', align_corners = True)

        # if cfg['global_params']['with_con'] == True:

        #     cfeature_pos = utils.calfeaturevectors(feat, ps.detach())
        #     cfeature_pos = F.normalize(cfeature_pos, dim = -1)
        #     cfeature_neg = utils.calfeaturevectors(feat, pp.detach())
        #     cfeature_neg = F.normalize(cfeature_neg, dim = -1)

        #     c_loss = utils.patch_contrast(cfeature_pos, cfeature_neg, device)
        #     c_loss = c_loss.mean(dim=-1)
        #     c_loss = c_loss.mean()
        #     loss = criterion(pred, tar.long().detach()) 
        #     total_loss = loss + c_loss
        # else:
        loss = criterion(pred, tar.long().detach()) 
        total_loss = loss 

        
        total_loss.backward()

        optimizer.step()

        # if cfg['global_params']['with_con'] == True:
        #     train_closs.update(c_loss.detach().cpu().item())
        train_sloss.update(loss.detach().cpu().item())
        
        
        intr, uni = batch_intersection_union(pred, tar, num_class = cfg['model_params']['num_class'])
        
        train_inter.update(intr)
        train_union.update(uni)
        
    
    train_softmax = train_sloss.avg

    # if cfg['global_params']['with_con'] == True:
    #     train_contrast = train_closs.avg
    
    train_IoU = train_inter.sum/(train_union.sum + 1e-10)
    train_IoU = train_IoU.tolist()
    train_mIoU = np.mean(train_IoU)
    train_mIoU = train_mIoU.tolist()
    
    with torch.no_grad():
        model.eval()
        val_inter = AverageMeter()
        val_union = AverageMeter()
        val_pred = []
        val_tar = []
        for img, tar in tqdm(validation_generator):
            img, tar = img.to(device), tar.to(device)
            pred = model(img)
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
            intr, uni = batch_intersection_union(pred, tar, num_class = cfg['model_params']['num_class'])
            val_inter.update(intr)
            val_union.update(uni)
            _, pred = torch.max(pred, 1)
            val_pred.append(pred.view(-1).cpu().numpy().tolist())
            val_tar.append(tar.view(-1).long().cpu().numpy().tolist())
            
        
        val_pred = list(itertools.chain(*val_pred))
        val_tar = list(itertools.chain(*val_tar))
        
        fpr, tpr, thresholds = metrics.roc_curve(np.array(val_tar), np.array(val_pred), pos_label=1)
        val_auc = metrics.auc(fpr, tpr)

        val_pred = []
        val_tar = []

        if val_auc > max_val_auc:
            max_val_auc = val_auc

        val_IoU = val_inter.sum/(val_union.sum + 1e-10)
        val_IoU = val_IoU.tolist()
        val_mIoU = np.mean(val_IoU)
        val_mIoU = val_mIoU.tolist()

        # if cfg['global_params']['with_con'] == True:
        #     logs = {'epoch': epoch, 'Softmax Loss':train_softmax, 'Contrastive Loss':train_contrast,
        #     'Train IoU':train_IoU, 'Validation IoU': val_IoU, 'Validation AUC': val_auc, 
        #     'Max Validaton_AUC': max_val_auc}
        
        logs = {'epoch': epoch, 'Softmax Loss':train_softmax,
            'Train IoU':train_IoU, 'Validation IoU': val_IoU, 'Validation AUC': val_auc, 
            'Max Validaton_AUC': max_val_auc}

        write_logger(filename_log, cfg, **logs)
        
        
           