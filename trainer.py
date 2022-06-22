import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
#import segmentation_models_pytorch as smp
from tqdm import tqdm
import utils.utils as utils
import torch.optim as optim
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
import itertools
from sklearn import metrics
import datetime
import timm
import yaml
import random
from model.model import ConSegNet
from torch.optim.lr_scheduler import StepLR
import torchmetrics

set_random_seed(1221)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

with torch.no_grad():
    test_model = timm.create_model(cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
    del test_model


if cfg['dataset_params']['dataset_name'] == 'casia':
    from dataloader.loader import generator
elif cfg['dataset_params']['dataset_name'] == 'imd_2020':
    from dataloader.loader_imd import generator

gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()

model = ConSegNet(cfg, in_planes).to(device)


if cfg['model_params']['lr_reduction_pretrained'] == True:
    pretrained_params = []
    other_params = []
    for key in list(model.named_parameters()):
        if len(key[0].split('encoder.')) == 2:
            pretrained_params.append(key[1])
        else:
            other_params.append(key[1])

    if cfg['model_params']['optimizer'] == 'sgd':
        optimizer = optim.SGD([{'params': pretrained_params, 'lr' : cfg['model_params']['lr']/5}, 
                {'params': other_params}], 
                lr = cfg['model_params']['lr'], weight_decay = 1e-4, momentum = 0.9)
    else:
        optimizer = optim.Adam([{'params': pretrained_params, 'lr' : cfg['model_params']['lr']/5}, 
                {'params': other_params}], 
                lr = cfg['model_params']['lr'])
else:
    if cfg['model_params']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                lr = cfg['model_params']['lr'], weight_decay = 1e-4, momentum = 0.9)
    else:
        optimizer = optim.Adam(model.parameters(), 
                lr = cfg['model_params']['lr'])

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)




casia_imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']).to(device)
criterion = nn.CrossEntropyLoss(weight = casia_imbalance_weight)



max_val_auc = 0
max_val_iou = [0.0, 0.0]
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()


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
        if cfg['global_params']['with_con'] == True:
            ps = sample[2].to(device)
            pp = sample[3].to(device)

        pred, feat = model(img)
        
        pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        feat = F.interpolate(feat, img.shape[2:], mode= 'bilinear', align_corners = True)

        if cfg['global_params']['with_con'] == True:

            cfeature_pos = utils.calfeaturevectors(feat, ps.detach())
            cfeature_pos = F.normalize(cfeature_pos, dim = -1)
            cfeature_neg = utils.calfeaturevectors(feat, pp.detach())
            cfeature_neg = F.normalize(cfeature_neg, dim = -1)

            contrast_temperature = cfg['dataset_params']['contrast_temperature']
            c_loss = utils.patch_contrast(cfeature_pos, cfeature_neg, device, contrast_temperature)
            c_loss = c_loss.mean(dim=-1)
            c_loss = c_loss.mean()
            loss = criterion(pred, tar.long().detach()) 
            total_loss = loss + cfg['model_params']['con_alpha']*c_loss
        else:
            loss = criterion(pred, tar.long().detach()) 
            total_loss = loss 

        
        total_loss.backward()

        optimizer.step()

        if cfg['global_params']['with_con'] == True:
            train_closs.update(c_loss.detach().cpu().item())
        train_sloss.update(loss.detach().cpu().item())
        
        
        intr, uni = batch_intersection_union(pred, tar, num_class = cfg['model_params']['num_class'])
        
        train_inter.update(intr)
        train_union.update(uni)
        
        
    # if epoch>20:
    #     scheduler.step()    
        
    train_softmax = train_sloss.avg

    if cfg['global_params']['with_con'] == True:
        train_contrast = train_closs.avg
    
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
        auc = []
        for img, tar in tqdm(validation_generator):
            img, tar = img.to(device), tar.to(device)
            pred, _ = model(img)
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
            intr, uni = batch_intersection_union(pred, tar, num_class = cfg['model_params']['num_class'])
            val_inter.update(intr)
            val_union.update(uni)
            # _, pred = torch.max(pred, 1)
            y_score = F.softmax(pred, dim=1)[:,1,:,:]
            # val_pred.extend(y_score.contiguous().view(-1).cpu().numpy().tolist())
            # val_tar.extend(tar.contiguous().view(-1).long().cpu().numpy().tolist())
            for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()) :
                this = metrics.roc_auc_score(yy_true.ravel(), yy_pred.ravel(), average = None)
                that = metrics.roc_auc_score(yy_true.ravel(), (1-yy_pred).ravel(), average = None)
                auc.append(this)
            

        
        # val_pred = list(itertools.chain(*val_pred))
        # val_tar = list(itertools.chain(*val_tar))

        # val_auc = metrics.roc_auc_score(val_tar, val_pred, average = None)
        val_auc = np.mean(auc)
        # fpr, tpr, thresholds = metrics.roc_curve(np.array(val_tar), np.array(val_pred), pos_label=1)
        # val_auc = metrics.auc(fpr, tpr)

        val_pred = []
        val_tar = []

        if val_auc > max_val_auc:
            max_val_auc = val_auc

        val_IoU = val_inter.sum/(val_union.sum + 1e-10)
        val_IoU = val_IoU.tolist()
        val_mIoU = np.mean(val_IoU)
        val_mIoU = val_mIoU.tolist()

        if val_IoU[1] > max_val_iou[1]:
            max_val_iou = val_IoU

        if cfg['global_params']['with_con'] == True:
            logs = {'epoch': epoch, 'Softmax Loss':train_softmax, 'Contrastive Loss':train_contrast,
            'Train IoU':train_IoU, 'Validation IoU': val_IoU, 'Validation AUC': val_auc, 
            'Max Validaton_AUC': max_val_auc, "Max IoU Tampered": max_val_iou}
        
        else:
            logs = {'epoch': epoch, 'Softmax Loss':train_softmax,
            'Train IoU':train_IoU, 'Validation IoU': val_IoU, 'Validation AUC': val_auc, 
            'Max Validaton_AUC': max_val_auc, "Max IoU Tampered": max_val_iou}

        tb.add_scalar("auc", val_auc, epoch+1)
        write_logger(filename_log, cfg, **logs)


        
        
           
