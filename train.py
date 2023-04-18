import os
from config import Config
import argparse

parser = argparse.ArgumentParser(description = 'Choose file')
parser.add_argument('--training', default='training.yml')
args = parser.parse_args()
opt = Config(args.training)
print(f"Using {args.training}")

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from WBnet import WBnet
import losses
from tqdm import tqdm
from wb_utils import get_sobel_kernel
from pdb import set_trace as stx

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### Model ###########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_channels = len(opt.TRAINING.WB_SETTINGS) * 3
model_restoration = WBnet(device=device, inchnls=num_channels)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999)) #,eps=1e-8)

class NotScheduler:
    def __init__(self, lr):
        self.lr = lr
    def step(self):
        pass
    def get_lr(self):
        return [self.lr]

######### Scheduler ###########
scheduler = NotScheduler(new_lr)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion = nn.MSELoss()
x_kernel, y_kernel = get_sobel_kernel(device, chnls=len(opt.TRAINING.WB_SETTINGS))

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS, 'wb_settings':opt.TRAINING.WB_SETTINGS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS, 'wb_settings':opt.TRAINING.WB_SETTINGS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(train_loader, 0):
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].float().cuda()
        input_ = data[1].float().cuda()

        rec_loss = 0
        smoothness_loss = 0
        for p in range(input_.shape[1]):
            patch    = input_[:, p, :, :, :]
            gt_patch = target[:, p, :, :, :]
            result, weights = model_restoration(patch)
            rec_loss += criterion(result, gt_patch)

            smoothness_loss += 100. * (
              torch.sum(F.conv2d(weights, x_kernel, stride=1) ** 2) +
              torch.sum(F.conv2d(weights, y_kernel, stride=1) ** 2))
 
        # Compute loss at each stage
        rec_loss = rec_loss / input_.shape[1]
        smoothness_loss = smoothness_loss / input_.shape[1]
        loss = rec_loss + smoothness_loss
       
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

    #### Evaluation ####
    if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].float().cuda()
            input_ = data_val[1].float().cuda()

            with torch.no_grad():
                restored, weights = model_restoration(input_)

            for res,tar in zip(restored,target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

