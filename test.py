"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from WBnet import WBnet
from skimage import img_as_ubyte
from pdb import set_trace as stx
import weight_refinement
from PIL import Image

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./mixedill_test_set_PNG/input/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gridnet1', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/WB/models/Gridnet1/model_epoch_200.pth', type=str, help='Path to weights')
parser.add_argument('--wb_settings', default=['D', 'S', 'T'], nargs='+', help='Test Dataset')
# parser.add_argument('--result_dir', default='./results/Gridnet2', type=str, help='Directory for results')
# parser.add_argument('--weights', default='./checkpoints/WB/models/Gridnet2/model_epoch_200.pth', type=str, help='Path to weights')
# parser.add_argument('--wb_settings', default=['D', 'S', 'T', 'F', 'C'], nargs='+', help='Test Dataset')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restoration = WBnet(device=device, inchnls=len(args.wb_settings)*3)


utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

rgb_dir_test = args.input_dir

test_dataset = get_test_data(rgb_dir_test, img_options={'wb_settings':args.wb_settings})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir  = args.result_dir
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        full_size = data_test[1].cuda()
        filenames = data_test[2]

        d_img = full_size[:,0:3,:,:]

        _, weights = model_restoration(input_)

        img_1 = F.interpolate(input_, size=(int(0.5 * input_.shape[2]), int(0.5 * input_.shape[3])), mode='bilinear', align_corners=True)
        _, weights_1 = model_restoration(img_1)
        weights_1 = F.interpolate(weights_1, size=(input_.shape[2], input_.shape[3]), mode='bilinear', align_corners=True)

        img_2 = F.interpolate(input_, size=(int(0.25 * input_.shape[2]), int(0.25 * input_.shape[3])), mode='bilinear', align_corners=True)
        _, weights_2 = model_restoration(img_2)
        weights_2 = F.interpolate(weights_2, size=(input_.shape[2], input_.shape[3]), mode='bilinear', align_corners=True)

        weights = (weights + weights_1 + weights_2) / 3

        weights = F.interpolate(weights, size=(d_img.shape[2], d_img.shape[3]), mode='bilinear', align_corners=True)

        # imgs = [d_img, s_img, t_img]

        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                ref = d_img[j, :, :, :]
                curr_weight = weights[j, i, :, :]
                refined_weight = weight_refinement.process_image(ref, curr_weight, tensor=True)
                weights[j, i, :, :] = refined_weight
                weights = weights / torch.sum(weights, dim=1)


        for i in range(weights.shape[1]):
            if i == 0:
                out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * full_size[:,i*3:i*3+3,:,:]
            else:
                out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * full_size[:,i*3:i*3+3,:,:]


        for i, fname in enumerate(filenames):
            img = out_img[i].permute(1, 2, 0).cpu().numpy()
            result = Image.fromarray((img * 255).astype(np.uint8))
            name = os.path.join(result_dir, fname)
            result.save(name)
