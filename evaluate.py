import os
import numpy as np
import cv2

from natsort import natsorted
from glob import glob
from tqdm import tqdm

from eval.evaluate_cc import evaluate_cc

from pdb import set_trace as stx

tar_dir = './mixedill_test_set_PNG/target'
prd_dir = './results/Gridnet1'
# prd_dir = './results/Gridnet2'

deltaE00s, MSEs, MAEs, deltaE76s = [], [], [] ,[]

tar_files = natsorted(glob(os.path.join(tar_dir, '*_G_AS.png')))
prd_files = natsorted(glob(os.path.join(prd_dir, '*.png')))

for tar_file, prd_file in zip(tar_files, prd_files):
    tar_img = cv2.imread(tar_file, cv2.IMREAD_COLOR)
    prd_img = cv2.imread(prd_file, cv2.IMREAD_COLOR)

    deltaE00, MSE, MAE, deltaE76 = evaluate_cc(prd_img, tar_img, 0, opt=4)
    # print(tar_file, prd_file)

    deltaE00s.append(deltaE00)
    MSEs.append(MSE)
    MAEs.append(MAE)
    deltaE76s.append(deltaE76)

deltaE00s = sum(deltaE00s) / len(deltaE00s)
MSEs = sum(MSEs) / len(MSEs)
MAEs = sum(MAEs) / len(MAEs)
deltaE76s = sum(deltaE76s) / len(deltaE76s)

print('MSE= %0.2f, MAE= %0.2f, DeltaE 2000: %0.2f, DeltaE 76= %0.2f\n' % (MSEs, MAEs, deltaE00s, deltaE76s))
