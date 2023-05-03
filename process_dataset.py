import os
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import numpy as np
from pdb import set_trace as stx

# Training dataset
src_inp_dir = 'Set1_input_images'
src_tar_dir = 'Set1_ground_truth_images'

dst_inp_dir = '../mixedill/train/input'
dst_tar_dir = '../mixedill/train/target'

# Validation dataset

# src_inp_dir = 'Cube_input_images'
# src_tar_dir = 'Cube_input_images'

# dst_inp_dir = '../cube/input'
# dst_tar_dir = '../cube/target'

os.makedirs(dst_inp_dir, exist_ok=True)
os.makedirs(dst_tar_dir, exist_ok=True)

extensions = ['_T_CS.png', '_F_CS.png', '_D_CS.png', '_C_CS.png', '_S_CS.png']

inp_files = []
for extension in extensions:
    filenames = glob(os.path.join(src_inp_dir, '*'+extension))
    filenames = [os.path.split(i)[-1][:-9] for i in filenames]
    inp_files.append(filenames)

files = inp_files[0]
for i in range(1,len(extensions)):
    files = np.intersect1d(files, inp_files[i])

for file_ in tqdm(files):
    for extension in extensions:
        src_file = os.path.join(src_inp_dir, file_+extension)
        dst_file = os.path.join(dst_inp_dir, file_+extension)
        os.system(f'cp {src_file} {dst_file}')

for file_ in tqdm(files):
    src_file = os.path.join(src_tar_dir, file_+'_G_AS.png')
    dst_file = os.path.join(dst_tar_dir, file_+'_G_AS.png')
    os.system(f'cp {src_file} {dst_file}')
