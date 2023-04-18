import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import random
from wb_utils import imresize, aug, extract_patch, get_mapping_func, apply_mapping_func, outOfGamutClipping
from pdb import set_trace as stx

def is_image_file(filename, ext_finder):
    return any(filename.endswith(ext_finder + '.' + extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def center_crop(img, ps):
    h,w,c = img.shape
    return img[(h-ps)//2:(h+ps)//2,(w-ps)//2:(w+ps)//2,:]

class DataLoaderTrain3Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain3Im, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x, 'G_AS')]

        # self.inpD_filenames = self.inpD_filenames[:40]
        # self.inpS_filenames = self.inpS_filenames[:40]
        # self.inpT_filenames = self.inpT_filenames[:40]
        # self.tar_filenames = self.tar_filenames[:40]

        # self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = img_options['patch_size']
        self.t_size = 320

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]
        tar_path  = self.tar_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)
        tar_img  = load_img(tar_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max
        tar_img  = np.float32(tar_img)/im_max


        inpD_img = imresize(inpD_img, output_shape=(self.t_size, self.t_size))
        inpS_img = imresize(inpS_img, output_shape=(self.t_size, self.t_size))
        inpT_img = imresize(inpT_img, output_shape=(self.t_size, self.t_size))
        tar_img  = imresize(tar_img, output_shape=(self.t_size, self.t_size))

        inpD_img, inpS_img, inpT_img, tar_img = aug(inpD_img, inpS_img, inpT_img, tar_img)

        inpD_img, inpS_img, inpT_img, tar_img = extract_patch(inpD_img, inpS_img, inpT_img, tar_img, patch_size=self.ps, patch_number=4)

        inpD_img = torch.from_numpy(inpD_img).permute(0,3,1,2)
        inpS_img = torch.from_numpy(inpS_img).permute(0,3,1,2)
        inpT_img = torch.from_numpy(inpT_img).permute(0,3,1,2)
        tar_img  = torch.from_numpy(tar_img).permute(0,3,1,2)

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img], 1)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderVal3Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal3Im, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x, 'G_AS')]

        # self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = img_options['patch_size']
        self.t_size = 320

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]
        tar_path  = self.tar_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)
        tar_img  = load_img(tar_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max
        tar_img  = np.float32(tar_img)/im_max

        inpD_img = center_crop(inpD_img, self.ps)
        inpS_img = center_crop(inpS_img, self.ps)
        inpT_img = center_crop(inpT_img, self.ps)
        tar_img  = center_crop(tar_img, self.ps)

        inpD_img = torch.from_numpy(inpD_img).permute(2,0,1)
        inpS_img = torch.from_numpy(inpS_img).permute(2,0,1)
        inpT_img = torch.from_numpy(inpT_img).permute(2,0,1)
        tar_img  = torch.from_numpy(tar_img).permute(2,0,1)

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img], 0)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest3Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTest3Im, self).__init__()

        inp_files = sorted(os.listdir(rgb_dir))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]

        self.img_options = img_options
        self.sizex       = len(self.inpD_filenames)  # get the size of target

        self.t_size = 384

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max

        full_size_d = inpD_img.copy()

        inpD_img = imresize(inpD_img, output_shape=(self.t_size, self.t_size))
        inpS_img = imresize(inpS_img, output_shape=(self.t_size, self.t_size))
        inpT_img = imresize(inpT_img, output_shape=(self.t_size, self.t_size))

        s_mapping = get_mapping_func(inpD_img, inpS_img)
        full_size_s = apply_mapping_func(full_size_d, s_mapping)
        full_size_s = outOfGamutClipping(full_size_s)

        t_mapping = get_mapping_func(inpD_img, inpT_img)
        full_size_t = apply_mapping_func(full_size_d, t_mapping)
        full_size_t = outOfGamutClipping(full_size_t)

        inpD_img = torch.from_numpy(inpD_img).permute(2,0,1).float()
        inpS_img = torch.from_numpy(inpS_img).permute(2,0,1).float()
        inpT_img = torch.from_numpy(inpT_img).permute(2,0,1).float()

        full_size_d = torch.from_numpy(full_size_d).permute(2,0,1).float()
        full_size_s = torch.from_numpy(full_size_s).permute(2,0,1).float()
        full_size_t = torch.from_numpy(full_size_t).permute(2,0,1).float()

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img], 0)
        full_size_img = torch.cat([full_size_d, full_size_s, full_size_t], 0)

        filename = os.path.splitext(os.path.split(inpD_path)[-1])[0][:-4] + '_WB.png'

        return inp_img, full_size_img, filename

class DataLoaderTrain5Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain5Im, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]
        self.inpF_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[3]+'_CS')]
        self.inpC_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[4]+'_CS')]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x, 'G_AS')]

        # self.inpD_filenames = self.inpD_filenames[:40]
        # self.inpS_filenames = self.inpS_filenames[:40]
        # self.inpT_filenames = self.inpT_filenames[:40]
        # self.tar_filenames = self.tar_filenames[:40]

        # self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = img_options['patch_size']
        self.t_size = 320

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]
        inpF_path = self.inpF_filenames[index_]
        inpC_path = self.inpC_filenames[index_]
        tar_path  = self.tar_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)
        inpF_img = load_img(inpF_path)
        inpC_img = load_img(inpC_path)
        tar_img  = load_img(tar_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max
        inpF_img = np.float32(inpF_img)/im_max
        inpC_img = np.float32(inpC_img)/im_max
        tar_img  = np.float32(tar_img)/im_max


        inpD_img = imresize(inpD_img, output_shape=(self.t_size, self.t_size))
        inpS_img = imresize(inpS_img, output_shape=(self.t_size, self.t_size))
        inpT_img = imresize(inpT_img, output_shape=(self.t_size, self.t_size))
        inpF_img = imresize(inpF_img, output_shape=(self.t_size, self.t_size))
        inpC_img = imresize(inpC_img, output_shape=(self.t_size, self.t_size))
        tar_img  = imresize(tar_img, output_shape=(self.t_size, self.t_size))

        inpD_img, inpS_img, inpT_img, inpF_img, inpC_img, tar_img = aug(inpD_img, inpS_img, inpT_img, inpF_img, inpC_img, tar_img)

        inpD_img, inpS_img, inpT_img, inpF_img, inpC_img, tar_img = extract_patch(inpD_img, inpS_img, inpT_img, inpF_img, inpC_img, tar_img, patch_size=self.ps, patch_number=4)

        inpD_img = torch.from_numpy(inpD_img).permute(0,3,1,2)
        inpS_img = torch.from_numpy(inpS_img).permute(0,3,1,2)
        inpT_img = torch.from_numpy(inpT_img).permute(0,3,1,2)
        inpF_img = torch.from_numpy(inpF_img).permute(0,3,1,2)
        inpC_img = torch.from_numpy(inpC_img).permute(0,3,1,2)
        tar_img  = torch.from_numpy(tar_img).permute(0,3,1,2)

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img, inpF_img, inpC_img], 1)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderVal5Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal5Im, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]
        self.inpF_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[3]+'_CS')]
        self.inpC_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x, wb_settings[4]+'_CS')]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x, 'G_AS')]

        # self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = img_options['patch_size']
        self.t_size = 320

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]
        inpF_path = self.inpF_filenames[index_]
        inpC_path = self.inpC_filenames[index_]
        tar_path  = self.tar_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)
        inpF_img = load_img(inpF_path)
        inpC_img = load_img(inpC_path)
        tar_img  = load_img(tar_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max
        inpF_img = np.float32(inpF_img)/im_max
        inpC_img = np.float32(inpC_img)/im_max
        tar_img  = np.float32(tar_img)/im_max

        inpD_img = center_crop(inpD_img, self.ps)
        inpS_img = center_crop(inpS_img, self.ps)
        inpT_img = center_crop(inpT_img, self.ps)
        inpF_img = center_crop(inpF_img, self.ps)
        inpC_img = center_crop(inpC_img, self.ps)
        tar_img  = center_crop(tar_img, self.ps)

        inpD_img = torch.from_numpy(inpD_img).permute(2,0,1)
        inpS_img = torch.from_numpy(inpS_img).permute(2,0,1)
        inpT_img = torch.from_numpy(inpT_img).permute(2,0,1)
        inpF_img = torch.from_numpy(inpF_img).permute(2,0,1)
        inpC_img = torch.from_numpy(inpC_img).permute(2,0,1)
        tar_img  = torch.from_numpy(tar_img).permute(2,0,1)

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img, inpF_img, inpC_img], 0)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest5Im(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTest5Im, self).__init__()

        inp_files = sorted(os.listdir(rgb_dir))
        wb_settings = img_options['wb_settings']

        self.inpD_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[0]+'_CS')]
        self.inpS_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[1]+'_CS')]
        self.inpT_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[2]+'_CS')]
        self.inpF_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[3]+'_CS')]
        self.inpC_filenames = [os.path.join(rgb_dir, x)  for x in inp_files if is_image_file(x, wb_settings[4]+'_CS')]

        self.img_options = img_options
        self.sizex       = len(self.inpD_filenames)  # get the size of target

        self.t_size = 384

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inpD_path = self.inpD_filenames[index_]
        inpS_path = self.inpS_filenames[index_]
        inpT_path = self.inpT_filenames[index_]
        inpF_path = self.inpF_filenames[index_]
        inpC_path = self.inpC_filenames[index_]

        inpD_img = load_img(inpD_path)
        inpS_img = load_img(inpS_path)
        inpT_img = load_img(inpT_path)
        inpF_img = load_img(inpF_path)
        inpC_img = load_img(inpC_path)

        im_max = 255. if inpD_img[0].dtype == 'uint8' else 65535. 

        inpD_img = np.float32(inpD_img)/im_max
        inpS_img = np.float32(inpS_img)/im_max
        inpT_img = np.float32(inpT_img)/im_max
        inpF_img = np.float32(inpF_img)/im_max
        inpC_img = np.float32(inpC_img)/im_max

        full_size_d = inpD_img.copy()

        inpD_img = imresize(inpD_img, output_shape=(self.t_size, self.t_size))
        inpS_img = imresize(inpS_img, output_shape=(self.t_size, self.t_size))
        inpT_img = imresize(inpT_img, output_shape=(self.t_size, self.t_size))
        inpF_img = imresize(inpF_img, output_shape=(self.t_size, self.t_size))
        inpC_img = imresize(inpC_img, output_shape=(self.t_size, self.t_size))

        s_mapping = get_mapping_func(inpD_img, inpS_img)
        full_size_s = apply_mapping_func(full_size_d, s_mapping)
        full_size_s = outOfGamutClipping(full_size_s)

        t_mapping = get_mapping_func(inpD_img, inpT_img)
        full_size_t = apply_mapping_func(full_size_d, t_mapping)
        full_size_t = outOfGamutClipping(full_size_t)

        f_mapping = get_mapping_func(inpD_img, inpF_img)
        full_size_f = apply_mapping_func(full_size_d, f_mapping)
        full_size_f = outOfGamutClipping(full_size_f)

        c_mapping = get_mapping_func(inpD_img, inpC_img)
        full_size_c = apply_mapping_func(full_size_d, c_mapping)
        full_size_c = outOfGamutClipping(full_size_c)

        inpD_img = torch.from_numpy(inpD_img).permute(2,0,1).float()
        inpS_img = torch.from_numpy(inpS_img).permute(2,0,1).float()
        inpT_img = torch.from_numpy(inpT_img).permute(2,0,1).float()
        inpF_img = torch.from_numpy(inpF_img).permute(2,0,1).float()
        inpC_img = torch.from_numpy(inpC_img).permute(2,0,1).float()

        full_size_d = torch.from_numpy(full_size_d).permute(2,0,1).float()
        full_size_s = torch.from_numpy(full_size_s).permute(2,0,1).float()
        full_size_t = torch.from_numpy(full_size_t).permute(2,0,1).float()
        full_size_f = torch.from_numpy(full_size_f).permute(2,0,1).float()
        full_size_c = torch.from_numpy(full_size_c).permute(2,0,1).float()

        inp_img = torch.cat([inpD_img, inpS_img, inpT_img, inpF_img, inpC_img], 0)
        full_size_img = torch.cat([full_size_d, full_size_s, full_size_t, full_size_f, full_size_c], 0)

        filename = os.path.splitext(os.path.split(inpD_path)[-1])[0][:-4] + '_WB.png'

        return inp_img, full_size_img, filename

