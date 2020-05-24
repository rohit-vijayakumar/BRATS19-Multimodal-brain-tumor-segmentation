# importing libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import random as r
import math
import glob
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


K.set_image_dim_ordering("th")

img_size = 240      #original img size is 240*240
smooth = 0.005 
num_of_aug = 2
num_epoch = 30
pul_seq = 'Flair'
sharp = False       # sharpen filter
LR = 1e-4

num_of_patch = 4 # must be a square number
label_num = 5   # 1 = necrosis+NET, 2 = tumor core, 3 = original, 4 = ET, 5 = complete tumor


def n4itk(img):         #must input with sitk img object
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    return corrected_img    

# data transforms for flair and t2
def create_data_ft(src, maskfl, maskt2, label=False, mod=1, resize=(155,img_size,img_size)):
    if mod==1:
        files = glob.glob(src + maskfl, recursive=True)
        r.seed(9)
        r.shuffle(files) # shuffle patients
        imgs = []
        print('Processing---Flair')
        for file in files:    # get flair
            img = io.imread(file, plugin='simpleitk')
            img = (img-img.mean()) / img.std()      # normalization => zero mean 
            img = img.astype('float32')
            for slice in range(0,155):     # choose the slice range
                img_t = img[slice,:,:]
                img_t =img_t.reshape((1,)+img_t.shape)
                img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
                for n in range(img_t.shape[0]):
                    imgs.append(img_t[n,:,:,:])
        return np.array(imgs)
        
        
    if mod==2:
        files = glob.glob(src + maskt2, recursive=True)
        r.seed(9)
        r.shuffle(files) # shuffle patients
        imgs = []
        print('Processing---T2')
        for file in files:    #  get t2
            img = io.imread(file, plugin='simpleitk')
            img = (img-img.mean()) / img.std()      #normalization => zero mean 
            img = img.astype('float32')
            for slice in range(0,155):     # choose the slice range
                img_t = img[slice,:,:]
                img_t =img_t.reshape((1,)+img_t.shape)
                img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
                for n in range(img_t.shape[0]):
                    imgs.append(img_t[n,:,:,:])
        return np.array(imgs)

# data transforms for t1
def create_data_t1(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---T1')
    for file in files:
        img = io.imread(file, plugin='simpleitk')

        if label:
            if label_num == 5:
                img[img != 0] = 1       # Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)

# data transforms for flair and t1c
def create_data_t1c(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---T1C')
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            if label_num == 5:
                img[img != 0] = 1       # Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     #choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)


# data transforms for full tumor mask
def create_data_mask_full(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---mask_full')
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            if label_num == 5:
                img[img != 0] = 1       #nRegion 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)

# data transforms for tumor core mask
def create_data_mask_core(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---mask_core')
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            if label_num == 5:
                img[img != 0] = 1       # Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)
    
# data transforms for enhancing tumor
def create_data_mask_et(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---mask_et')
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            if label_num == 5:
                img[img != 0] = 1       # Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)
    
# data transforms complete tumor mask
def create_data_mask_all(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---mask_all')
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            if label_num == 5:
                img[img != 0] = 1       # Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       # only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       # turn edema to 0
                img[img != 0] = 1       # only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       # only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   # become rank 4
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)


# Loading dataset

# flair
flair = create_data_ft('E:/code/custom final testing/', '**/*_flair.nii.gz', '**/*_t2.nii.gz', label=False, mod=1, resize=(155,img_size,img_size))

# t2
t2 = create_data_ft('E:/code/custom final testing/', '**/*_flair.nii.gz', '**/*_t2.nii.gz', label=False, mod=2, resize=(155,img_size,img_size))

# t1
t1 = create_data_t1('E:/code/custom final testing/', '**/*_t1.nii.gz', label=False, resize=(155,img_size,img_size))

# t1c
t1c = create_data_t1c('E:/code/custom final testing/', '**/*_t1ce.nii.gz', label=False, resize=(155,img_size,img_size))

# full tumor mask
label_num = 5
mask_full = create_data_mask_full('E:/code/custom final testing/', '**/*_seg.nii.gz', label=True, resize=(155,img_size,img_size))

# tumor core mask
label_num = 2
mask_core = create_data_mask_core('E:/code/custom final testing/', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))

# enhancing tumor mask
label_num = 4
mask_ET = create_data_mask_et('E:/code/custom final testing/', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))

# complete tumor mask
label_num = 3
mask_all = create_data_mask_all('E:/code/custom final testing/', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))