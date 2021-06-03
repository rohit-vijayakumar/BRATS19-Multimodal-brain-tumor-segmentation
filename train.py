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
from data_preprocess import *
from models import *
from metrics import *

K.set_image_dim_ordering("th")

img_size = 240      #original img size is 240*240
smooth = 0.005 
num_of_aug = 2
num_epoch = 30
pul_seq = 'Flair'
sharp = False 
LR = 1e-4

label_num = 5   # 1 = necrosis+NET, 2 = tumor core, 3 = original, 4 = ET, 5 = complete tumor 

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
            img = (img-img.mean()) / img.std() 
            img = img.astype('float32')
            for slice in range(0,155):     # choose the slice range
                img_t = img[slice,:,:]
                img_t =img_t.reshape((1,)+img_t.shape)
                img_t =img_t.reshape((1,)+img_t.shape)
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
            img = (img-img.mean()) / img.std()
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
                img[img == 3] = 1   
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
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
                img[img == 3] = 1     
            img = img.astype('float32')
        for slice in range(0,155):     #choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
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
                img[img == 3] = 1   
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
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
                img[img == 3] = 1  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
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
                img[img == 3] = 1  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
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
                img[img == 3] = 1  
            img = img.astype('float32')
        for slice in range(0,155):     # choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape) 
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
        return np.array(imgs)


# Loading dataset

# flair
flair = create_data_ft('E:/code/custom final/', '**/*_flair.nii.gz', '**/*_t2.nii.gz', label=False, mod=1, resize=(155,img_size,img_size))

# t2
t2 = create_data_ft('E:/code/custom final/', '**/*_flair.nii.gz', '**/*_t2.nii.gz', label=False, mod=2, resize=(155,img_size,img_size))

# t1
t1 = create_data_t1('E:/code/custom final', '**/*_t1.nii.gz', label=False, resize=(155,img_size,img_size))

# t1c
t1c = create_data_t1c('E:/code/custom final/', '**/*_t1ce.nii.gz', label=False, resize=(155,img_size,img_size))

# full tumor mask
label_num = 5
mask_full = create_data_mask_full('E:/code/custom final', '**/*_seg.nii.gz', label=True, resize=(155,img_size,img_size))

# tumor core mask
label_num = 2
mask_core = create_data_mask_core('E:/code/custom final', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))

# enhancing tumor mask
label_num = 4
mask_ET = create_data_mask_et('E:/code/custom final', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))

# complete tumor mask
label_num = 3
mask_all = create_data_mask_all('E:/code/custom final/', '**/*seg.nii.gz', label=True, resize=(155,img_size,img_size))

# Defining metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# U-net for full tumor segmentation    
def unet_model():
    inputs = Input((2, img_size, img_size))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (inputs)
    batch1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch1)
    batch1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D((2, 2)) (batch1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (pool1)
    batch2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch2)
    batch2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D((2, 2)) (batch2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (pool2)
    batch3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch3)
    batch3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D((2, 2)) (batch3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (pool3)
    batch4 = BatchNormalization(axis=1)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch4)
    batch4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (batch4)
    
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (pool4)
    batch5 = BatchNormalization(axis=1)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (batch5)
    batch5 = BatchNormalization(axis=1)(conv5)
    
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (batch5)
    up6 = concatenate([up6, conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same') (up6)
    batch6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch6)
    batch6 = BatchNormalization(axis=1)(conv6)
    
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (batch6)
    up7 = concatenate([up7, conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (up7)
    batch7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch7)
    batch7 = BatchNormalization(axis=1)(conv7)
    
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (batch7)
    up8 = concatenate([up8, conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (up8)
    batch8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch8)
    batch8 = BatchNormalization(axis=1)(conv8)
    
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (batch8)
    up9 = concatenate([up9, conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (up9)
    batch9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch9)
    batch9 = BatchNormalization(axis=1)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(batch9)

    model = Model(inputs=[inputs], outputs=[conv10])

    adam = Adam(lr = 0.0001)
    model.compile(optimizer = adam, loss =dice_coef_loss, metrics = [dice_coef])


    return model


# U-net for Tumor core and ET
img_size_nec = 64

def unet_model_nec3():
    inputs = Input((1, img_size_nec, img_size_nec))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (inputs)
    batch1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch1)
    batch1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D((2, 2)) (batch1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (pool1)
    batch2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch2)
    batch2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D((2, 2)) (batch2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (pool2)
    batch3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch3)
    batch3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D((2, 2)) (batch3)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same') (pool3)
    batch5 = BatchNormalization(axis=1)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch5)
    batch5 = BatchNormalization(axis=1)(conv5)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (batch5)
    up7 = concatenate([up7, conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (up7)
    batch7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch7)
    batch7 = BatchNormalization(axis=1)(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (batch7)
    up8 = concatenate([up8, conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (up8)
    batch8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch8)
    batch8 = BatchNormalization(axis=1)(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (batch8)
    up9 = concatenate([up9, conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (up9)
    batch9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch9)
    batch9 = BatchNormalization(axis=1)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(batch9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=LR), loss=dice_coef_loss, metrics=[dice_coef])

    return model


# Defining function for post-processing data
def paint_color_algo(pred_full, pred_core , pred_ET , li):   
        # first put the pred_full on T1c
        pred_full[pred_full > 0.2] = 2      # 240x240
        pred_full[pred_full != 2] = 0
        pred_core[pred_core > 0.2] = 1      # 64x64
        pred_core[pred_core != 1] = 0
        pred_ET[pred_ET > 0.2] = 4          # 64x64
        pred_ET[pred_ET != 4] = 0

        total = np.zeros((1,240,240),np.float32)  
        total[:,:,:] = pred_full[:,:,:]
        for i in range(pred_core.shape[0]):
            for j in range(64):
                for k in range(64):
                    if pred_core[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                        total[0,li[i][0]+j,li[i][1]+k] = pred_core[i,0,j,k]
                    if pred_ET[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                        total[0,li[i][0]+j,li[i][1]+k] = pred_ET[i,0,j,k]



        return total


# Defining cropping function to crop T1c image with the output of full tumor preiction
def crop_tumor_tissue(x, pred, size):
   # args: T1c image, prediction of full tumor, size default  64x64
    crop_x = []
    list_xy = []
    p_tmp = pred[0,:,:]
    p_tmp[p_tmp>0.2] = 1    # threshold
    p_tmp[p_tmp !=1] = 0
    # get middle point from prediction of full tumor
    index_xy = np.where(p_tmp==1)   # get all the axial of pixel which value is 1

    if index_xy[0].shape[0] == 0:   # skip when no tumor
        return [],[]  
    center_x = (max(index_xy[0]) + min(index_xy[0])) / 2 
    center_y = (max(index_xy[1]) + min(index_xy[1])) / 2 
    
    if center_x >= 176:
            center_x = center_x-8
        
    length = max(index_xy[0]) - min(index_xy[0])
    width = max(index_xy[1]) - min(index_xy[1])
        
    if width <= 64 and length <= 64:  # 64x64
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size/2),int(center_y - size/2)))
            
    if width > 64 and length <= 64:  # 64x128
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size/2),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size/2) : int(center_x + size/2),int(center_y + 1) : int(center_y + size + 1)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size/2),int(center_y)))
            
    if width <= 64 and length > 64:  # 128x64  
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size),int(center_y - size/2)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y - size/2) : int(center_y + size/2)]
        crop_x.append(img_x)
        list_xy.append((int(center_x),int(center_y - size/2)))
            
    if width > 64 and length > 64:  # 128x128
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y - size) : int(center_y)]
        crop_x.append(img_x)
        list_xy.append((int(center_x),int(center_y - size)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x - size) : int(center_x),int(center_y + 1) : int(center_y + size + 1)]
        crop_x.append(img_x)
        list_xy.append((int(center_x - size),int(center_y)))
            
        img_x = np.zeros((1,size,size),np.float32)
        img_x[:,:,:] = x[:,int(center_x + 1) : int(center_x + size + 1),int(center_y + 1) : int(center_y + size + 1)]
        crop_x.append(img_x)
        list_xy.append((int(center_x),int(center_y)))
        
    
        
    return np.array(crop_x) , list_xy   # (y,x)

# Training full tumor
model_full= unet_model()

unet_save = ModelCheckpoint('Full-Tumor-weights.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
callbacks_list = [unet_save]

model_full.fit(full, y_full, batch_size=8,epochs = 30, validation_split=0.2, verbose=1,callbacks=callbacks_list, shuffle=True)

from keras.models import load_model
model_full.save('fulltumor.h5',include_optimizer=True)


# cropping T1c with full tumor mask for obtaining patches of T1c for tumor core and enhancing tumor
crop = np.zeros((7232,1,64,64),np.float32)
li = []
j= 11315
n=0
sum=0
for i in range(0,j):
    sub_crop , sub_li = crop_tumor_tissue(t1c[i,:,:,:],y_full[i,:,:,:],64)
    for k in range(0,num):
        crop[n,:,:,:]=sub_crop[k,:,:,:]
        n=n+1;

# cropping tumor core mask with full tumor mask for obtaining patches of tumor core mask
cropyc = np.zeros((7232,1,64,64),np.float32)
liyc = []
j= 11315
n=0
sum=0
for i in range(0,j):
    sub_cropyc , sub_liyc = crop_tumor_tissue(y_core[i,:,:,:],y_full[i,:,:,:],64)
    for k in range(0,num):
        cropyc[n,:,:,:]=sub_cropyc[k,:,:,:]
        n=n+1;

# cropping enhancing tumor mask with full tumor mask for obtaining patches of enhancing tumor mask
cropyet = np.zeros((7232,1,64,64),np.float32)
liyet = []
j= 11315
n=0
sum=0
for i in range(0,j):
    sub_cropyet , sub_liyet = crop_tumor_tissue(y_et[i,:,:,:],y_full[i,:,:,:],64)
    for k in range(0,num):
        cropyet[n,:,:,:]=sub_cropyet[k,:,:,:]
        n=n+1;


# Training tumor core
model_core = unet_model_nec3()

unet_save = ModelCheckpoint('Tumor-Core-weights.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
callbacks_list = [unet_save]

model_core.fit(crop, cropyc, batch_size=8,epochs = 50, validation_split=0.2, verbose=1,callbacks=callbacks_list, shuffle=True)

from keras.models import load_model
model_core.save('tumorcore.h5',include_optimizer=True)


# Training enhancing tumor
model_ET = unet_model_nec3()

unet_save = ModelCheckpoint('Enhancing-Tumor-weights.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
callbacks_list = [unet_save]

model_ET.fit(crop, cropyet, batch_size=8,epochs = 10, validation_split=0.2, verbose=1,callbacks=callbacks_list, shuffle=True)

from keras.models import load_model
model_ET.save('enhancingtumor.h5',include_optimizer=True)
