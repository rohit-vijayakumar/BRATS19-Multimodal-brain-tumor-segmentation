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


model_full= unet_model()
model_core = unet_model_nec3()
model_ET = unet_model_nec3()

model_full.load_weights('C:/Users/Rohit/Full-Tumor-weights.h5')
model_core.load_weights('C:/Users/Rohit/Tumor-Core-weights.h5')
model_ET.load_weights('C:/Users/Rohit/Enhancing-Tumor-weights.h5')


# Peform segmentation of complete tumor
def predict(i,j) :

    full = np.zeros((1,2,240,240),np.float32)  ####x for full tumor
    full[:,:1,:,:] = flair[i:j,:,:,:]
    full[:,1:,:,:] = t2[i:j,:,:,:]

    pred_full = model_full.predict(full)
    crop , li = crop_tumor_tissue(t1c[j,:,:,:],pred_full[0,:,:,:],64)
    if(len(crop)!=0):
        pred_core = model_core.predict(crop)
        pred_ET = model_ET.predict(crop)

        tmp = paint_color_algo(pred_full[0,:,:,:], pred_core, pred_ET, li)

        core = np.zeros((1,240,240),np.float32)
        ET = np.zeros((1,240,240),np.float32)
        core[:,:,:] = tmp[:,:,:]
        ET[:,:,:] = tmp[:,:,:]
        core[core == 4] = 1
        core[core != 1] = 0
        ET[ET != 4] = 0
        return np.array(tmp)
    else:
        return np.array(pred_full)


# Saving segmented output as 3D nii file
import nibabel as nib
final1 = np.zeros((240,240,155),np.float32)
a= 0
b=1
num=0
while(a<=153):
    final1[:,:,num] = test(a,b)
    a=a+1
    b=b+1
    num=num+1;
tmp1 = np.rot90(final1)
final2 = np.flipud(tmp1)
img1 = nib.load('E:/code/custom final testing/BraTS19_TCIA01_231_1/BraTS19_TCIA01_231_1_flair.nii.gz')
img_affine = img1.affine
img = nib.Nifti1Image(final2, img_affine)
print("Processing Numpy Array to 3D Nifti")
img.to_filename('C:/Users/Rohit/Desktop/prediction.nii.gz')