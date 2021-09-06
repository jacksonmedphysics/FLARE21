from build_unet_localiser import build_model
import pandas as pd

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import matplotlib
matplotlib.use('Agg')  ##change back after testing!!!
from matplotlib import pyplot as plt

import tensorflow.keras as keras
import numpy as np
import os
from os.path import join
import random
import SimpleITK as sitk
import numpy
from scipy import ndimage
import tensorflow as tf

from scipy.ndimage.measurements import center_of_mass
from numpy.random import rand
import datetime

from scipy.signal import fftconvolve


import sys
import argparse
from tensorflow.keras.callbacks import CSVLogger
import random

parser=argparse.ArgumentParser(description='Keras Region Finder')
parser.add_argument("region", type=str,help="region name")
parser.add_argument('extent', type=str,help='"X Y Z" Physical Extent of Whole Body Volume in mm')
parser.add_argument('resolution',type=str,help='"X Y Z" number of voxels on each axis in whole body volume')
parser.add_argument('expansion_range',type=float,help='Range to Expand Structure in mm')

args=parser.parse_args()    
region=args.region
extent=args.extent.split(' ')
expansion_range=args.expansion_range

xextent=float(extent[0])
yextent=float(extent[1])
zextent=float(extent[2])
resolution=args.resolution.split(' ')
xdim=int(resolution[0])
ydim=int(resolution[1])
zdim=int(resolution[2])

spacing_string=str(int(xextent))+'_'+str(int(yextent))+'_'+str(int(zextent))+'-'+str(xdim)+'_'+str(ydim)+'_'+str(zdim)

"""
python Region_Finder_min_output.py pancreas "400 400 1500" "96 96 144" 25.0
trains on region 'pancreas' with WB extent of 400x400x1500mm, 96x96x144 resolution (both xyz), and 25mm label expansion
"""


fname=os.path.basename(__file__).replace('.py','')+'-'+region+'-'+spacing_string+'-'+str(expansion_range)
print(fname)

def new_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return
new_dir('logs')
new_dir('models')


data_dir='../data' #location of ct and lable .nii folders
n_epochs=100 #100

casedf=pd.read_csv('flare21_caselist_shuffled.csv',index_col=0)

casedf=casedf[casedf[region]>0] #addition to remove nil labels which don't have a crop region

cases=casedf.case.values.tolist()


log_dir="logs/"+fname + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
label_dir=join(data_dir,region)
ct_dir=join(data_dir,'ct')


training_percent=85

zeds=np.zeros((zdim,ydim,xdim))
xy_max_crop_dist=50 #mm
resample_extent=(xextent,yextent,zextent)  #max extent was 1942
final_resample_dims=(zdim,ydim,xdim)  #numpy zyx
final_resample_spacing=np.array((resample_extent[0]/final_resample_dims[2],resample_extent[1]/final_resample_dims[1],resample_extent[2]/final_resample_dims[0])) #sitk xyz
rss=final_resample_spacing


validation_case_path=False

radii=[1,3,5,8,11,15,19,24,29,35,40]


def get_expansion_sphere(radius,spacing): #note need to reverse xyz... i think
    xlim=np.ceil(radius/spacing[0])
    ylim=np.ceil(radius/spacing[1])
    zlim=np.ceil(radius/spacing[2])
    x=np.arange(-xlim,xlim+1,1)
    y=np.arange(-ylim,ylim+1,1)
    z=np.arange(-zlim,zlim+1,1)   
    xx,yy,zz=np.meshgrid(x,y,z)
    sphere=(np.sqrt((xx*spacing[0])**2+(yy*spacing[1])**2+(zz*spacing[2])**2)<=radius).astype(np.float32)
    sphere=np.swapaxes(sphere,0,2)
    return sphere

expansion_filter=get_expansion_sphere(expansion_range,final_resample_spacing)

def get_multi_spheres(radius_list,spacing):
    xlim=np.ceil(max(radius_list)/spacing[0])
    ylim=np.ceil(max(radius_list)/spacing[1])
    zlim=np.ceil(max(radius_list)/spacing[2])
    x=np.arange(-xlim,xlim+1,1)
    y=np.arange(-ylim,ylim+1,1)
    z=np.arange(-zlim,zlim+1,1)   
    xx,yy,zz=np.meshgrid(x,y,z)
    spheres=np.zeros((xx.shape[0],xx.shape[1],xx.shape[2],len(radius_list)))
    for i in range(len(radius_list)):
        sphere=(np.sqrt((xx*spacing[0])**2+(yy*spacing[1])**2+(zz*spacing[2])**2)<=radius_list[i]).astype(np.float32)
        spheres[...,i]=sphere/sphere.sum()
    return tf.convert_to_tensor(np.expand_dims(spheres.astype('float32'),-2))

tf_spheres=get_multi_spheres(radii,final_resample_spacing)

def weight_label_voxels(y_true, spheres):
    greater_thresh=0.99
    less_thress=0.01
    y_true=tf.expand_dims(y_true,-1)
    conved1=tf.nn.conv3d(y_true,spheres,strides=[1,1,1,1,1],padding='SAME')
    inner1=tf.cast(tf.greater(conved1,greater_thresh),dtype=tf.float32)
    outer1=tf.cast(tf.less(conved1,less_thress),dtype=tf.float32)
    dist1=tf.expand_dims(tf.reduce_sum(outer1,axis=-1),-1)
    dist2=tf.expand_dims(tf.reduce_sum(inner1,axis=-1),-1)
    return (dist1/3.), (1+dist2/2.)  #weighted


def non_intersecting(y_true,y_pred):
    return K.sum(tf.abs(tf.subtract(y_true[...,1],y_pred[...,1])))

def agreement(y_true,y_pred):
    return K.sum(tf.multiply(y_true[...,1],y_pred[...,1]))

def accuracy(y_true,y_pred):
    return (agreement(y_true,y_pred)-non_intersecting(y_true,y_pred))/(K.sum(y_true[...,1]))

def non_intersecting_fraction(y_true,y_pred):
    return non_intersecting(y_true,y_pred)/(K.sum(y_true[...,1]))

def volume_ratio(y_true,y_pred):
    return K.sum(y_pred[...,1])/K.sum(y_true[...,1])



def calc_z_COM(y):
    zlin=tf.convert_to_tensor(np.linspace(1,zdim,zdim).astype('float32'))
    #print(zlin)
    z_1axis=K.sum(y[0,...,1],axis=(1,2))
    return K.sum(zlin*z_1axis)/K.sum(y[...,1])-1

def calc_y_COM(y):
    ylin=tf.convert_to_tensor(np.linspace(1,ydim,ydim).astype('float32'))
    y_1axis=K.sum(y[0,...,1],axis=(0,2))
    return K.sum(ylin*y_1axis)/K.sum(y[...,1])-1

def calc_x_COM(y):
    xlin=tf.convert_to_tensor(np.linspace(1,xdim,xdim).astype('float32'))
    x_1axis=K.sum(y[0,...,1],axis=(0,1))
    return K.sum(xlin*x_1axis)/K.sum(y[...,1])-1
    
def x_com_error(y_true,y_pred):
    return abs(rss[0]*(calc_x_COM(y_true)-calc_x_COM(y_pred)))

def y_com_error(y_true,y_pred):
    return abs(rss[1]*(calc_y_COM(y_true)-calc_y_COM(y_pred)))

def z_com_error(y_true,y_pred):
    return abs(rss[2]*(calc_z_COM(y_true)-calc_z_COM(y_pred)))

def standard_dice(y_true,y_pred):
    return (2* agreement(y_true,y_pred))/ (K.sum(y_true[...,1]) + K.sum(y_pred[...,1]))

def distance_dice(y_true, y_pred): #distance weighted dice coefficient
    weighted_outer,weighted_inner=weight_label_voxels(y_true[...,1],tf_spheres)
    weighted_prediction=tf.expand_dims(y_pred[...,1],-1) *  weighted_outer
    numerator=K.sum(tf.expand_dims(tf.multiply(y_true[...,1],y_pred[...,1]),-1) * weighted_inner)
    return numerator / (K.sum(y_true[...,1]) + K.sum(weighted_prediction))

def accuracy_loss(y_true,y_pred):
    return -accuracy(y_true,y_pred)


def standard_dice_loss(y_true,y_pred):
    return -standard_dice(y_true,y_pred)

def distance_dice_loss(y_true,y_pred):
    return -distance_dice(y_true,y_pred)

def random_crop(image,label):
    np.random.seed(random.randint(0,65535))
    try:
        ar=sitk.GetArrayFromImage(label)
        com=center_of_mass(ar)
        spacing=image.GetSpacing()    
        cropfilter=sitk.CropImageFilter()
        x_crop=int(1E6)
        y_crop=int(1E6)
        while x_crop>com[2] and (ar.shape[2]-x_crop)<com[2]:
            x_crop=int(xy_max_crop_dist/spacing[0]*rand())
        while y_crop>com[1] and (ar.shape[1]-y_crop)<com[1]:
            y_crop=int(xy_max_crop_dist/spacing[1]*rand())
        z_lower_crop=int(0.7*com[0]*rand())
        z_upper_crop=int(0.7*(ar.shape[0]-com[0])*rand())
        cropfilter.SetLowerBoundaryCropSize([x_crop,y_crop, z_lower_crop])
        cropfilter.SetUpperBoundaryCropSize([x_crop,y_crop, z_upper_crop])
        image_crop = cropfilter.Execute(image)
        label_crop= cropfilter.Execute(label)
        return image_crop, label_crop
    except Exception as e:
        print(image.GetSize(),e)
        return image,label

def augment_case(x,y):
    np.random.seed(random.randint(0,65535))
    ct=x
    lab=y
    max_rotation_deg=20.
    max_translation=7.  #25
    max_gauss_sigma=1.0
    max_hu_shift=30
    max_noise=100
    sharpening_range=0.6
    sharpening_alpha=0.5
    rotx=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    roty=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    rotz=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    tx=(np.random.rand()-0.5)*max_translation
    ty=(np.random.rand()-0.5)*max_translation
    tz=(np.random.rand()-0.5)*max_translation
    sig=np.random.rand()*max_gauss_sigma
    sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
    salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
    hu_shift=(np.random.rand()-0.5)*max_hu_shift
    img=sitk.GetImageFromArray(ct)
    sitk_label=sitk.GetImageFromArray(lab)
    img.SetSpacing(final_resample_spacing)
    sitk_label.SetSpacing(final_resample_spacing)
    com=center_of_mass(lab)
    initial_transform=sitk.Euler3DTransform()
    initial_transform.SetCenter([com[0],com[1],com[2]])
    registration_method = sitk.ImageRegistrationMethod()
    initial_transform.SetParameters((rotx,roty,rotz,tx,ty,tz))
    img = sitk.Resample(img, img, initial_transform, sitk.sitkLinear, -1000., sitk.sitkInt16)
    label=sitk.Resample(sitk_label,img,initial_transform,sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    ar=sitk.GetArrayFromImage(img)
    lab_ar=sitk.GetArrayFromImage(label)
    blurred_ar=ndimage.gaussian_filter(ar,sharp)
    sharpened=ar+salpha*(ar-blurred_ar)
    ar=sharpened
    ar=ndimage.gaussian_filter(ar,sigma=sig)
    ar+=int(hu_shift)
    ar+=((np.random.random(ar.shape)-0.5)*max_noise).astype('int16')
    return ar.astype('float32'),lab_ar.astype('float32')

def load_case(label_name,augment=True):
    while True:
        ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name)),sitk.sitkInt16)
        label=sitk.ReadImage(join(label_dir,label_name))
        rs=sitk.ResampleImageFilter()
        rs.SetReferenceImage(ct)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        label=sitk.Cast(rs.Execute(label),sitk.sitkInt16)
        y=sitk.GetArrayFromImage(label)
        label=sitk.GetImageFromArray(y)
        label.SetOrigin(ct.GetOrigin())
        label.SetSpacing(ct.GetSpacing())
        label.SetDirection(ct.GetDirection())
        if augment:
            ct,label=random_crop(ct,label) #applies random cropping to CT/Label combo
        origin=np.array(ct.GetOrigin())
        original_dims=np.array(ct.GetSize())
        original_spacing=np.array(ct.GetSpacing())
        original_extent=original_dims*original_spacing
        if True:
            origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
        origin[2]=origin[2]-origin_shift
        delta_extent=resample_extent-original_extent
        delta_x=delta_extent[0]/2.
        delta_y=delta_extent[1]/2.
        new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
        ref=sitk.GetImageFromArray(zeds)
        ref.SetSpacing(final_resample_spacing)
        ref.SetOrigin(new_origin)
        rs.SetReferenceImage(ref)
        rs.SetDefaultPixelValue(-1000)
        rs.SetInterpolator(sitk.sitkLinear)
        ct_midres=rs.Execute(ct)
        rs.SetDefaultPixelValue(0)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        label_midres=rs.Execute(label)
        x=sitk.GetArrayFromImage(ct_midres)
        y=sitk.GetArrayFromImage(label_midres)
        if augment:
            x,y=augment_case(x,y)
        if y.sum()>0:
            if x.shape==(zdim,ydim,xdim):
                if y.shape==(zdim,ydim,xdim):
                    y=(fftconvolve(y,expansion_filter,'same')>0.99).astype('float32')
                    y=np.expand_dims(y,-1)
                    y=numpy.append((y!=1),y,axis=-1)
                    x=np.expand_dims(np.expand_dims(x,0),-1)
                    y=np.expand_dims(y,0)
                    return x,y
                else:
                    print('y.shape incorrect',label_name, y.shape)
            else:
                print('x.shape incorrect',label_name, x.shape)
        else:
            print('y.sum()==0',label_name,y.sum())

def load_validation_case(label_name):
    if True:
        ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name)),sitk.sitkInt16)
        label=sitk.ReadImage(join(label_dir,label_name))
        rs=sitk.ResampleImageFilter()
        rs.SetReferenceImage(ct)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        label=rs.Execute(label)
        origin=np.array(ct.GetOrigin())
        original_dims=np.array(ct.GetSize())
        original_spacing=np.array(ct.GetSpacing())
        original_extent=original_dims*original_spacing
        origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
        origin[2]=origin[2]-origin_shift
        delta_extent=resample_extent-original_extent
        delta_x=delta_extent[0]/2.
        delta_y=delta_extent[1]/2.
        new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
        ref=sitk.GetImageFromArray(zeds)
        ref.SetSpacing(final_resample_spacing)
        ref.SetOrigin(new_origin)
        rs.SetReferenceImage(ref)
        rs.SetDefaultPixelValue(-1000)
        rs.SetInterpolator(sitk.sitkLinear)
        ct_midres=rs.Execute(ct)
        rs.SetDefaultPixelValue(0)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetReferenceImage(ct_midres)
        label_midres=rs.Execute(label)
        x=sitk.GetArrayFromImage(ct_midres)
        y=sitk.GetArrayFromImage(label_midres)
        y=(fftconvolve(y,expansion_filter,'same')>0.99).astype('float32')
        y=np.expand_dims(y,-1)
        y=numpy.append((y!=1),y,axis=-1)
        x=np.expand_dims(np.expand_dims(x,0),-1)
        y=np.expand_dims(y,0)
        return x,y

def training_generator(training_cases):
    idx=0
    size=len(training_cases)
    while True:
        if idx==size:
            idx=0
        y=np.array([0])
        while y.sum()==0:
            x,y=load_case(training_cases[idx],augment=True) #changed to false
        yield x.astype('float32'),y.astype('float32')
        idx+=1

def testing_generator(validation_cases):
    idx_t=0
    size=len(validation_cases)
    while True:
        if idx_t==size:
            idx_t=0
        #x,y=load_case(validation_cases[idx_t],augment=False)
        x,y=load_validation_case(validation_cases[idx_t])
        yield x.astype('float32'),y.astype('float32')
        idx_t+=1


     
if __name__ == '__main__':

    num_training=int(len(cases)*(training_percent/100.))
    num_validation=int(len(cases))-num_training
    if num_validation<5:
        num_training=int(len(cases))-5 #makes sure there are at least 5 validation cases
    training_cases=cases[:num_training]
    validation_cases=cases[num_training:]
    print(cases)
    num_training_cases=len(training_cases)
    num_testing_cases=len(validation_cases)
    gen=training_generator(training_cases)
    gen_test=testing_generator(validation_cases)
    inp_shape= (zdim,ydim,xdim,1)    

    print('X[0].shape: ',inp_shape)    
    model = build_model(inp_shape)
    adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.0)
    model.compile(optimizer=adam, loss=distance_dice_loss, metrics=[agreement,non_intersecting,accuracy,standard_dice,x_com_error,y_com_error,z_com_error])
    model.summary()
    ##########################################################################################
    checkpointer = ModelCheckpoint('models/'+fname+'.hdf5', save_best_only=True, mode='max', monitor='val_standard_dice')

    csv_logger = CSVLogger('logs/'+fname+'.csv')
    

    callbacks=[checkpointer,csv_logger]
    history=model.fit(x=gen,validation_data=gen_test, epochs=n_epochs, callbacks=callbacks,steps_per_epoch=int(num_training_cases),validation_steps=int(num_testing_cases))  #(available_cases-validation_size)
    print(history.history.keys())
    hist_df = pd.DataFrame(history.history)

    
