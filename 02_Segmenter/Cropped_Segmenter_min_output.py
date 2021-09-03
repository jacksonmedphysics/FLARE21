from build_unet_model_4res import build_model
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

global epoch_dir
import sys
import argparse
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd

from load_cropped_case_v1 import load_cropped

parser=argparse.ArgumentParser(description='Keras Region Finder')
parser.add_argument("region", type=str,help="region name")
parser.add_argument('resolution',type=str,help='"X Y Z" number of voxels on each axis in whole body volume')

args=parser.parse_args()    
region=args.region
df_extent=pd.read_csv('crop_offsets_v2_flare.csv')
crop_offsets=df_extent[df_extent.region==region].values[0][1:]
print(crop_offsets)
xextent=float(crop_offsets[1]-crop_offsets[0])
yextent=float(crop_offsets[3]-crop_offsets[2])
zextent=float(crop_offsets[5]-crop_offsets[4])
resolution=args.resolution.split(' ')
xdim=int(resolution[0])
ydim=int(resolution[1])
zdim=int(resolution[2])

rss=[(xextent/xdim),(yextent/ydim),(zextent/zdim)] #resample spacing

spacing_string=str(xdim)+'_'+str(ydim)+'_'+str(zdim)
crop_dimensions=(xdim,ydim,zdim)

"""
python Cropped_Segmenter_min_output.py pancreas "160 160 144" 
"""


fname=os.path.basename(__file__).replace('.py','')+'-'+region+'-'+spacing_string
print(fname)

#sys.exit()
def new_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return
new_dir('logs')
new_dir('models')
#new_dir(fname)
validation_epochs=join('validation_epochs',fname)
new_dir(validation_epochs)
data_dir='../data'

casedf=pd.read_csv('flare21_caselist_shuffled.csv',index_col=0)

casedf=casedf[casedf[region]>0] #addition to remove nil labels which don't have a crop region

cases=casedf.case.values.tolist()

log_dir="logs/"+fname + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
label_dir=join(data_dir,region)
ct_dir=join(data_dir,'ct')

training_percent=85

validation_case_path=False



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

zero_remover=10

def standard_dice(y_true,y_pred):
    return (2* agreement(y_true,y_pred)+zero_remover)/ (K.sum(y_true[...,1]) + K.sum(y_pred[...,1])+zero_remover)

def accuracy_loss(y_true,y_pred):
    return -accuracy(y_true,y_pred)

def dice_loss(y_true,y_pred):
    return -standard_dice(y_true,y_pred)

def training_generator(training_cases):
    idx=0
    size=len(training_cases)
    while True:
        if idx==size:
            idx=0
        y=np.array([0])
        while y.sum()==0:
            x,y=load_cropped(join(ct_dir,training_cases[idx]),join(label_dir,training_cases[idx]),crop_offsets,crop_dimensions,augment=True) #changed to false
        yield x.astype('float32'),y.astype('float32')
        idx+=1

def testing_generator(validation_cases):
    idx_t=0
    size=len(validation_cases)
    while True:
        if idx_t==size:
            idx_t=0
        #x,y=load_case(validation_cases[idx_t],augment=False)
        x,y=load_cropped(join(ct_dir,validation_cases[idx_t]),join(label_dir,validation_cases[idx_t]),crop_offsets,crop_dimensions,augment=False)
        #x,y=load_validation_case(validation_cases[idx_t])
        yield x.astype('float32'),y.astype('float32')
        idx_t+=1


     
if __name__ == '__main__':

    epoch_dir=validation_epochs

    
    if validation_case_path:  #catch to retrain with selected validation cases
        validation_cases=[]
        f=open(validation_case_path,'r')
        for case in f.readlines():
            validation_cases.append(case.replace('\n',''))
        f.close()
        training_cases=[]
        for case in cases:
            if case not in validation_cases:
                training_cases.append(case)
    else: 

        num_training=int(len(cases)*(training_percent/100.))
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
    model.compile(optimizer=adam, loss=dice_loss, metrics=[agreement,non_intersecting,accuracy,standard_dice])
    model.summary()
    ##########################################################################################
    checkpointer = ModelCheckpoint('models/'+fname+'.hdf5', save_best_only=True, mode='max', monitor='val_standard_dice')

    csv_logger = CSVLogger('logs/'+fname+'.csv')
    
    callbacks=[checkpointer,csv_logger]
    history=model.fit(x=gen,validation_data=gen_test, epochs=100, callbacks=callbacks,steps_per_epoch=int(num_training_cases),validation_steps=int(num_testing_cases))  #(available_cases-validation_size)
    print(history.history.keys())
    hist_df = pd.DataFrame(history.history)

    
