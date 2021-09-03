import SimpleITK as sitk
from tensorflow.keras.models import load_model
#from keras import backend as K
#import smooth_labels
import numpy as np
import pandas as pd
#from numba import cuda
import time
start=time.time()
#from smooth_labels_v3 import smooth_labels

"""
Start Interactive Session with:

sinteractive -p gpu --mem 64G --gres=gpu:T4 --time=0-16:00
module load tensorflow/2.4.0-gpu
export SINGULARITY_BINDPATH="/researchers/price.jackson,/physical_sciences"
cd /physical_sciences/organ_contouring/FLARE21_challenge/two_stage_inference
tensorflow.sif
python FLARE21_inference_v1.py
"""

import sys, os, shutil
from os.path import join
import pydicom as dicom
#sys.path.append('/home/price/Neural_Networks/SliceSelector/keras')
#import crop_liver_spleen_v4 as crop_liver_spleen
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
identity=sitk.Transform(3,sitk.sitkIdentity)
from scipy import ndimage
#import matplotlib.pyplot as plt
##
##import gdcm_lister2_modality_filenames_v4 as gdcm_lister2

#sys.path.append('/home/price/NET_Tumour_Burden')
#import dcm_lister2
import shutil
#import blob_detector_py8_func

#import plastimatch_pj
#import dicom
import os
import subprocess



##crop_df=pd.DataFrame(columns=['region','xmin','xmax','ymin','ymax','zmin','zmax'],
##                data=[['spleen',-90,75,-100,80,-100,90],
##                      ['trachea',-75,120,-70,150,-130,120],
##                      ['bladder',-65,65,-70,80,-65,70],
##                      ['esophagus',45,45,-70,45,-140,170],
##                      ['heart',-90,95,-80,85,-80,85],
##                      ['liver',-150,220,-165,180,-175,115],
##                      ['lt_kidney',-65,65,-65,65,-95,95],
##                      ['lt_lung',-110,95,-130,120,-170,160],
##                      ['rt_kidney',-65,65,-65,65,-95,95],
##                      ['rt_lung',-100,130,-140,135,-165,220],
##                      ['prostate',-90,90,-90,90,-90,90]])

crop_df=pd.read_csv('crop_offsets_v2_flare.csv')

seg_model_dir='../cropped_seg/models'
region_model_dir='../region_finder/models'
input_dir='input'
delete_input=False
combined_label_dir='combined_labels'


model_df=pd.DataFrame(columns=['region','region_model','seg_model'],
                      data=[['liver','Region_Finder_update_v5_std_dice-liver-400_400_1500-96_96_144-25.0.hdf5','Cropped_Segmenter_v1-liver-160_160_144.hdf5'],
                            ['lt_kidney','Region_Finder_update_v5_std_dice-lt_kidney-400_400_1500-96_96_144-25.0.hdf5','Cropped_Segmenter_v1-lt_kidney-160_160_144.hdf5'],
                            ['rt_kidney','Region_Finder_update_v5_std_dice-rt_kidney-400_400_1500-96_96_144-25.0.hdf5','Cropped_Segmenter_v1-rt_kidney-160_160_144.hdf5'],
                            ['pancreas','Region_Finder_update_v5_std_dice-pancreas-400_400_1500-96_96_144-25.0.hdf5','Cropped_Segmenter_v1-pancreas-160_160_144.hdf5'],
                            ['spleen','Region_Finder_update_v5_std_dice-spleen-400_400_1500-96_96_144-25.0.hdf5','Cropped_Segmenter_v1-spleen-160_160_144.hdf5']])

regions=['liver','lt_kidney','rt_kidney','pancreas','spleen']


def get_physical_bounding_box(image):
    size=image.GetSize()
    corners=[(0,0,0),(0,size[1]-1,0),(0,size[1]-1,size[2]-1),(0,0,size[2]-1),
             (size[0]-1,0,0,),(size[0]-1,size[1]-1,0),(size[0]-1,size[1]-1,size[2]-1),(size[0]-1,0,size[2]-1)]
    x,y,z=[],[],[]
    for corner in corners:
        coord=image.TransformIndexToPhysicalPoint(corner)
        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])
    
    return np.array((min(x),min(y),min(z))),np.array((max(x),max(y),max(z)))

def resample_orthogonal(image,label=False,reference_spacing=(1.0,1.0,2.0),default_value=-1024):
    """Resamples sitk image and (optionally label object) to orthogonal direction cosines
    output image spacing can be defined (x,y,z) as well as default padding value"""
    min_extent,max_extent=get_physical_bounding_box(image)
    physical_extent=max_extent-min_extent
    reference_spacing=np.array(reference_spacing)
    dimensions=physical_extent/reference_spacing
    zeds=np.zeros(np.ceil(dimensions[::-1]).astype('int16')) #need to reverse dimensions (z,y,x)
    ref=sitk.GetImageFromArray(zeds)
    ref.SetSpacing(reference_spacing)
    ref.SetOrigin(min_extent)
    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetDefaultPixelValue(default_value)
    resampled_image=rs.Execute(image)
    if not isinstance(label,bool):
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        resampled_label=rs.Execute(label)
        return resampled_image,resampled_label
    else:
        return resampled_image


def image_to_wb_input(im,resample_extent,resample_dimensions):
    zeds=np.zeros(np.flip(np.array(resample_dimensions)))
    final_resample_spacing=np.array(resample_extent)/np.array(resample_dimensions)
    rs=sitk.ResampleImageFilter()
    origin=np.array(im.GetOrigin())
    original_dims=np.array(im.GetSize())
    original_spacing=np.array(im.GetSpacing())
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
    ct_midres=rs.Execute(im)
    x=sitk.GetArrayFromImage(ct_midres)
    x=np.expand_dims(np.expand_dims(x,0),-1)
    return x,ct_midres

def load_inference_ct(im,centroid,crop_offsets,crop_dimensions):
    crop_extent=np.array([crop_offsets[1]-crop_offsets[0],crop_offsets[3]-crop_offsets[2],crop_offsets[5]-crop_offsets[4]])
    origin_offset=np.array([crop_offsets[0],crop_offsets[2],crop_offsets[4]])    
    crop_dimensions=np.array(crop_dimensions)
    crop_spacing=crop_extent/crop_dimensions         
    ref=sitk.GetImageFromArray(np.zeros(crop_dimensions).swapaxes(0,2))
    ref.SetSpacing(crop_spacing)
    crop_origin=np.array(centroid)+origin_offset
    ref.SetOrigin(crop_origin)
    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1000)
    im=rs.Execute(im)
    x=sitk.GetArrayFromImage(im)
    x=np.expand_dims(np.expand_dims(x,0),-1)
    return x, im

def match_array_to_image(ar,ref):
    #Function to convert numpy array to SITK image based on reference image with same spatial coordinates
    im=sitk.GetImageFromArray(ar)
    im.SetOrigin(ref.GetOrigin())
    im.SetSpacing(ref.GetSpacing())
    im.SetDirection(ref.GetDirection())
    return im




#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #to force cpu




def run(nii_path):
##    f=open('plastimatch_path.txt','r')
##    plm_path=f.readline().replace('\n','')
##    f.close()

    temp_dir='temp'

    start_time=time.time()
##    reader=sitk.ImageSeriesReader()
##
##    for f in os.listdir(os.path.join(temp_dir,'ct')):
##        os.unlink(os.path.join(temp_dir,'ct',f))
##    for f in os.listdir(os.path.join(temp_dir,'plm')):
##        os.unlink(os.path.join(temp_dir,'plm',f))
##    print('Copying Series CT images to temp folder...')
##    series_uid=dicom.read_file(dcm_file_list[0]).SeriesInstanceUID
    print(nii_path)
    im_orig=sitk.ReadImage(nii_path)
    im=resample_orthogonal(im_orig,label=False,reference_spacing=im_orig.GetSpacing(),default_value=-1024)
    
    series_uid=os.path.basename(nii_path).replace('.nii.gz','')
    uid_dir=join(temp_dir,series_uid)
    if not os.path.exists(uid_dir):
        os.mkdir(uid_dir)
##    temp_ct_dir=join(uid_dir,'ct')
##    if not os.path.exists(temp_ct_dir):
##        os.mkdir(temp_ct_dir)
    struct_dir=join(uid_dir,'structs')
    if not os.path.exists(struct_dir):
        os.mkdir(struct_dir)
##    for f in dcm_file_list:
##        shutil.copyfile(f,join(temp_ct_dir,os.path.basename(f)))
##    print('Reading input directory...')
##    reader.SetFileNames(reader.GetGDCMSeriesFileNames(temp_ct_dir))
##    im=reader.Execute()
    #sitk.WriteImage(im,os.path.join(temp_dir,'CT_original.nrrd'))

    x,im_wb=image_to_wb_input(im,(400,400,1500),(96,96,144))
    centroids=[]
    found_regions=[]
    for region in regions:
        region_model_path=join(region_model_dir,model_df[model_df.region==region].region_model.values[0])
        model=load_model(region_model_path,compile=False)
        pred=model.predict(x)
        print(pred.shape)
        print(pred[...,1].max())
        #PROBABILITY CHANGE...
        lab_ar=(pred[0,...,1]>0.5).astype('int16')
        #lab_ar=pred[0,...,1]
        label=match_array_to_image(lab_ar,im_wb)
        ls=sitk.LabelShapeStatisticsImageFilter()
        ls.Execute(label)
        try:
            centroid=np.array(ls.GetCentroid(1))
            found_regions.append(region)
            print(region,'1st Stage Centroid coordinate',centroid)
            centroids.append(centroid)
        except Exception as e:
            print(e, region)
    for i in range(len(found_regions)):
        region=found_regions[i]
        print('segmenting ',region)
        centroid=centroids[i]
        crop_offsets=crop_df[crop_df.region==region].values[0][1:]
        crop_model_path=join(seg_model_dir,model_df[model_df.region==region].seg_model.values[0])
        crop_dimensions=[160,160,144]
        x,im_crop=load_inference_ct(im,centroid,crop_offsets,crop_dimensions)
        model=load_model(crop_model_path,compile=False)
        pred=model.predict(x)
        print('Labelled Voxels: ',(pred[0,...,1]>0.5).sum())
        label_pred=match_array_to_image(pred[0,...,1],im_crop)
##        sitk.WriteImage(label_pred,join(uid_dir,region+'_rawpred.nii.gz'))
        rs=sitk.ResampleImageFilter()
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(0)
        #rs.SetReferenceImage(im)
        rs.SetReferenceImage(im_orig)  #changed to im_orig to account for non-orthogonal cases
##        label=rs.Execute(label)
        label=rs.Execute(label_pred)
        ar=sitk.GetArrayFromImage(label)
        ar=(ar>0.5).astype('int16')
        #PROBABILITY CHANGE..?
##        ar=ar

        blobs,blob_num=ndimage.label(ar) #remove small blobs
        counts=np.array([])
        if blob_num>1:
            for j in range(blob_num):
                counts=np.append(counts,blobs[blobs==(j+1)].sum())
            ar=(blobs==(np.argmax(counts)+1)).astype('int16')

        label=match_array_to_image(ar,label)
        label=sitk.Cast(label,sitk.sitkInt16)
        sitk.WriteImage(label,join(struct_dir,region+'.nii.gz'))
        print(region,' segmented, processing time',time.time()-start)
    car=np.zeros(sitk.GetArrayFromImage(im_orig).shape)
    for f in os.listdir(struct_dir):
        #1=liver
        #2=kidney
        #3=spleen
        #4=pancreas
        if 'liver' in f:
            lar=sitk.GetArrayFromImage(sitk.ReadImage(join(struct_dir,f)))
            car[lar>0]=1
        if 'kidney' in f:
            lar=sitk.GetArrayFromImage(sitk.ReadImage(join(struct_dir,f)))
            car[lar>0]=2
        if 'spleen' in f:
            lar=sitk.GetArrayFromImage(sitk.ReadImage(join(struct_dir,f)))
            car[lar>0]=3
        if 'pancreas' in f:
            lar=sitk.GetArrayFromImage(sitk.ReadImage(join(struct_dir,f)))
            car[lar>0]=4
    label=sitk.Cast(match_array_to_image(car,im_orig),sitk.sitkInt16)
    sitk.WriteImage(label,join(struct_dir,series_uid.replace('_0000','')+'.nii.gz'))
    sitk.WriteImage(label,join(combined_label_dir,series_uid.replace('_0000','')+'.nii.gz'))

##    for f in os.listdir(join(temp_dir,'rt_out')):
##        os.unlink(os.path.join(temp_dir,'rt_out',f))
##
##    print('Writing RT structure file')
##    call=plm_path+' convert --input-prefix '+os.path.join(temp_dir,'plm')+' --output-dicom '+join(temp_dir,'rt_out')+' --referenced-ct '+temp_ct_dir
##    #print(call)
##    #os.popen(call).read()
##    p=subprocess.Popen(call,stdout=subprocess.PIPE)
##    while True:
##        line=p.stdout.readline().decode('utf-8')
##        print(line,end='')
##        if not line: break
##
##    rt_path=os.listdir(join(temp_dir,'rt_out'))
##    dcm=dicom.read_file(os.path.join(temp_dir,'rt_out',rt_path[0]))
##    dcm.SeriesDescription='PMCC AI Organ Contours - development version 3 - May 2020'
##    dcm.save_as(os.path.join(output_dir,rt_path[0]))
##    print('RT Structure written to',os.path.join(output_dir,rt_path[0]))
##    shutil.rmtree(temp_ct_dir)
    #shutil.rmtree(dcm_dir)
    return

##df=gdcm_lister2.nested_dir_to_df(input_dir)

#df=df[df.Modality=='CT']
for f in os.listdir(input_dir):
    print(f)
    run(join(input_dir,f))
    
##for i in range(len(df)):
##    pathlist=df.iloc[i].Filenames
##    run(pathlist,'structure_out')
if delete_input:
    for f in os.listdir(input_dir):
        if os.path.isdir(join(input_dir,f)):
            shutil.rmtree(join(input_dir,f))
        else:
            os.unlink(join(input_dir,f))

print('total processing time:',time.time()-start)
"""
for i in range(len(df)):
    
    run('input','structure_out')
"""
