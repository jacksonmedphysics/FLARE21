import SimpleITK as sitk
import numpy as np
from numpy.random import rand
#from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.ndimage.measurements import center_of_mass

import os
from os.path import join


def match_array_to_image(ar,ref):
    im=sitk.GetImageFromArray(ar)
    im.SetOrigin(ref.GetOrigin())
    im.SetSpacing(ref.GetSpacing())
    im.SetDirection(ref.GetDirection())
    return im


def load_cropped(image_path,label_path,crop_offsets,crop_dimensions,augment=True,
                 max_rotation_deg=20.,max_translation=30.,max_gauss_sigma=1.0,
                 max_hu_shift=30,max_noise=100,sharpening_range=0.6,sharpening_alpha=0.5):
    #crop_offsets should be of the form (-x, +x, -y, +y, -z, +z)
    crop_extent=np.array([crop_offsets[1]-crop_offsets[0],crop_offsets[3]-crop_offsets[2],crop_offsets[5]-crop_offsets[4]])
    origin_offset=np.array([crop_offsets[0],crop_offsets[2],crop_offsets[4]])
    im=sitk.ReadImage(image_path)
    label=sitk.ReadImage(label_path)
    crop_dimensions=np.array(crop_dimensions)
    crop_spacing=crop_extent/crop_dimensions                 
    rs=sitk.ResampleImageFilter()
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetReferenceImage(im)
    label=sitk.Cast(rs.Execute(label),sitk.sitkInt16)
    ls=sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(label)
    centroid=np.array(ls.GetCentroid(1))
    crop_origin=centroid+origin_offset
    ref=sitk.GetImageFromArray(np.zeros(crop_dimensions).swapaxes(0,2))
    ref.SetOrigin(crop_origin)
    ref.SetSpacing(crop_spacing)
    if augment:
        sig=np.random.rand()*max_gauss_sigma
        sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
        salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
        hu_shift=(np.random.rand()-0.5)*max_hu_shift
        sig=np.random.rand()*max_gauss_sigma
        sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
        salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
        hu_shift=(np.random.rand()-0.5)*max_hu_shift
        lab_ar=sitk.GetArrayFromImage(label)
        #com=center_of_mass(lab_ar)
        rotx=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        roty=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        rotz=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        tx=(np.random.rand()-0.5)*max_translation
        ty=(np.random.rand()-0.5)*max_translation
        tz=(np.random.rand()-0.5)*max_translation
        initial_transform=sitk.Euler3DTransform()
        #initial_transform.SetCenter([com[0],com[1],com[2]])
        initial_transform.SetCenter([centroid[0],centroid[1],centroid[2]])
        registration_method = sitk.ImageRegistrationMethod()
        initial_transform.SetParameters((rotx,roty,rotz,tx,ty,tz))
        im = sitk.Resample(im, ref, initial_transform, sitk.sitkLinear, -1000., sitk.sitkInt16)
        label=sitk.Resample(label,ref,initial_transform,sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)    
        ar=sitk.GetArrayFromImage(im)
        lab_ar=sitk.GetArrayFromImage(label)
        blurred_ar=gaussian_filter(ar,sharp)
        sharpened=ar+salpha*(ar-blurred_ar)
        ar=sharpened
        ar=gaussian_filter(ar,sigma=sig)
        ar+=int(hu_shift)
        ar+=((np.random.random(ar.shape)-0.5)*max_noise).astype('int16')
    else:
        rs.SetReferenceImage(ref)
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1000)
        im=rs.Execute(im)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        label=rs.Execute(label)
        ar=sitk.GetArrayFromImage(im)
        lab_ar=sitk.GetArrayFromImage(label)
    x=ar        
    x=np.expand_dims(np.expand_dims(x,0),-1)
    y=lab_ar
    y=np.expand_dims(y,-1)
    y=np.append((y!=1),y,axis=-1)
    y=np.expand_dims(y,0)
    return x,y

def load_inference_ct(image_path,centroid,crop_offsets,crop_dimensions):
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
    return x




