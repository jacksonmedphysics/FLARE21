import os
from os.path import join
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

dimg='TrainingImg'
dlab='TrainingMask'
datadir='data'
#1=liver
#2=kidney
#3=spleen
#4=pancreas


def match_array_to_im(ar,ref):
    im=sitk.GetImageFromArray(ar)
    im.SetSpacing(ref.GetSpacing())
    im.SetDirection(ref.GetDirection())
    im.SetOrigin(ref.GetOrigin())
    return im

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

min_volume=20.

for f in os.listdir(dimg):
    if f.endswith('.nii.gz'):
        fl=f.replace('_0000','')
        #print(fl)
        case=fl.replace('.nii.gz','').replace('train_','')
        case='FLARE21_'+case+'.nii.gz'
        print(case)
        ct=sitk.ReadImage(join(dimg,f))
        lab=sitk.ReadImage(join(dlab,fl))
        spacing=ct.GetSpacing()
        voxel_vol=spacing[0]*spacing[1]*spacing[2]/1000.
        ct,lab=resample_orthogonal(ct,lab,spacing,-1000)
        car=sitk.GetArrayFromImage(ct)

        dar=(car+1000)/1000
        dar[dar<0]=0
        xave=np.average(np.average(dar,0),0)
        lin=np.linspace(1,xave.shape[0],xave.shape[0])
        xcom=int((lin*xave).sum()/xave.sum())-1
        

        lar=sitk.GetArrayFromImage(lab)
        kar=(lar==2).astype('uint8')
        lkar=np.zeros(lar.shape)
        rkar=np.zeros(lar.shape)
        livar=(lar==1).astype('uint8') #liver
        splar=(lar==3).astype('uint8') #spleen
        panar=(lar==4).astype('uint8') #pancreas
        
        blobs,n_blobs=ndimage.label(kar)
        for i in range(n_blobs): #loop to remove small non-contiguous voxels
            if (blobs==(i+1)).sum()*voxel_vol>min_volume:
                com=ndimage.center_of_mass((blobs==(i+1)))
                if com[2]<xcom:
                    rkar[(blobs==(i+1))]=1
                else:
                    lkar[(blobs==(i+1))]=1                          
            else:
                print((blobs==(i+1)).sum(),'non-contiguous voxels removed')
        sitk.WriteImage(ct,join(datadir,'ct',case))
        if True:
            sitk.WriteImage(match_array_to_im(livar,lab),join(datadir,'liver',case))
        if True:
            sitk.WriteImage(match_array_to_im(lkar,lab),join(datadir,'lt_kidney',case))
        if True:
            sitk.WriteImage(match_array_to_im(rkar,lab),join(datadir,'rt_kidney',case))
        if True: 
            sitk.WriteImage(match_array_to_im(splar,lab),join(datadir,'spleen',case))
        if True:
            sitk.WriteImage(match_array_to_im(panar,lab),join(datadir,'pancreas',case))
        

