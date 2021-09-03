from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate #as concatenate
from tensorflow.keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import Reshape, Activation

from tensorflow.keras.layers import BatchNormalization



filter_depths=[12,18,28,54,96,36,2]

def double_conv_block_down(input_layer,filter_depth1,filter_depth2,k_size=3):
    layer=Convolution3D(padding='same',filters=filter_depth1,kernel_size=k_size)(input_layer)
    layer=BatchNormalization()(layer)
    layer=Activation('relu')(layer)
    layer=Convolution3D(padding='same',filters=filter_depth2,kernel_size=k_size)(layer)
    layer=BatchNormalization()(layer)
    layer=Activation('relu')(layer)
    pool=MaxPooling3D(pool_size=(2,2,2))(layer)
    return pool,layer

def upsample_merge_double_conv(upsample_layer,skip_layer,filter_depth1,filter_depth2,k_size=3):
    merge_axis = -1
    up = UpSampling3D(size=(2, 2, 2))(upsample_layer)
    merged= concatenate([up, skip_layer], axis=merge_axis)
    layer=Convolution3D(padding='same',filters=filter_depth1,kernel_size=k_size)(merged)
    layer=BatchNormalization()(layer)
    layer=Activation('relu')(layer)
    layer=Convolution3D(padding='same',filters=filter_depth2,kernel_size=k_size)(layer)
    layer=BatchNormalization()(layer)
    layer=Activation('relu')(layer)
    return layer

def build_model(inp_shape,k_size=3):
    data=Input(shape=inp_shape)
    pool1,skip1=double_conv_block_down(data,filter_depths[0],filter_depths[1],k_size)
    pool2,skip2=double_conv_block_down(pool1,filter_depths[1],filter_depths[2],k_size)
    pool3,skip3=double_conv_block_down(pool2,filter_depths[2],filter_depths[3],k_size)

    bottom_u=Convolution3D(padding='same',filters=filter_depths[4],kernel_size=k_size)(pool3)
    bottom_u=BatchNormalization()(bottom_u)
    bottom_u=Activation('relu')(bottom_u)
    bottom_u=Convolution3D(padding='same',filters=filter_depths[4],kernel_size=k_size)(bottom_u)
    bottom_u=BatchNormalization()(bottom_u)
    bottom_u=Activation('relu')(bottom_u)
    
    up2=upsample_merge_double_conv(bottom_u,skip3,filter_depths[3],filter_depths[3],k_size=3)
    up3=upsample_merge_double_conv(up2,skip2,filter_depths[2],filter_depths[2],k_size=3)
    up4=upsample_merge_double_conv(up3,skip1,filter_depths[1],filter_depths[1],k_size=3)
    logit = Convolution3D(padding='same', filters=filter_depths[6], kernel_size=k_size)(up4)
    logit = Reshape([-1, filter_depths[6]])(logit)
    logit = Activation('softmax')(logit)
    logit = Reshape(inp_shape[:-1] + (filter_depths[6],))(logit)

    model = Model(data, logit)
    return model    
    
    
