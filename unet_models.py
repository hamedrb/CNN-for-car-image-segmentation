#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:28:16 2018

@author: analysisstation3
"""

def unet(input_shape):
    '''
    Params: input_shape -- the shape of the images that are input to the model
                           in the form (width_or_height, width_or_height,
                           num_color_channels)

    Returns: model -- a model that has been defined, but not yet compiled.
                      The model is an implementation of the Unet paper
                      (https://arxiv.org/pdf/1505.04597.pdf) and comes
                      from this repo https://github.com/zhixuhao/unet. It has
                      been modified to keep up with API changes in keras 2.
    
    By: Microsoft [https://notebooks.azure.com/mameehan/libraries/unet/html/unet_pipeline.ipynb]
    '''
    
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     MaxPooling2D,
                                     UpSampling2D,
                                     Dropout,
                                     Concatenate,
                                     Cropping2D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    ## down
    inputs = Input(input_shape)

    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    ## center
    conv5 = Conv2D(1024,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    ## up
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up6)
    drop4_cropped = Cropping2D(cropping=((0, 0), (0, 0)))(drop4)
    merge6 = Concatenate(axis=3)([drop4_cropped, up6])
    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up7)
    conv3_cropped = Cropping2D(cropping=((1, 0), (0, 0)))(conv3)
    merge7 = Concatenate(axis=3)([conv3_cropped, up7])
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up8)
    conv2_cropped = Cropping2D(cropping=((1, 1), (0, 0)))(conv2)
    merge8 = Concatenate(axis=3)([conv2_cropped, up8])
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up9)
    conv1_cropped = Cropping2D(cropping=((3, 2), (0, 0)))(conv1)
    merge9 = Concatenate(axis=3)([conv1_cropped, up9])
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    
    ## classification
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

###############################################################################
###############################################################################

def unet_64_512_4(input_shape):
    '''
    Params: input_shape -- the shape of the images that are input to the model
                           in the form (width_or_height, width_or_height,
                           num_color_channels)

    Returns: model -- a model that has been defined, but not yet compiled.
                      The model is an implementation of the Unet paper
                      (https://arxiv.org/pdf/1505.04597.pdf) and comes
                      from this repo https://github.com/zhixuhao/unet. It has
                      been modified to keep up with API changes in keras 2.
    Note: In the name of the net (here unet_8_3): 
                    the first number is the number of the first feature map (i.e. 64)
                    the second number is the number of the central feature map (i.e. 1024)
                    the third number is the number of down layers (i.e. 4)
    By: Microsoft [https://notebooks.azure.com/mameehan/libraries/unet/html/unet_pipeline.ipynb]
    '''
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     MaxPooling2D,
                                     UpSampling2D,
                                     Dropout,
                                     Concatenate,
                                     Cropping2D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    ## down
    inputs = Input(input_shape)
    
    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    ## center
    conv5 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    ## up
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up6)
    drop4_cropped = Cropping2D(cropping=((0, 0), (0, 0)))(drop4)
    merge6 = Concatenate(axis=3)([drop4_cropped, up6])
    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up7)
    conv3_cropped = Cropping2D(cropping=((1, 0), (0, 0)))(conv3)
    merge7 = Concatenate(axis=3)([conv3_cropped, up7])
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up8)
    conv2_cropped = Cropping2D(cropping=((1, 1), (0, 0)))(conv2)
    merge8 = Concatenate(axis=3)([conv2_cropped, up8])
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(up9)
    conv1_cropped = Cropping2D(cropping=((3, 2), (0, 0)))(conv1)
    merge9 = Concatenate(axis=3)([conv1_cropped, up9])
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    
    ## classification
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model





