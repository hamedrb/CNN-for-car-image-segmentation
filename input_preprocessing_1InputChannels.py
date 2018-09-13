#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:51:20 2018

@author: analysisstation3
"""

import os 
from glob import glob
import numpy as np # linear algebra
import matplotlib.pylab as plt
import cv2
from PIL import Image # PIL == Python Image Library
from tensorflow.keras import backend as K
from tqdm import tqdm
import pickle
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
from unet_models import unet, unet_4_3
###############################################################################

### Data constants

INPUT_PATH = '/home/analysisstation3/projects/CNNForCarSegmentation/Input/kaggle_Carvana_Image_Masking_Challenge/all/'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_masks")
TEST_DATA = os.path.join(DATA_PATH, "test")

OUTPUT_PATH = '/home/analysisstation3/projects/CNNForCarSegmentation/output'
DATA_OUTPUT_PATH = OUTPUT_PATH

RESIZED_HEIGHT = 1564
RESIZED_WIDTH = 1995
PERCENT_OF_EVAL = 10

### lists of train/test files
train_files = glob(os.path.join(TRAIN_DATA, "*.jpg"))
train_ids = [s[len(TRAIN_DATA)+1:-4] for s in train_files]

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]
###############################################################################

### load and preprocess images and masks

def get_input_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        ext = 'jpg'
        data_path = TRAIN_DATA
        suffix = ''
    elif "Train_mask" in image_type:
        ext = 'gif'
        data_path = TRAIN_MASKS_DATA
        suffix = '_mask'
    elif "Test" in image_type:
        ext = 'jpg'
        data_path = TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))



def get_preprocessedData_filename(image_type, resized_height, resized_width):
    ext = 'npy'
    data_path = DATA_OUTPUT_PATH
    common_name = 'data_preprocessed'
    
    if image_type == 'Train':
        suffix = 'train'
    elif image_type == 'Train_mask':
        suffix = 'train_mask'
    elif image_type == 'Test':
        suffix = 'test'
    
    return os.path.join(data_path, "{}_{}_h{}_w{}.{}".format(suffix, \
                            common_name, resized_height, resized_width, ext))
        
        

def get_image_data(image_id, image_type, **kwargs):
    if 'mask' in image_type:
        img = _get_image_data_pil(image_id, image_type, **kwargs)
    else:
        img = _get_image_data_opencv(image_id, image_type, **kwargs)
    return img

def resize_image_while_maintaining_aspect_ratio(image, resized_width):
    '''
    resize image while maintaining aspect ratio
    
    Params: image -- input image to be resized, a numpy array
            resized_width: new width of the image, an integer
            image_type e.g. "Train", "Train_mask" or "Test"
            
    Returns: image -- the resized image, a numpy array
    
    By: Hamed
    '''
    image_height = float(image.shape[0])
    image_width = float(image.shape[1])
    width_ratio = resized_width / image_width
    resized_height = int(image_height * width_ratio)
    image = cv2.resize(image, (resized_width, resized_height))
    return image


def hard_resize(image, output_shape):
      image = cv2.resize(image, output_shape)
      return image
    
    
def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_input_filename(image_id, image_type)
    img = cv2.imread(fname)
    img = resize_image_while_maintaining_aspect_ratio(img, RESIZED_WIDTH)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Changing Color-space to gray scale
    return img

def _get_image_data_pil(image_id, image_type):
    fname = get_input_filename(image_id, image_type)
    img_pil = Image.open(fname)
    img = np.asarray(img_pil)
    img = hard_resize(img, output_shape=(1992, 1560))
    return img


def image_shape_dtype(image_type):
    '''
        Params: image_type e.g. "Train", "Train_mask" or "Test"
        
        Returns: 
            img.shape -- size of image
            img.dtype -- digital type of image array. Usually uint8
        
        By: Hamed
    '''
    # get a sample of images
    if (image_type == "Train") or (image_type == "Train_mask"):
        image_id = train_ids[0] 
    elif image_type == "Test":
        image_id = test_ids[0]
    
    img = get_image_data(image_id, image_type)
    return img.shape, img.dtype

def load_images(image_type):
    '''
        loads images and normalize them
        
        Params: image_type e.g. "Train", "Train_mask" or "Test"
        
        Returns: 
            img_array -- an array which includes all images of image_type
        
        By: Hamed
    '''
    
    img_shape, img_dtype = image_shape_dtype(image_type)
    
    if image_type == "Train":
        img_array = np.zeros(shape=(len(train_ids), img_shape[0], img_shape[1]), dtype='float32')
        counter = 0
        for image_id in tqdm(train_ids):
            img_array[counter] = get_image_data(image_id, image_type)/255.
            counter+=1
            
    elif image_type == "Test":
        img_array = np.zeros(shape=(len(test_ids), img_shape[0], img_shape[1]), dtype='float32')
        counter = 0
        for image_id in tqdm(test_ids):
            img_array[counter] = get_image_data(image_id, image_type)/255.
            counter+=1
            
    elif image_type == "Train_mask":
        img_array = np.zeros(shape=(len(train_ids), img_shape[0], img_shape[1]), dtype=img_dtype)
        counter = 0
        for image_id in tqdm(train_ids):
            img_array[counter] = get_image_data(image_id, image_type)
            counter+=1
    return img_array

def split_train_eval(images, masks, percent_of_eval):
    '''
    From input images, it splits a part of it for evaluation of the model and then reshape them to be consistent with Keras input layer
    
    Params: images -- a numpy array of input images. it should have the shape of (number of images, image height, image width, number of channels)
            masks -- corresponding masks of input images. It is a numpy array of shape (number of images, image height, image width)
            percent_of_eval -- percentage of input data to split
            
    Returns: images -- a numpy array of input images that will be used for training
             masks -- corresponding masks of input images that will be  used for training
             images_eval -- a numpy array of input images that will be used for evaluation
             masks_eval -- corresponding masks of input images that will be  used for evaluation
    
    By: Hamed
    '''
    ## splitting 
    number_of_eval = int(images.shape[0]/percent_of_eval) # number of images to be splitted for evaluation
    slice_eval = np.s_[-number_of_eval::1] # a part of input images and masks arrays  to be splitted for evaluation
    images_eval = images[slice_eval]
    masks_eval = masks[slice_eval]
    
    slice_train = np.s_[:-number_of_eval:1]
    images = images[slice_train]
    masks = masks[slice_train]
    
    ## reshaping
    images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))
    masks = np.reshape(masks, (masks.shape[0], masks.shape[1], masks.shape[2], 1))
    images_eval = np.reshape(images_eval, (images_eval.shape[0], images_eval.shape[1], images_eval.shape[2], 1))
    masks_eval = np.reshape(masks_eval, (masks_eval.shape[0], masks_eval.shape[1], masks_eval.shape[2], 1))
    
    return images, masks, images_eval, masks_eval
    

image_shape  = image_shape_dtype('Train')[0]
resized_height = image_shape[0]
TRAIN_OUTPUT_DATA_FPATH = get_preprocessedData_filename('Train',\
                                            resized_height, RESIZED_WIDTH)
if os.path.exists(TRAIN_OUTPUT_DATA_FPATH):
    images = np.load(TRAIN_OUTPUT_DATA_FPATH)
    print('Preprocessed data loaded!')
else:
    images = load_images('Train')
    np.save(TRAIN_OUTPUT_DATA_FPATH, images)
    print('Data preprocessed and saved!')


image_shape = image_shape_dtype('Train_mask')
resized_height = image_shape[0]
resized_width = image_shape[1]
TRAIN_OUTPUT_DATA_MASK_FPATH = get_preprocessedData_filename('Train_mask',\
                                                resized_height, resized_width)
if os.path.exists(TRAIN_OUTPUT_DATA_MASK_FPATH):
    masks = np.load(TRAIN_OUTPUT_DATA_MASK_FPATH)
    print('Preprocessed data loaded!')
else:
    masks = load_images('Train_mask')
    np.save(TRAIN_OUTPUT_DATA_MASK_FPATH, masks)
    print('Data preprocessed and saved!')
        
img_train, mask_train, img_eval, mask_eval = split_train_eval(images, masks, PERCENT_OF_EVAL)
del images, masks, test_files, test_ids, train_files, train_ids

###############################################################################

### train helper functions
def dice_coef(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: dice_coeff -- A metric that accounts for precision and recall
                           on the scale from 0 - 1. The closer to 1, the
                           better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    
    By: Microsoft [https://notebooks.azure.com/mameehan/libraries/unet/html/unet_helpers.py]
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)


def dice_coef_loss(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: 1 - dice_coeff -- a negation of the dice coefficient on
                               the scale from 0 - 1. The closer to 0, the
                               better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
                            
    By: Microsoft [https://notebooks.azure.com/mameehan/libraries/unet/html/unet_helpers.py]
    '''
    return 1-dice_coef(y_true, y_pred)
 


### train constants
MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH = os.path.join(OUTPUT_PATH, 'val_loss_min_unet.hdf5')
FINAL_MODEL_ABSOLUTE_FPATH = os.path.join(OUTPUT_PATH, 'unet.hdf5')
HISTORY_ABSOLUTE_FPATH = os.path.join(OUTPUT_PATH, 'history.pickle')
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-2
EARLY_STOP_VAL_PATIENCE = 25


### train
def train(val_loss_min_fpath,
          final_model_output_fpath,
          history_output_fpath,
          batch_size,
          num_epochs,
          learning_rate,
          early_stop_val_patience):
    '''Trains the unet model, saving the model+weights that performs the
       best on the validation set after each epoch (if there is improvement)
       as well as the final model+weights at the last epoch,
       regardless of performance. History is pickeled once training is
       finished.

    Params: batch_size -- the number of images to be processed together
                          in one step through the model
            num_epochs -- the number of times the entire data set is passed
                          through the model
                          (1 epoch=(num steps through the model)*(batch_size))
            learning_rate -- a scaler value that controls how much the
                             weights get updated in the descent
            early_stop_val_patience -- number of epochs to stop training after
                                       if no improvement on the validation
                                       dataset occurs
            val_loss_min_fpath -- the path (including file name) where the
                                  model that performed the best on the
                                  validation set should be saved
            final_model_output_fpath -- the path (including file name) where
                                        the final trained model is saved
            history_output_fpath -- the path (including file name) where
                                    the history's dictionary is saved
            

    Returns (tuple): history -- keys and values that are useful for analyzing
                                (and plotting) data collected during training
                     model -- the trained model. Note: this model likely
                              overfits to the training data. To use the most
                              performant model on the validation set,
                              refer to the model saved to 'val_loss_min_fpath'
    
    By: Microsofy[https://notebooks.azure.com/mameehan/libraries/unet/html/unet_pipeline.ipynb]
        modified by Hamed
    '''
    input_shape = img_train.shape[1:] #he shape of one image in the dataset, for instance (341, 512, 3)
        
    model = unet_4_3(input_shape)
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=dice_coef_loss,
                  metrics=[dice_coef])
    
    val_loss_checkpoint = ModelCheckpoint(val_loss_min_fpath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='min')
    
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=early_stop_val_patience)
    
    
    history = model.fit(img_train,
                        mask_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(img_eval, mask_eval),
                        callbacks=[early_stop, val_loss_checkpoint])
    
    model.save(final_model_output_fpath)
    
    with open(history_output_fpath, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    
    return history, model

###############################################################################

### Model creation (grabbed from https://notebooks.azure.com/mameehan/libraries/unet/html/unet_pipeline.ipynb)

history, model = train(MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH,
                       FINAL_MODEL_ABSOLUTE_FPATH,
                       HISTORY_ABSOLUTE_FPATH,
                       BATCH_SIZE,
                       NUM_EPOCHS,
                       LEARNING_RATE,
                       EARLY_STOP_VAL_PATIENCE)




###############################################################################
### Misc.
val_loss_min_fpath = MODEL_WITH_MINIMUM_LOSS_ABSOLUTE_FPATH
final_model_output_fpath = FINAL_MODEL_ABSOLUTE_FPATH
history_output_fpath = HISTORY_ABSOLUTE_FPATH
batch_size = BATCH_SIZE
num_epochs = NUM_EPOCHS
learning_rate = LEARNING_RATE
early_stop_val_patience = EARLY_STOP_VAL_PATIENCE
#


#### An example: display a single car with its mask
image_id = train_ids[0]
img = get_image_data(image_id, "Train")
mask = get_image_data(image_id, "Train_mask")
img_masked = cv2.bitwise_and(img, img, mask=mask)

print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape, mask.dtype) )

plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(mask)
plt.subplot(133)
plt.imshow(img_masked)

del img, mask, img_masked, image_id





