#******************************************************************************
# Image pre-processing functions.                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 11-01-2020.                                                   *
# *****************************************************************************


# Imports.
import glob
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
from Functions.data_augmentation import *


#--------------------------------DECODING---------------------------------------
# Function for decoding the compressed string into an RGB image. Validates
# .tiff and .png formats, specifically for the three channels of an RGB image.
def decode_img_rgb(img, flag_tiff):
    # Defining rescaling layer.
    # For [0, 1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    if(not flag_tiff):
        # For PNG format.
        img = tf.io.decode_image(img, channels=3)
        # Resizing.
        img = tf.cast(img, tf.float32)
        img = tf.image.resize_with_pad(img,384,512,method='bilinear',antialias=False)
        # Rescaling.
        img = normalization_layer(img)
        # img = ()
    else:
        # For tiff format.
        # Decode the compressed string to a tensor.
        img = tfio.experimental.image.decode_tiff(img, index=0, name=None)
        # Remove alpha channel.
        img = img[:,:,:3]
        # Resizing.
        img = tf.cast(img, tf.float32)
        img = tf.image.resize_with_pad(img,384,512,method='bilinear',antialias=False)
        # Rescaling.
        img = normalization_layer(img)
    return img

# Function for decoding the compressed string into an image. Validates
# .tiff and .png formats, specifically for a three-channel grayscale image.
def decode_img_ir(img, flag_tiff):
    # Defining rescaling layer.
    # For [0, 1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    if(not flag_tiff):
        # For PNG format.
        img = tf.io.decode_image(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize_with_pad(img,384,512,method='bilinear',antialias=False)
        img = normalization_layer(img)
        print("Shape IR: ",tf.shape(img))
    else:
        # For tiff format.
        # Decode the compressed string to a tensor.
        img = tfio.experimental.image.decode_tiff(img, index=0, name=None)
        # Remove alpha channel.
        img = img[:,:,:3]
        # Resizing with padding.
        img = tf.cast(img, tf.float32)
        img = tf.image.resize_with_pad(img,384,512,method='bilinear',antialias=False)
        # Rescale.
        img = normalization_layer(img)
        # We convert it to a single-channel grayscale image.
        #img = tf.image.rgb_to_grayscale(img, name=None)
    return img
#--------------------------------DECODING---------------------------------------


#----------------------------PATH PROCESSING------------------------------------
# Processes the path for a given image. Calls the decode_img_rgb function.
def process_path_rgb(file_path, flag_tiff):
    img = tf.io.read_file(file_path)
    img = decode_img_rgb(img, flag_tiff)
    return img

# Processes the path for a given image. Calls the decode_img_ir function.
def process_path_ir(file_path, flag_tiff):
    img = tf.io.read_file(file_path)
    img = decode_img_ir(img, flag_tiff)
    return img
#----------------------------PATH PROCESSING------------------------------------


#----------------------CONFIGURING FOR PERFORMANCE------------------------------
# Configures the dataset for good performance.
def configure_for_performance_(ds,batch_size,augment=False):
    if(augment):
        ds_mirror = ds.map(a_mirror_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.concatenate(ds_mirror)
        ds_rotate_180 = ds.map(a_rotate_180,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.concatenate(ds_rotate_180)
        ds_rotate_90 = ds.map(a_rotate_90,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.concatenate(ds_rotate_90)
        ds_crop = ds.map(a_central_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.concatenate(ds_crop)
    ds_size = tf.data.experimental.cardinality(ds).numpy()
    ds_size = ds_size*batch_size
    #print("Size of dataset: ", tf.data.experimental.cardinality(ds).numpy())
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=ds_size, seed=1)
    ds = ds.batch(batch_size) # Batch size of 32.
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return ds
#----------------------CONFIGURING FOR PERFORMANCE------------------------------


#-----------------------LOADING DATASET AND PROCESSING IMAGES-------------------
# Function for loading the datasets. It loads both the RGB and IR images,
# shuffles them equally, splits them into training and validation sets if
# required,  and returns the resulting datasets.
def load_datasets(rgb_path, ir_path, v_split, batch_size):
    # Initializing flag that indicates tiff file format.
    flag_tiff = True
    # Getting the total number of images.
    image_count = len(list(glob.glob(rgb_path)))
    print("Number of images: ",image_count)
    # Loading and shuffling the datasets' file name lists.
    # For RGB.
    dataset_rgb = tf.data.Dataset.list_files(rgb_path, shuffle = False)
    dataset_rgb = dataset_rgb.shuffle(image_count, reshuffle_each_iteration=False, seed=1)
    # For IR.
    dataset_ir = tf.data.Dataset.list_files(ir_path, shuffle = False)
    dataset_ir = dataset_ir.shuffle(image_count, reshuffle_each_iteration=False, seed=1)
    print("Finished loading datasets.")

    # for f in dataset_ir.take(5):
    #     print(f.numpy())

    # Identifying tiff file extensions.
    for f in dataset_rgb.take(1):
        if((b'tiff' in f.numpy()) or (b'TIFF' in f.numpy())):
            flag_tiff = True
        else:
            flag_tiff = False
    # If the v_split flag is True, we divide the dataset into test and
    # validation sets. We perform data augmentation on the training dataset
    # only.
    if(v_split):
        # Splitting into train and validation.
        # Getting validation set size.
        validation_size = int(image_count * 0.2)
        # For RGB.
        train_rgb = dataset_rgb.skip(validation_size)
        validation_rgb = dataset_rgb.take(validation_size)
        # For IR.
        train_ir = dataset_ir.skip(validation_size)
        validation_ir = dataset_ir.take(validation_size)
        # Feedback on partition sizes.
        print("Train RGB size: ",tf.data.experimental.cardinality(train_rgb).numpy())
        print("Validation RGB size: ",tf.data.experimental.cardinality(validation_rgb).numpy())
        print("Train IR size: ",tf.data.experimental.cardinality(train_ir).numpy())
        print("Validation IR size: ",tf.data.experimental.cardinality(validation_ir).numpy())
        # Preparing the datasets.
        # RGB.
        # Training set.
        rgb_images_train = train_rgb.map(lambda x: process_path_rgb(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        rgb_images_train = configure_for_performance_(rgb_images_train,batch_size,augment=True)
        print("Prepared train RGB batches: ",tf.data.experimental.cardinality(rgb_images_train).numpy())
        # Validation set.
        rgb_images_val = validation_rgb.map(lambda x: process_path_rgb(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        rgb_images_val = configure_for_performance_(rgb_images_val,batch_size)
        print("Prepared validation RGB batches: ",tf.data.experimental.cardinality(rgb_images_val).numpy())
        # IR.
        # Training set.
        ir_images_train = train_ir.map(lambda x: process_path_ir(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ir_images_train = configure_for_performance_(ir_images_train,batch_size,augment=True)
        print("Prepared train IR batches: ",tf.data.experimental.cardinality(ir_images_train).numpy())
        # Validation set.
        ir_images_val = validation_ir.map(lambda x: process_path_ir(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ir_images_val = configure_for_performance_(ir_images_val,batch_size)
        print("Prepared validation IR batches: ",tf.data.experimental.cardinality(ir_images_val).numpy())
        # Returning the four datasets.
        return rgb_images_train, rgb_images_val, ir_images_train, ir_images_val
    else: # No validation split.
        # Feedback on dataset size.
        print("Dataset RGB size: ",tf.data.experimental.cardinality(dataset_rgb).numpy())
        print("Dataset IR size: ",tf.data.experimental.cardinality(dataset_ir).numpy())
        # Preparing the datasets.
        # RGB.
        rgb_images_full = dataset_rgb.map(lambda x: process_path_rgb(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        rgb_images_full = configure_for_performance_(rgb_images_full,batch_size,augment=False)
        print("Prepared full RGB batches: ",tf.data.experimental.cardinality(rgb_images_full).numpy())
        # IR.
        ir_images_full = dataset_ir.map(lambda x: process_path_ir(x, flag_tiff), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ir_images_full = configure_for_performance_(ir_images_full,batch_size,augment=False)
        print("Prepared full IR batches: ",tf.data.experimental.cardinality(ir_images_full).numpy())
        # Return the two datasets.
        return rgb_images_full, ir_images_full
#-----------------------LOADING DATASET AND PROCESSING IMAGES-------------------
