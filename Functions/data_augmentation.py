#******************************************************************************
# Functions for data augmentation.                                            *
# @author Jorge Ciprián.                                                      *
# Last updated: 23-10-2020.                                                   *
# *****************************************************************************


# Imports.
import tensorflow as tf

# Data augmentation: horizontal flip.
def a_mirror_image(image):
    #image = tf.cast(image, tf.float32)
    image = tf.image.flip_left_right(image)
    image = tf.image.resize_with_pad(image,384,512,method='bilinear',antialias=False)
    return image

# Data augmentation: central crop.
def a_central_crop(image):
    #image = tf.cast(image, tf.float32)
    image = tf.image.central_crop(image,central_fraction=0.5)
    image = tf.image.resize_with_pad(image,384,512,method='bilinear',antialias=False)
    return image

# Data augmentation: rotation 180 degrees.
def a_rotate_180(image):
    #image = tf.cast(image, tf.float32)
    image = tf.image.rot90(image,2)
    image = tf.image.resize_with_pad(image,384,512,method='bilinear',antialias=False)
    return image

# Data augmentation: rotation 90 degrees.
def a_rotate_90(image):
    #image = tf.cast(image, tf.float32)
    image = tf.image.rot90(image,1)
    image = tf.image.resize_with_pad(image,384,512,method='bilinear',antialias=False)
    return image
