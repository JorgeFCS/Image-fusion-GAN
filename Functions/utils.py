#******************************************************************************
# Utility functions.                                                          *
# @author Jorge Ciprián.                                                      *
# Last updated: 29-10-2020.                                                   *
# *****************************************************************************

# Imports.
import tensorflow as tf

#-------------------------------GRADIENT----------------------------------------
# Function to compute the gradient of an image. Based on the FusionGAN
# implementation by Ma et al.
def gradient(img):
    # Creating filter.
    filter = tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.],[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.],[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[3,3,3,1])
    # Applying convolution to image with filter.
    grad = tf.nn.conv2d(img,filter,strides=[1,1,1,1], padding='SAME')
    return grad
#-------------------------------GRADIENT----------------------------------------
