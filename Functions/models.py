#******************************************************************************
# Functions for creating the required CNN models, as well as related          *
# functions such as training, testing, etc.                                   *
# @author Jorge Ciprián.                                                      *
# Last updated: 11-01-2020.                                                   *
# *****************************************************************************

# Imports.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, LeakyReLU, Flatten, Dense
from Functions.utils import gradient
from Functions.sn import SpectralNormalization

# Function that creates the Generator 1.
def create_g1(spectral_norm):
    # Creating initializer.
    initializer = tf.random_normal_initializer(0., 0.02)
    # The input will be an RGB image.
    # Validating spectral_norm flag:
    if(spectral_norm):
        # Creating model.
        generator1 = keras.Sequential()
        # Input layer.
        #generator1.add(keras.Input(shape=(512, 384, 3), batch_size=32))
        generator1.add(keras.Input(shape=(384, 512, 3)))
        # Encoder.
        # First layer.
        generator1.add(SpectralNormalization(Conv2D(64, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        # Second layer.
        generator1.add(SpectralNormalization(Conv2D(128, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Third layer.
        generator1.add(SpectralNormalization(Conv2D(256, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Fourth layer.
        generator1.add(SpectralNormalization(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Fifth layer.
        generator1.add(SpectralNormalization(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Sixth layer.
        generator1.add(SpectralNormalization(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Seventh layer.
        generator1.add(SpectralNormalization(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True)))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())

        # Decoder.
        # First layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Second layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Third layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Fourth layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(256,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Fifth layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(128,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Sixth layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(64,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True)))
        generator1.add(BatchNormalization())
        # Seventh layer.
        generator1.add(SpectralNormalization(Conv2DTranspose(3,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='tanh', use_bias=True)))
        #generator1.add(BatchNormalization())
    else:
        # Creating model.
        generator1 = keras.Sequential()
        # Input layer.
        #generator1.add(keras.Input(shape=(512, 384, 3), batch_size=32))
        generator1.add(keras.Input(shape=(384, 512, 3)))
        # Encoder.
        # First layer.
        generator1.add(Conv2D(64, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        # Second layer.
        generator1.add(Conv2D(128, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Third layer.
        generator1.add(Conv2D(256, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Fourth layer.
        generator1.add(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Fifth layer.
        generator1.add(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Sixth layer.
        generator1.add(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())
        # Seventh layer.
        generator1.add(Conv2D(512, (4, 4), strides = (2,2), kernel_initializer=initializer, padding='same', use_bias=True))
        generator1.add(LeakyReLU(alpha=0.2))
        generator1.add(BatchNormalization())

        # Decoder.
        # First layer.
        generator1.add(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Second layer.
        generator1.add(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Third layer.
        generator1.add(Conv2DTranspose(512,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Fourth layer.
        generator1.add(Conv2DTranspose(256,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Fifth layer.
        generator1.add(Conv2DTranspose(128,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Sixth layer.
        generator1.add(Conv2DTranspose(64,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='relu', use_bias=True))
        generator1.add(BatchNormalization())
        # Seventh layer.
        generator1.add(Conv2DTranspose(3,(4,4), strides = (2,2), kernel_initializer=initializer, padding='same', activation='tanh', use_bias=True))
        #generator1.add(BatchNormalization())

    # Showing summary of the model.
    generator1.summary()

    # Returning the model.
    return generator1

# Function that creates the Generator 2.
def create_g2(spectral_norm):
    # The input will be a tensor of the RGB and generated IR images.
    # Validating spectral normalization flag.
    if(spectral_norm):
        # Creating model.
        generator2 = keras.Sequential()
        # Input layer.
        #generator2.add(keras.Input(shape=(512, 384, 6), batch_size=32))
        generator2.add(keras.Input(shape=(384, 512, 6)))
        # First layer.
        generator2.add(SpectralNormalization(Conv2DTranspose(256, (5, 5), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Second layer.
        generator2.add(SpectralNormalization(Conv2D(128, (5, 5), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Third layer.
        generator2.add(SpectralNormalization(Conv2DTranspose(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Fourth layer.
        generator2.add(SpectralNormalization(Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Fifth layer.
        generator2.add(SpectralNormalization(Conv2D(3, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='valid', activation='tanh', use_bias=True)))
    else:
        # Creating model.
        generator2 = keras.Sequential()
        # Input layer.
        #generator2.add(keras.Input(shape=(512, 384, 6), batch_size=32))
        generator2.add(keras.Input(shape=(384, 512, 6)))
        # First layer.
        generator2.add(Conv2DTranspose(256, (5, 5), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Second layer.
        generator2.add(Conv2D(128, (5, 5), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Third layer.
        generator2.add(Conv2DTranspose(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Fourth layer.
        generator2.add(Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        generator2.add(BatchNormalization())
        generator2.add(LeakyReLU(alpha=0.2))
        # Fifth layer.
        generator2.add(Conv2D(3, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='valid', activation='tanh', use_bias=True))

    # Showing summary of the model.
    generator2.summary()

    # Returning the model.
    return generator2

# Function that returns the discriminators. Both Discriminator 1 and
# Discriminator 2 have the same architecture.
def create_d(spectral_norm):
    # The input will be a tensor of a 3-channel image.

    # Validating spectral normalization flag.
    if(spectral_norm):
        # Creating model.
        discriminator = keras.Sequential()
        # Input layer.
        #discriminator.add(keras.Input(shape=(512, 384,3), batch_size=32))
        discriminator.add(keras.Input(shape=(384, 512,3)))
        # First layer.
        discriminator.add(SpectralNormalization(Conv2D(32, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        discriminator.add(LeakyReLU(alpha=0.2))
        # Second layer.
        discriminator.add(SpectralNormalization(Conv2D(64, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Third layer.
        discriminator.add(SpectralNormalization(Conv2D(128, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Fourth layer.
        discriminator.add(SpectralNormalization(Conv2D(256, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True)))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Reshaping layer.
        discriminator.add(Flatten())
        # Fully connected layer.
        discriminator.add(Dense(1))
    else:
        # Creating model.
        discriminator = keras.Sequential()
        # Input layer.
        #discriminator.add(keras.Input(shape=(512, 384,3), batch_size=32))
        discriminator.add(keras.Input(shape=(384, 512,3)))
        # First layer.
        discriminator.add(Conv2D(32, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        discriminator.add(LeakyReLU(alpha=0.2))
        # Second layer.
        discriminator.add(Conv2D(64, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Third layer.
        discriminator.add(Conv2D(128, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Fourth layer.
        discriminator.add(Conv2D(256, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='valid', use_bias=True))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(alpha=0.2))
        # Reshaping layer.
        discriminator.add(Flatten())
        # Fully connected layer.
        discriminator.add(Dense(1))

    # Showing summary of the model.
    discriminator.summary()

    # Returning model.
    return discriminator

#---------------------------------GEN1 UNET-------------------------------------
# This section is adapted from the Pix2Pix Tensorflow implementation available
# at: https://www.tensorflow.org/tutorials/generative/pix2pix

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def create_g1_unet():
    #inputs = tf.keras.layers.Input(shape=[256,256,3])
    inputs = tf.keras.layers.Input(shape=[384, 512, 3])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    #downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    #upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    g1_unet = tf.keras.Model(inputs=inputs, outputs=x)
    g1_unet.summary()

    return g1_unet
#---------------------------------GEN1 UNET-------------------------------------


# Function that implements the loss function of the Generator 1.
def loss_g1(d2_res_g1, g1_res, ir_in):
    # Setting lambda parameter.
    lambd = 100
    # Generating binary cross entropy loss object.
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Calculating the binary cross entropy.
    gan_loss = loss_object(tf.ones_like(d2_res_g1), d2_res_g1)
    # Calculating the L1 loss.
    l1_loss = tf.reduce_mean(tf.abs(ir_in - g1_res))
    # Calculating the total loss value.
    total_loss = gan_loss + (lambd*l1_loss)
    # Returning result.
    return total_loss

# Function that implements the loss function for the Generator 2.
def loss_g2(disc1_out_g2, disc2_out_g2, ir_in, rgb_in, out_g2, batch_size):
    xi = 8
    lambd = 100
    gamma = 1
    # In here, [32,1] --> 32 is the batch size.
    g_loss_1 = tf.reduce_mean(tf.square(disc1_out_g2-tf.random.uniform(shape=[batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
    g_loss_2 = tf.reduce_mean(tf.square(disc2_out_g2-tf.random.uniform(shape=[batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
    #g_loss_3 = tf.reduce_mean(tf.square(out_g2 - ir_in))+(xi*tf.reduce_mean(tf.square(gradient(tf.image.rgb_to_grayscale(out_g2)) - gradient(tf.image.rgb_to_grayscale(rgb_in)))))
    g_loss_3 = tf.reduce_mean(tf.square(out_g2 - ir_in))+(xi*tf.reduce_mean(tf.square(gradient(out_g2) - gradient(rgb_in))))
    # Calculating total cost.
    # Calculating total cost.
    total_loss = (gamma*g_loss_1) + g_loss_2 + (lambd*g_loss_3)
    # Returning total result.
    return total_loss

# Function that computes the loss funciton of Discriminator 1.
def loss_d1(rgb_in, out_g2, batch_size):
    # In here, [32,1] --> 32 is the batch size.
    rgb_loss = tf.reduce_mean(tf.square(rgb_in-tf.random.uniform(shape=[batch_size,1],minval=0.7,maxval=1.2)))
    g2_loss = tf.reduce_mean(tf.square(out_g2-tf.random.uniform(shape=[batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
    # Calculating total cost.
    total_loss = rgb_loss + g2_loss
    # Returning total result.
    return total_loss

# Function thtat computes the loss function of Discriminator 2.
def loss_d2(ir_in, out_g2, out_g1, batch_size):
    # In here, [32,1] --> 32 is the batch size.
    ir_loss = tf.reduce_mean(tf.square(ir_in-tf.random.uniform(shape=[batch_size,1],minval=0.7,maxval=1.2)))
    # Originally the same as g2_loss.
    g1_loss = tf.reduce_mean(tf.square(out_g1-tf.random.uniform(shape=[batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
    g2_loss = tf.reduce_mean(tf.square(out_g2-tf.random.uniform(shape=[batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
    # Calculating total cost.
    total_loss = ir_loss + g1_loss + g2_loss
    # Returning total result.
    return total_loss
