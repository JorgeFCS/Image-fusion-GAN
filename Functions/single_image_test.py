################################################################################
# Simple demo to test the model on a single image pair.                        #
#                                                                              #
# @ author Jorge Francisco Ciprian Sanchez                                     #
# Last updated: 06-01-2019                                                     #
################################################################################

# Imports.
import tensorflow as tf
from tensorflow.keras import layers

# Importing custom functions.
from Functions.models import *

# Function that receives a single visible image, resizes it, gets the artificial
# IR image and the fused image, and saves the results. While the model only
# needs a source visible image, I recommend that you also have a source IR
# image for comparison.

def single_image_test(params):
    # Unpacking parameters.
    rgb_img_path = params['IMAGE_PATH_RGB']
    save_path = params['SAVE_PATH']
    model_path = params['MODEL_PATH']
    gen1_path = model_path + "GEN1.h5"
    gen2_path = model_path + "GEN2.h5"
    g1_spec = params['GEN1_SPECTRAL']
    g2_spec = params['GEN2_SPECTRAL']
    g1_unet_flag = params['GEN1_UNET']
    # Normalization layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # Loading images.
    print("Loading images...")
    rgb_image = tf.keras.preprocessing.image.load_img(rgb_img_path)
    rgb_image = tf.keras.preprocessing.image.img_to_array(rgb_image)
    rgb_image = tf.convert_to_tensor(rgb_image)
    rgb_image = tf.cast(rgb_image, tf.float32)
    rgb_image_res = tf.image.resize_with_pad(rgb_image,384,512,method='bilinear',antialias=False)
    rgb_image_res = normalization_layer(rgb_image_res)
    print("Shape no batch: ", rgb_image.numpy().shape)
    rgb_image_res = tf.expand_dims(rgb_image_res, axis=0)
    print("Shape batch: ", rgb_image_res.numpy().shape)

    # Resizing images with padding.
    print("... done.")
    # Loading model.
    print("Loading model...")
    if(g1_unet_flag):
        gen_1 = create_g1_unet()
    else:
        gen_1 = create_g1(g1_spec)
    gen_2 = create_g2(g2_spec)
    # Loading model weights.
    gen_1.load_weights(gen1_path)
    gen_2.load_weights(gen2_path)
    print("... done.")
    # Generate artificial IR image.
    print("Generating artificial IR image...")
    gen1_out = gen_1(rgb_image_res)
    print("... done.")
    # Generate fused image.
    print("Generating fused image...")
    # Create input for generator 2.
    in_gen2 = tf.concat([rgb_image_res, gen1_out], 3)
    # Get output from generator 2.
    gen2_out = gen_2(in_gen2)
    print("... done.")
    # Save results.
    print("Saving images...")
    rgb_save_path = save_path + "resized_rgb.png"
    gir_save_path = save_path + "gir.png"
    fused_save_path = save_path + "fused.png"
    tf.keras.preprocessing.image.save_img(rgb_save_path, rgb_image_res[0].numpy(), data_format="channels_last")
    tf.keras.preprocessing.image.save_img(gir_save_path, gen1_out[0].numpy(), data_format="channels_last")
    tf.keras.preprocessing.image.save_img(fused_save_path, gen2_out[0].numpy(), data_format="channels_last")
    print("... done.")
