#******************************************************************************
# Functions for testing the full DL model for image fusion.                   *
#                                                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 14-12-2020.                                                   *
# *****************************************************************************

# Imports.
import time
import numpy as np
from scipy import stats
import tensorflow as tf
from scipy.io import savemat
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
# Importing custom functions.
from Functions.models import *

# Function for evaluating the model on the validation set. Receives the training
# dataset for IR and RGB images, as well as pertinent parameters.
def test_set_eval(test_rgb, test_ir, test_params):
    # Loading general parameters.
    model_path = test_params['MODEL_PATH']
    model_id = test_params['MODEL_ID']
    g1_unet_flag = test_params['GEN1_UNET']
    # loading spectral normalization parameters.
    g1_spec = test_params['GEN1_SPEC']
    g2_spec = test_params['GEN2_SPEC']
    # Creating individual network's paths.
    gen1_path = model_path + 'GEN1.h5'
    gen2_path = model_path + 'GEN2.h5'
    disc1_path = model_path + 'DISC1.h5'
    disc2_path = model_path + 'DISC2.h5'
    # Creating models.
    if(g1_unet_flag):
        gen_1 = create_g1_unet()
    else:
        gen_1 = create_g1(g1_spec)
    gen_2 = create_g2(g2_spec)
    # Loading model weights.
    gen_1.load_weights(gen1_path)
    gen_2.load_weights(gen2_path)

    # Initializing metric lists.
    data = {}
    # Image entropy.
    data['en_rgb'] = []
    data['en_ir'] = []
    data['en_gir'] = [] # Generated infrared image.
    data['en_fused'] = []
    # Correlation coefficient.
    data['cc_rgb_fused'] = []
    data['cc_ir_fused'] = []
    data['cc_ir_gir'] = []
    # PSNR.
    data['psnr_rgb_fused'] = []
    data['psnr_ir_fused'] = []
    data['psnr_ir_gir'] = []
    # SSIM.
    data['ssim_rgb_fused'] = []
    data['ssim_ir_fused'] = []
    data['ssim_ir_gir'] = []

    # Average time per image.
    avg_time = 0

    # Batch counter.
    batch_count = 0

    # Image counter.
    img_count = 0

    # Iterating on dataset.
    for batch_rgb, batch_ir in zip(test_rgb, test_ir):

        print("Batch: ", batch_count)
        # Start measuring time per batch.
        start_time = time.time()
        # Getting output from generator 1.
        gen1_out = gen_1(batch_rgb)
        # Create input for generator 2.
        in_gen2 = tf.concat([batch_rgb, gen1_out], 3)
        # Get output from generator 2.
        gen2_out = gen_2(in_gen2)
        # Calculating time per batch and displaying it.
        elapsed_time = time.time() - start_time
        print("Elapsed time per image: ", elapsed_time/len(batch_rgb))
        # Adding to the avg_time counter.
        avg_time = avg_time + (elapsed_time/len(batch_rgb))
        # Calculate PSNR for full batch.
        # Between IR and generated IR.
        psnr_ir_gir = tf.image.psnr(batch_ir, gen1_out, max_val=1.0)
        # Between fused and RGB.
        psnr_rgb_fused = tf.image.psnr(batch_rgb, gen2_out, max_val=1.0)
        # Between fused and IR.
        psnr_ir_fused = tf.image.psnr(batch_ir, gen2_out, max_val=1.0)
        # Calculate SSIM for full batch.
        # Between IR and generated IR.
        ssim_ir_gir = tf.image.ssim(batch_ir, gen1_out, max_val=1.0)
        # Between fused and RGB.
        ssim_rgb_fused = tf.image.ssim(batch_rgb, gen2_out, max_val=1.0)
        # Between fused and IR.
        ssim_ir_fused = tf.image.ssim(batch_ir, gen2_out, max_val=1.0)
        # Calculating metrics per batch.
        for i in range(len(batch_rgb)):
            # Get specific images.
            rgb_img = batch_rgb[i].numpy()
            ir_img = batch_ir[i].numpy()
            gir_img = gen1_out[i].numpy()
            fused_img = gen2_out[i].numpy()
            # Saving images.
            # Generating paths.
            sample_img_rgb_path = "./Results_test/RGB/" + str(img_count) + "rgb.png"
            sample_img_ir_path = "./Results_test/IR/" + str(img_count) + "ir.png"
            sample_img_gir_path = "./Results_test/GIR/" + str(img_count) + "gir.png"
            sample_img_fused_path = "./Results_test/FUSED/" + str(img_count) + "fused.png"
            # Saving sample images.
            tf.keras.preprocessing.image.save_img(sample_img_rgb_path, rgb_img, data_format="channels_last")
            tf.keras.preprocessing.image.save_img(sample_img_ir_path, ir_img, data_format="channels_last")
            tf.keras.preprocessing.image.save_img(sample_img_gir_path, gir_img, data_format="channels_last")
            tf.keras.preprocessing.image.save_img(sample_img_fused_path, fused_img, data_format="channels_last")
            # Updating image counter.
            img_count += 1

            # Calculating metrics.
            # Image Entropy.
            # For RGB image.
            en_rgb = round(shannon_entropy(rgb_img, 4))
            data['en_rgb'].append(en_rgb)
            # For IR image.
            en_ir = round(shannon_entropy(ir_img, 4))
            data['en_ir'].append(en_ir)
            # For generated IR image.
            en_gir = round(shannon_entropy(gir_img, 4))
            data['en_gir'].append(en_gir)
            # For fused image.
            en_fused = round(shannon_entropy(fused_img, 4))
            data['en_fused'].append(en_fused)
            # Correlation Coefficient.
            # Between IR and generated IR. stats.pearsonr(a, b)
            cc_ir_gir = round(stats.pearsonr(tf.image.rgb_to_grayscale(ir_img).numpy().flatten(), tf.image.rgb_to_grayscale(gir_img).numpy().flatten())[0], 4)
            data['cc_ir_gir'].append(cc_ir_gir)
            # Between RGB and fused.
            cc_rgb_fused = round( stats.pearsonr(tf.image.rgb_to_grayscale(rgb_img).numpy().flatten(), tf.image.rgb_to_grayscale(fused_img).numpy().flatten())[0], 4)
            data['cc_rgb_fused'].append(cc_rgb_fused)
            # Between IR and fused.
            cc_ir_fused = round( stats.pearsonr(tf.image.rgb_to_grayscale(ir_img).numpy().flatten(), tf.image.rgb_to_grayscale(fused_img).numpy().flatten())[0], 4)
            data['cc_ir_fused'].append(cc_ir_fused)
            # PSNR.
            # Between IR and generated IR.
            data['psnr_ir_gir'].append(psnr_ir_gir[i].numpy())
            # Between RGB and fused.
            data['psnr_rgb_fused'].append(psnr_rgb_fused[i].numpy())
            # Between IR and fused.
            data['psnr_ir_fused'].append(psnr_ir_fused[i].numpy())
            # SSIM.
            # Between IR and generated IR.
            data['ssim_ir_gir'].append(ssim_ir_gir[i].numpy())
            # Between RGB and fused.
            data['ssim_rgb_fused'].append(ssim_rgb_fused[i].numpy())
            # Between IR and fused.
            data['ssim_ir_fused'].append(ssim_ir_fused[i].numpy())
        # Incrementing batch counter.
        batch_count += 1
    # Calculating avg time per image.
    avg_time = avg_time/batch_count
    data['avg_time'] = avg_time
    # Save to mat file.
    path_res = "./Metrics_test/"+model_id+".mat"
    savemat(path_res, data)
