#******************************************************************************
# Functions for evaluating a trained model in the validation set.             *
# @author Jorge Ciprián.                                                      *
# Last updated: 15-12-2020.                                                   *
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
def validation_set_eval(val_rgb, val_ir, val_params):
    # Loading general parameters.
    model_path = val_params['MODEL_PATH']
    model_id = val_params['MODEL_ID']
    g1_unet_flag = val_params['GEN1_UNET']
    disp_imgs = val_params['DISP_IMGS']
    # loading spectral normalization parameters.
    g1_spec = val_params['GEN1_SPEC']
    g2_spec = val_params['GEN2_SPEC']
    # Creating individual network's paths.
    gen1_path = model_path + 'GEN1.h5'
    gen2_path = model_path + 'GEN2.h5'

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

    # Iterating on dataset.
    for batch_rgb, batch_ir in zip(val_rgb, val_ir):
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
            if(disp_imgs):
                # Test plots.
                ax0 = plt.subplot(1, 4, 1)
                plt.imshow(rgb_img)
                plt.axis("off")
                ax0.title.set_text('RGB')
                ax1 = plt.subplot(1, 4, 2)
                plt.imshow(ir_img)
                plt.axis("off")
                ax1.title.set_text('IR')
                ax2 = plt.subplot(1, 4, 3)
                plt.imshow(gir_img)
                plt.axis("off")
                ax2.title.set_text('IR gen')
                ax3 = plt.subplot(1, 4, 4)
                plt.imshow(fused_img)
                plt.axis("off")
                ax3.title.set_text('Fused')
                plt.show()
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
            cc_ir_gir = round( stats.pearsonr(tf.image.rgb_to_grayscale(ir_img).numpy().flatten(), tf.image.rgb_to_grayscale(gir_img).numpy().flatten())[0], 4)
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
    path_res = "./Results_val/"+model_id+".mat"
    savemat(path_res, data)
