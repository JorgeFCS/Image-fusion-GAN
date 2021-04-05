#******************************************************************************
# Functions for training the full DL model for image fusion.                  *
#                                                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 11-01-2020.                                                   *
# *****************************************************************************

# Imports.
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
# Importing custom functions.
from Functions.models import *

# Function for training the model. Receives the training dataset for IR and
# RGB images, as well as pertinent parameters.
def train(rgb_train, ir_train, train_params):
    # save_model, epochs, check_path, batch_size, learn_rates, spec_norm, sample_img_dir
    # Loading general training parameters.
    save_model = train_params['SAVE']
    epochs = train_params['EPOCHS']
    check_path = train_params['CHECK_PATH']
    batch_size = train_params['BATCH']
    sample_img_dir = train_params['SAMPLE_IMG_DIR']
    # Loading learning rates.
    g1_lr = train_params['GEN1_LR']
    g2_lr = train_params['GEN2_LR']
    d1_lr = train_params['DISC1_LR']
    d2_lr = train_params['DISC2_LR']
    # Loading spectral normalization flags.
    g1_spec = train_params['GEN1_SPEC']
    g2_spec = train_params['GEN2_SPEC']
    d1_spec = train_params['DISC1_SPEC']
    d2_spec = train_params['DISC2_SPEC']
    # Loading U-Net flag.
    g1_unet_flag = train_params['GEN1_UNET']

    sample_img_path = ""
    if(save_model):
        # Getting timestamp string that will work as id for this model.
        dateTimeObj = datetime.now()
        timestamp_str = dateTimeObj.strftime("%d-%b-%Y_(%H_%M_%S.%f)")
        sample_img_path = sample_img_dir + timestamp_str + "/SamplesTrain/"
        print("img path: ", sample_img_path)
        os.makedirs(sample_img_path)

    # Creating models.
    print("Generator 1 summary:")
    if(g1_unet_flag):
        gen_1 = create_g1_unet()
    else:
        gen_1 = create_g1(g1_spec)
    print("Generator 2 summary:")
    gen_2 = create_g2(g2_spec)
    print("Discriminator 1 summary:")
    disc_1 = create_d(d1_spec)
    print("Discriminator 2 summary:")
    disc_2 = create_d(d2_spec)
    # Creating an Adam optimizer.
    # We use Two Time-Scale Update Rule:
    g1_optimizer = tf.keras.optimizers.Adam(learning_rate=g1_lr)
    g2_optimizer = tf.keras.optimizers.Adam(learning_rate=g2_lr)
    d1_optimizer = tf.keras.optimizers.Adam(learning_rate=d1_lr)
    d2_optimizer = tf.keras.optimizers.Adam(learning_rate=d2_lr)

    # Initializing training.
    for epoch in range(epochs):
        print("\nStart of epoch %d:" % (epoch,))
        # Save flag.
        save_flag = True
        # Iterating over the batches of the datasets.
        # Initializing batch counter.
        batch_count = 0
        # Start measuring time per epoch.
        start_time = time.time()
        for batch_rgb, batch_ir in zip(rgb_train, ir_train):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.   tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape
            with tf.GradientTape() as g1_tape, tf.GradientTape() as g2_tape, tf.GradientTape() as d1_tape, tf.GradientTape() as d2_tape:
                # Running the forward pass for all models:
                # Getting the output from Generator 1.
                gen1_out = gen_1(batch_rgb)
                # Generating the input for Generator 2.
                # Concatenating visible and generated IR images to generate input to
                # Generator 2.
                in_gen2 = tf.concat([batch_rgb, gen1_out], 3)
                #print("Shape in_gen2: ",tf.shape(in_gen2))
                # Getting the output from Generator 2.
                gen2_out = gen_2(in_gen2)
                # Outputs of discriminators.
                # Discriminator 1.
                # Getting the output of Discriminator 1 for the generated fused image.
                disc1_out_g2 = disc_1(gen2_out)
                #print("Output batch Disc1 fused: ",tf.shape(disc1_out_g2))
                # Getting the output of Discriminator 1 for the RGB images.
                disc1_out_rgb = disc_1(batch_rgb)
                #print("Output batch Disc1 RGB: ",tf.shape(disc1_out_rgb))
                # Discriminator 2.
                # Getting the output of Discriminator 2 for the generated fused image.
                disc2_out_g2 = disc_2(gen2_out)
                #print("Output batch Disc2 fused: ",tf.shape(disc2_out_g2))
                # Getting the output of Discriminator 2 for the generated IR image.
                disc2_out_g1 = disc_2(gen1_out)
                #print("Output batch Disc2 fake IR: ",tf.shape(disc2_out_g1))
                # Getting the output of Discriminator 2 for the real IR images.
                disc2_out_ir = disc_2(batch_ir)
                #print("Output batch Disc2 real IR: ",tf.shape(disc2_out_ir))

                # Computing the loss values.
                # Calculating loss for Generator 1.
                gen1_loss = loss_g1(disc2_out_g1, gen1_out, batch_ir)
                #print("G1 cost: ",gen1_loss)
                # Calculating loss for Generator 2.
                gen2_loss = loss_g2(disc1_out_g2, disc2_out_g2, batch_ir, batch_rgb, gen2_out, batch_size)
                #print("G2 cost: ",gen2_loss)
                # Calculating loss for Discriminator 1.
                disc1_loss = loss_d1(disc1_out_rgb, disc1_out_g2, batch_size)
                #print("D1 cost: ",disc1_loss)
                # Calculating loss for Discriminator 2.
                disc2_loss = loss_d2(disc2_out_ir, disc2_out_g2, disc2_out_g1, batch_size)
                #print("D2 cost: ",disc2_loss)

            # Get the gradients of the different losses with the tape.
            # We optimize the discriminators once per every 2
            # generator runs.
            if(not(batch_count%3 == 0)): # With the (not), we optimize the discriminator 1 per every 2 generator runs.
                grads_gen1 = g1_tape.gradient(gen1_loss, gen_1.trainable_weights)
                grads_gen2 = g2_tape.gradient(gen2_loss, gen_2.trainable_weights)
                g1_optimizer.apply_gradients(zip(grads_gen1, gen_1.trainable_weights))
                g2_optimizer.apply_gradients(zip(grads_gen2, gen_2.trainable_weights))
            else:
                grads_disc1 = d1_tape.gradient(disc1_loss, disc_1.trainable_weights)
                grads_disc2 = d2_tape.gradient(disc2_loss, disc_2.trainable_weights)
                d1_optimizer.apply_gradients(zip(grads_disc1, disc_1.trainable_weights))
                d2_optimizer.apply_gradients(zip(grads_disc2, disc_2.trainable_weights))
            # Saving sample images of batch.
            if(save_flag and save_model):
                if(epoch%1 == 0):
                    save_flag = False
                    # Getting sample images.
                    sample_img_rgb = batch_rgb[0].numpy()
                    sample_img_ir = batch_ir[0].numpy()
                    sample_img_gir = gen1_out[0].numpy()
                    sample_img_fused = gen2_out[0].numpy()
                    # Generating paths.
                    sample_img_rgb_path = sample_img_path + "e" + str(epoch) + "rgb.png"
                    sample_img_ir_path = sample_img_path + "e" + str(epoch) + "ir.png"
                    sample_img_gir_path = sample_img_path + "e" + str(epoch) + "gir.png"
                    sample_img_fused_path = sample_img_path + "e" + str(epoch) + "fused.png"
                    # Saving sample images.
                    tf.keras.preprocessing.image.save_img(sample_img_rgb_path, sample_img_rgb, data_format="channels_last")
                    tf.keras.preprocessing.image.save_img(sample_img_ir_path, sample_img_ir, data_format="channels_last")
                    tf.keras.preprocessing.image.save_img(sample_img_gir_path, sample_img_gir, data_format="channels_last")
                    tf.keras.preprocessing.image.save_img(sample_img_fused_path, sample_img_fused, data_format="channels_last")

            # Log every [4] batches.
            if(batch_count%4 == 0):
                print("--------------------------------------------------------")
                print("Epoch: ", epoch)
                print("GEN1: Training loss (for one epoch) at batch %d: %.4f" % (batch_count, float(gen1_loss)))
                print("Seen so far: %s samples" % ((batch_count + 1) * batch_size))
                print("GEN2: Training loss (for one epoch) at batch %d: %.4f" % (batch_count, float(gen2_loss)))
                print("Seen so far: %s samples" % ((batch_count + 1) * batch_size))
                print("DISC1: Training loss (for one epoch) at batch %d: %.4f" % (batch_count, float(disc1_loss)))
                print("Seen so far: %s samples" % ((batch_count + 1) * batch_size))
                print("DISC2: Training loss (for one epoch) at batch %d: %.4f" % (batch_count, float(disc2_loss)))
                print("Seen so far: %s samples" % ((batch_count + 1) * batch_size))
                # Displaying sample images.
                # print("Last output of batch Gen1: ",tf.shape(gen1_out))
                # aux_gen1_o = plt.imshow(gen1_out[0])
                # plt.show()
                # print("Last output of batch Gen2: ",tf.shape(gen2_out))
                # aux_gen2_o = plt.imshow(gen2_out[0])
                # plt.show()
                print("--------------------------------------------------------")
            # Incrementing batch count.
            batch_count += 1

        # Calculating time per epoch and displaying it.
        elapsed_time = time.time() - start_time
        print("Elapsed time in epoch: ", elapsed_time)
        # save sample images of this epoch.

    # When the training is done, saves the weights of the
    # models if required.
    if(save_model):
        # Creating directory with this timestamp. "/home/ciprian/Checkpoints/"
        dir_path = check_path + timestamp_str + "/Weights/"
        os.makedirs(dir_path)
        # Generating save paths.
        gen1_path = dir_path + "GEN1.h5"
        gen2_path = dir_path + "GEN2.h5"
        disc1_path = dir_path + "DISC1.h5"
        disc2_path = dir_path + "DISC2.h5"
        # Saving models.
        gen_1.save_weights(gen1_path)
        gen_2.save_weights(gen2_path)
        disc_1.save_weights(disc1_path)
        disc_2.save_weights(disc2_path)
        # Feedback for end of training.
        print("The final weights of the models are saved in:")
        print(dir_path)
    # Feedback for end of training.
    print("Training finished!")
