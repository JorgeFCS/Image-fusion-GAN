################################################################################
# Visible-infrared image fusion.                                               #
# Implementation and extension of the method by Zhao et al.:                   #
# https://www.hindawi.com/journals/mpe/2020/3739040/                           #
#                                                                              #
# Tecnologico de Monterrey                                                     #
# MSc Computer Science                                                         #
# Jorge Francisco Ciprian Sanchez                                              #
################################################################################

# Imports.
import configparser
import tensorflow as tf
from Functions.menu import *


# GPU configuration
print("Configuring GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
  # Restrict TensorFlow to only use a given GPU.
  try:
    print("GPUs [5]: ", gpus[5])
    tf.config.experimental.set_visible_devices(gpus[5], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Logical GPUs: ", logical_gpus)
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
print("... done.")

# Loading configuration file.
print("Reading configuration file...")
config = configparser.ConfigParser()
config.read('config.ini')
print("... done.")

# Calling the main menu.
main_menu(config)
