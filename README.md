# FIRe-GAN
Implementation of the *FIRe-GAN* model, a GAN-based method for the fusion of visible-infrared images.

## System specifications
This model was developed and tested on Ubuntu 16.04.6 LTS and Python 3.6.9.

It was also tested on two different versions of Tensorflow:
  * Tensorflow 2.4.0 (tested with no GPU support).
  * Tensorflow 2.3.0 (tested with single GPU support).

Two separate requirements.txt files are provided. Each one reflects the dependencies used for the program on the GPU and non-GPU environments.
  
**Note:** You can get the h5 files for the pre-trained weights [here](https://drive.google.com/drive/folders/1eqeywhESfunwuIlO1-RskRqLXUFQ5omG?usp=sharing). You will need to copy them to the **Sample_model** folder.
  
### GPU support
The code should run seamlessly if no GPU is detected. It is designed to work also with a **single** GPU (tested with a Tesla P100 GPU). You should specify in the **main.py** file the ID of the GPU you are using (even if you only have one). The latter is meant to prevent Tensorflow from allocating memory on multiple GPUs on a shared environment.

## Model summary
The basic structure is the one proposed by [Zhao et al.](https://www.hindawi.com/journals/mpe/2020/3739040/); for several implementation details (e.g. the number of neurons/filters per layer, among others), I referred to the [Pix2Pix model](https://paperswithcode.com/paper/image-to-image-translation-with-conditional) by Isola et al. for the *Generator 1*. For the *Generator 2* and both discriminators, I referred to the [FusionGAN model](https://www.researchgate.net/publication/327393843_FusionGAN_A_generative_adversarial_network_for_infrared_and_visible_image_fusion) by Ma et al.

The model was extended into the *FIRe-GAN* model by modifying the original architecture of the *Generator 1* from an encoder-decoder structure to a U-Net one. Additionally, the output layers of the generators were modified to allow for three-channel output images, and Spectral Normalization and the Two Time-Scale Update Rule (TTUR) were added for training stability. For more details on the architecture and the implemented extensions, see the paper [here](https://link.springer.com/article/10.1007/s00521-021-06691-3) (full-text, view-only version [here](https://rdcu.be/cBihC)).

## Running the program
To run the program, just type the following in the terminal:
```
python main.py
```

## Functionalities
There are five functions available on the main menu of the program:

 * **Training phase:** Trains a new model on a given dataset. Supports PNG, JPEG, and 4-channel TIFF images. Has the option of splitting the dataset into testing and validation sets. If said option is chosen, the program will perform as well data augmentation on the generated training set. Has the option of saving the weights of the trained model as h5 files, and saving the images generated during training.
 * **Validation phase:** Outputs the evaluation of a model on the generated validation set. Outputs a MAT file containing the results over four metrics: *image entropy*, *correlation coefficient*, *PSNR*, and *SSIM*. It also saves the average elapsed time per image.
 * **Testing phase:** Outputs the evaluation of a model on the testing set. It is expected that said set is different from the training and validation ones. Supports PNG, JPEG, and 4-channel TIFF images. Saves the generated images, and outputs a MAT file with the evaluation results.
 * **Transfer learning phase:** Trains a pre-trained model on a new dataset. Instead of initializing the weights of the model, takes as a starting point the weights of the given pre-trained one. Supports PNG, JPEG, and 4-channel TIFF images. Has the option of splitting the dataset into testing and validation sets. If said option is chosen, the program will perform as well data augmentation on the generated training set. Has the option of saving the weights of the trained model as h5 files, and saving the images generated during training.
 * **Single image demo:** Simple demo that loads a model, takes as input a single visible image, and saves the resulting infrared and fused images. Supports PNG and JPEG formats.
 
### Saving the models
At the start of the trainig phase, a unique timestamp ID is assigned to a model. This ID will be used and required by the rest of the functions. The program assumes a given directory architecture to store the model weights and generated images on the training and transfer learning phases:
 
```
   .
   ├── ...
   ├── Checkpoints             
   │   ├── ModelID_1             # The model ID is the timestamp at the start of the training phase.
   │   |    ├── SamplesTrain     # Here, the images generated during the training phase are saved.
   |   |    ├── WeightsTrain     # Here, the final weights of the model after the training phase are saved.
   |   |    ├── SamplesTransfer  # Here, the images generated during the transfer learning phase are saved.
   │   |    └── WeightsTrasfer   # Here, the final weights of the model after the transfer learning phase are saved.
   |   ├── ModelID_2
   |   |    └── ...
   |   └── ...
   └── ...
```

### Dataset loading
The program is constructed in a way in which, given the same dataset of image pairs (in the same order), it will always generate the exact same partitions with the exact same image order. Thanks to this, you can run every functionality at a different time, provided that the source datsets are not changed. In addition, the input images are always resized with padding to a height of 384 and a width of 512, as these are the input sizes expected by the model. All images are read and treated internally as three-channel images.

### Configuration variables
In the **config.ini** file you can specify all the parameters used for each functionality of the program. For more advanced configuration (loss functions, etc.), you will need to go to the corresponding Python file.
