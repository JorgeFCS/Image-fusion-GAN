################################################################################
# Condensed menu functionality.                                                #
#                                                                              #
# Tecnologico de Monterrey                                                     #
# MSc Computer Science                                                         #
# @ author Jorge Francisco Ciprian Sanchez                                     #
# Last updated: 11-01-2019                                                     #
################################################################################


# Importing custom functions.
from Functions.transfer_learning import *
from Functions.single_image_test import *
from Functions.image_processing import *
from Functions.validation import *
from Functions.models import *
from Functions.train import *
from Functions.test import *

def main_menu(config):
    buffer = input("Press any key to continue: ")
    # Menu flag.
    menu_flag = False
    # Data flag.
    data_flag = False
    # Menu loop.
    while(not menu_flag):
        # Resetting option flag.
        option_flag = False
        # Clearing screen.
        print(chr(27) + "[2J")
        print(chr(27) + "[1;1f")
        # Printing menu.
        print("Tecnologico de Monterrey")
        print("MSc Computer Science")
        print("Visible-infrared image fusion")
        print("Authors: Jorge Ciprian")
        print("Available options to execute:")
        print("1.- Train the model.")
        print("2.- Get validation set results.")
        print("3.- Test the model on new dataset.")
        print("4.- Transfer learning on new dataset.")
        print("5.- Single image demo.")
        print("6.- Exit.")
        # Validating input.
        while(not option_flag):
            try:
                option = int(input('Selected option: '))
                if(option < 1 or option > 6):
                    raise ValueError
                option_flag = True
            except ValueError:
                print("Not a valid option! The number must be in the range of 1-6.")
        # Executing according to option.
        if(option == 1):
            print("Model training.")
            # Initializing training parameters dictionary.
            train_params = {}
            # Asking if you want to save the model's weights after training.
            # The models would be saved in the specified path.
            save_flag = False
            while(not save_flag):
                option = input('Would you like to save the model weights after training? [y/n] ')
                if(option == 'y'):
                    # Adding parameters to dictionary.
                    train_params['SAVE'] = True
                    train_params['CHECK_PATH'] = config['TRAIN'].get('CHECK_PATH')
                    # Feedback.
                    save_flag = True # Flag for the loop.
                    print_str = "The weights will be saved in: "+train_params["CHECK_PATH"]+"[timestamp]/Weights/"
                    print(print_str)
                    # Waiting for input.
                    buffer = input("Press any key to continue: ")
                elif(option == 'n'):
                    # Adding parameters to dictionary.
                    train_params['SAVE'] = False
                    train_params['CHECK_PATH'] = None
                    # Feedback.
                    save_flag = True
                    print("The weights will not be saved.")
                    buffer = input("Press any key to continue: ")
                else:
                    print("Invalid input!")
            # Defining image paths.
            img_path_rgb = config['TRAIN'].get('RGB_PATH')
            img_path_ir = config['TRAIN'].get('IR_PATH')
            # Defining batch size.
            batch_size = config['TRAIN'].getint('BATCH_SIZE')
            # Saving batch size in train parameters as well.
            train_params['BATCH'] = batch_size
            # Split flag.
            v_split = config['TRAIN'].getboolean('V_SPLIT')
            # Generating training and validation sets.
            rgb_images_train, rgb_images_val, ir_images_train, ir_images_val = \
            load_datasets(img_path_rgb,img_path_ir,v_split,batch_size)
            # Setting data flag so that we don't load the data again for
            # validation.
            data_flag = True
            # Loading trainig epochs value.
            train_params['EPOCHS'] = config['TRAIN'].getint('EPOCHS')
            # Defining sample image directory.
            train_params['SAMPLE_IMG_DIR'] = config['TRAIN'].get('SAMPLE_IMG_DIR')
            # Storing the learning rates per network.
            train_params['GEN1_LR'] = config['TRAIN'].getfloat('GEN1_LRATE')
            train_params['GEN2_LR'] = config['TRAIN'].getfloat('GEN2_LRATE')
            train_params['DISC1_LR'] = config['TRAIN'].getfloat('DISC1_LRATE')
            train_params['DISC2_LR'] = config['TRAIN'].getfloat('DISC2_LRATE')
            # Storing the indicators for spectral normalization per network.
            train_params['GEN1_SPEC'] = config['TRAIN'].getboolean('GEN1_SPECTRAL')
            train_params['GEN2_SPEC'] = config['TRAIN'].getboolean('GEN2_SPECTRAL')
            train_params['DISC1_SPEC'] = config['TRAIN'].getboolean('DISC1_SPECTRAL')
            train_params['DISC2_SPEC'] = config['TRAIN'].getboolean('DISC2_SPECTRAL')
            # Storing flag for U-Net architecture.
            train_params['GEN1_UNET'] = config['TRAIN'].getboolean('GEN1_UNET')
            # Calling the train function.
            train(rgb_images_train, ir_images_train, train_params)
            buffer = input("Done! Press any key to continue: ")
        elif(option == 2):
            print("Model validation.")
            # Initializing validation parameters dictionary.
            val_params = {}
            # Generating datasets if required.
            if(not data_flag):
                # Loading image paths.
                img_path_rgb = config['VALIDATION'].get('RGB_PATH')
                img_path_ir = config['VALIDATION'].get('IR_PATH')
                # Defining batch size.
                batch_size = config['VALIDATION'].getint('BATCH_SIZE')
                # Split flag.
                v_split = config['VALIDATION'].getboolean('V_SPLIT')
                # Generating datasets.
                rgb_images_train, rgb_images_val, ir_images_train, ir_images_val = \
                load_datasets(img_path_rgb,img_path_ir,v_split,batch_size)
                # Updating data flag.
                data_flag = True
            # Storing the indicators for spectral normalization per network.
            val_params['GEN1_SPEC'] = config['VALIDATION'].getboolean('GEN1_SPECTRAL')
            val_params['GEN2_SPEC'] = config['VALIDATION'].getboolean('GEN2_SPECTRAL')
            # Loading flag for U-Net architecture.
            val_params['GEN1_UNET'] = config['VALIDATION'].getboolean('GEN1_UNET')
            # Loading the path for the model weights.
            val_params['MODEL_PATH'] = config['VALIDATION'].get('MODEL_PATH')
            # Getting model ID.
            val_params['MODEL_ID'] = config['VALIDATION'].get('MODEL_ID')
            # Loading display iamge flag.
            val_params['DISP_IMGS'] = config['VALIDATION'].getboolean('DISP_IMGS')
            # Call the function to evaluate the validation set.
            validation_set_eval(rgb_images_val, ir_images_val, val_params)
            print("Done!")
            buffer = input("Press any key to continue: ")
        elif(option == 3):
            print("Model testing.")
            # Initializing validation parameters dictionary.
            test_params = {}
            # Loading image paths.
            img_path_rgb = config['TEST'].get('RGB_PATH')
            img_path_ir = config['TEST'].get('IR_PATH')
            # Defining batch size.
            batch_size = config['TEST'].getint('BATCH_SIZE')
            # Split flag - we assume that we will be using the whole dataset
            # for testing.
            v_split = False
            # Generating datasets.
            rgb_images_test, ir_images_test = \
            load_datasets(img_path_rgb,img_path_ir,v_split,batch_size)
            # Storing the indicators for spectral normalization per network.
            test_params['GEN1_SPEC'] = config['TEST'].getboolean('GEN1_SPECTRAL')
            test_params['GEN2_SPEC'] = config['TEST'].getboolean('GEN2_SPECTRAL')
            # Loading flag for U-Net architecture.
            test_params['GEN1_UNET'] = config['TEST'].getboolean('GEN1_UNET')
            # Loading the path for the model weights.
            test_params['MODEL_PATH'] = config['TEST'].get('MODEL_PATH')
            # Getting model ID.
            test_params['MODEL_ID'] = config['TEST'].get('MODEL_ID')
            # Call the function to evaluate the validation set.
            test_set_eval(rgb_images_test, ir_images_test, test_params)
            print("Done!")
            buffer = input("Press any key to continue: ")
        elif(option == 4):
            print("Model transfer learning.")
            # Initializing validation parameters dictionary.
            trans_params = {}
            # Loading main path.
            trans_params["MAIN_PATH"] = config['TRANSFER'].get('MAIN_PATH')
            # loading batch size.
            trans_params['BATCH'] = config['TRANSFER'].getint('BATCH_SIZE')
            # Asking if you want to save the model's weights after training.
            # The models would be saved in the specified path.
            save_flag = False
            while(not save_flag):
                option = input('Would you like to save the model weights after training? [y/n] ')
                if(option == 'y'):
                    # Adding parameters to dictionary.
                    trans_params['SAVE'] = True
                    # Feedback.
                    save_flag = True # Flag for the loop.
                    print_str = "The weights will be saved in: "+trans_params["MAIN_PATH"]+"/WeightsTransfer/"
                    print(print_str)
                    # Waiting for input.
                    buffer = input("Press any key to continue: ")
                elif(option == 'n'):
                    # Adding parameters to dictionary.
                    trans_params['SAVE'] = False
                    # Feedback.
                    save_flag = True
                    print("The weights will not be saved.")
                    buffer = input("Press any key to continue: ")
                else:
                    print("Invalid input!")
            # Generating datasets if required.
            if(not data_flag):
                # Loading image paths.
                img_path_rgb = config['TRANSFER'].get('RGB_PATH')
                img_path_ir = config['TRANSFER'].get('IR_PATH')
                # Defining batch size.
                batch_size = config['TRANSFER'].getint('BATCH_SIZE')
                # Split flag.
                v_split = config['TRANSFER'].getboolean('V_SPLIT')
                # Generating datasets.
                rgb_images_train, rgb_images_val, ir_images_train, ir_images_val = \
                load_datasets(img_path_rgb,img_path_ir,v_split,batch_size)
                # Updating data flag.
                data_flag = True
            # Loading trainig epochs value.
            trans_params['EPOCHS'] = config['TRANSFER'].getint('EPOCHS')
            # Loading model id.
            trans_params['MODEL_ID'] = config['TRANSFER'].get('MODEL_ID')
            # Storing the learning rates per network.
            trans_params['GEN1_LR'] = config['TRANSFER'].getfloat('GEN1_LRATE')
            trans_params['GEN2_LR'] = config['TRANSFER'].getfloat('GEN2_LRATE')
            trans_params['DISC1_LR'] = config['TRANSFER'].getfloat('DISC1_LRATE')
            trans_params['DISC2_LR'] = config['TRANSFER'].getfloat('DISC2_LRATE')
            # Storing the indicators for spectral normalization per network.
            trans_params['GEN1_SPEC'] = config['TRANSFER'].getboolean('GEN1_SPECTRAL')
            trans_params['GEN2_SPEC'] = config['TRANSFER'].getboolean('GEN2_SPECTRAL')
            trans_params['DISC1_SPEC'] = config['TRANSFER'].getboolean('DISC1_SPECTRAL')
            trans_params['DISC2_SPEC'] = config['TRANSFER'].getboolean('DISC2_SPECTRAL')
            # Storing flag for U-Net architecture.
            trans_params['GEN1_UNET'] = config['TRANSFER'].getboolean('GEN1_UNET')
            # Calling the train function.
            transfer_learning(rgb_images_train, ir_images_train, trans_params)
            buffer = input("Done! Press any key to continue: ")
        elif(option == 5):
            print("Single image demo.")
            # Unpacking parameters from config file.
            demo_params = {}
            demo_params['IMAGE_PATH_RGB'] = config['DEMO'].get('IMAGE_PATH_RGB')
            demo_params['SAVE_PATH'] = config['DEMO'].get('SAVE_PATH')
            demo_params['MODEL_PATH'] = config['DEMO'].get('MODEL_PATH')
            demo_params['GEN1_SPECTRAL'] = config['DEMO'].getboolean('GEN1_SPECTRAL')
            demo_params['GEN2_SPECTRAL'] = config['DEMO'].getboolean('GEN2_SPECTRAL')
            demo_params['GEN1_UNET'] = config['DEMO'].getboolean('GEN1_UNET')
            # Calling demo function.
            single_image_test(demo_params)
            buffer = input("Done! Press any key to continue: ")
        else:
            menu_flag = True
    print("Bye!")
