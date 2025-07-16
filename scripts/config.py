"""
Configuration file for paths, constants, and hyperparameters
used throughout the coral reef segmentation and classification pipeline.

This module centralizes all file paths and key parameter settings
for easy access and modification.
"""

#Trained models / tuned hyperparameters
FILTER_MODELS_DIR = "models/filter"
HYPERPARAM_FILE = f"data/segmentation/SAM2hyperparameters.json"
MASK_DATA_PATH = "data/segmentation/maskloader_128_tolerance=0.1.pt"

#Needed to reference the SAM2 backbone
SAM2_PATH = "C:\\Users\\lab\\Box\\Research\\WHOI\\sam2"
SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_PATH = f"{SAM2_PATH}/checkpoints/sam2.1_hiera_large.pt"

#This is our personal dictionary to map our roboflow annotations to classes for a ML model
REMAP_PATH = "data/remap.json"

#Do you want to save the predicted annotations as an image?
SAVE_MASKS = True

#Training Variables #################################################################################
#(You don't have to touch these parameters if you don't wish to train the model on new data)
VERSION = 2.0
VERBOSE = False

#Where coco annotations (from roboflow) and images are located
TRAIN_DIR = "data/train"
#The code will find all images in the TRAIN_DIR with this extension
EXT = ".jpg"
#NOTE: COCO JSON file must be named "_annotations.coco.json" within this directory
#Standard image size that all images will be loaded in as
IMG_SIZE = (1024, 1024)

#For tuning the base segmenter
TUNE_SEGMENTER = False
#Maximum of number of iterations the minimizer will run to find an optimal parameter set
N_CALLS = 100
#The sample size used for each Monte Carlo validation step
K = 30

#For creating the mask data set used for classification
CREATE_MASK_DATASET = False
TOLERANCE = 0.1 #I've found that lower tolerance is generally better (by reducing noise in the training data)
#These are the size of the mask inputs that will be used for classification
MASK_SIZE = (128, 128)

#For training the coral filter
TRAIN_CORAL_FILTER = False
#Number of models in the ensemble
M = 11
#Max number of iterations through the entire data set
EPOCHS = 30
#How many data points are passed forward through the model before the gradient is updated
BATCH_SIZE = 32
#The 'size' of the step used for each gradient update
LR = 1e-4
#Regularization parameter
WEIGHT_DECAY = 1e-5
#How much of the data is perserved to train the ensembler
SPLIT = 0.3

#Run the trained model on the images in TRAIN_DIR
#and obtain evaluation metrics for data
EVAL = False
#Optionally, save the side-by-side comparisons
SAVE_IMG = False
FIG_SIZE = (16, 9)
#Data from yellowfin to be merged with prediction metric data
METADATA = "data/metadata/Day3_Photo_MetaData_sr4.xlsx"