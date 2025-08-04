"""
Configuration file for paths, constants, and hyperparameters
used throughout the coral reef segmentation and classification pipeline.

This module centralizes all file paths and key parameter settings
for easy access and modification.
"""

#Trained models / tuned hyperparameters
FILTER_MODELS_DIR = "models/filter_res34_5_v18"
HYPERPARAM_FILE = "data/segmentation/SAM2hyperparameters.json"

#Needed to reference the SAM2 backbone
SAM2_PATH = "C:\\Users\\lab\\Desktop\\segmentation\\sam2"
SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_PATH = f"{SAM2_PATH}/checkpoints/sam2.1_hiera_large.pt"

#This is our personal dictionary to map our roboflow annotations to classes for a ML model
REMAP_PATH = "data/remap.json"

#Do you want to save the predicted annotations as an image?
SAVE_MASKS = True
#Do you want to save the predicted annotations in COCO format?
SAVE_COCO = True
#How many pixels in the image file are not part of the actual image
CROP_SPACE = 7130

#Training Variables #################################################################################
#(You don't have to touch these parameters if you don't wish to train the model on new data)
VERSION = 1.0
VERBOSE = True

#Where coco annotations (from roboflow) and images are located
TRAIN_DIR = "data/versions/1.1/train"
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
CREATE_MASK_DATASET = True
MASK_DATA_PATH = "data/segmentation/maskloader_128_tolerance=0.2_v18.pt"
#I've found that lower tolerance is generally better (by reducing noise in the training data)
#The tradeoff is that with lower tolerance, some of the big proposed masks will be thrown out of the training set
TOLERANCE = 0.2
#These are the size of the mask inputs that will be used for classification
MASK_SIZE = (128, 128)
#Minimum area needed to accept a proposed mask
MIN_AREA = 1000

#For training the coral classifier
TRAIN_CORAL_FILTER = True
#Number of models in the ensemble
M = 5
#Max number of iterations through the entire data set
EPOCHS = 100
#How many data points are passed forward through the model before the gradient is updated
BATCH_SIZE = 32
#The 'size' of the step used for each gradient update
LR = 5*1e-5
#Regularization parameter
WEIGHT_DECAY = 5*1e-6
#How much of the data is perserved to train the ensembler
SPLIT = 0.3
#Patience parameter for early stopping
PATIENCE = 5
#Class Dictionary File
CLASSES_FILE = "data/classes_v18.json"
#Weight for increasing recall
NEG_WEIGHT = 0.25
#Which ResNet backbone the classifiers will use
RES = 34
#What is the minimum number of observations you want in each class? (synthetically oversample)
UPSAMPLE = 1000

#Run the trained model on the images in TRAIN_DIR
#and obtain evaluation metrics for data
EVAL = True
VAL_SIZE = 0.2  #Proportion of images we want in the held-out set
#Optionally, save the side-by-side comparisons
SAVE_IMG = True
FIG_SIZE = (16, 9)
#Data from yellowfin to be merged with prediction metric data
METADATA = "data/metadata/Day3_Photo_MetaData_sr4.xlsx"

#ResNet34, 3 models, upsampling=1000 + bootstrapping
#Ensemble model trained with out-of-sample accuracy: 0.7616, Recall: 0.8697, Precision: 0.8214

#ResNet34, 3 models, upsampling=1000 w/o bootstrapping
#Ensemble model trained with out-of-sample accuracy: 0.7883, Recall: 0.7899, Precision: 0.8910

#ResNet34, 7 models, upsampling=1000 w/o bootstrapping
#Ensemble model trained with out-of-sample accuracy: 0.7841, Recall: 0.8655, Precision: 0.8207

#ResNet34, 3 models, upsampling=1000 + bootstrapping
#Ensemble model trained with out-of-sample accuracy: 0.8790, Recall: 0.9307, Precision: 0.9210