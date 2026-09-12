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
#CHANGE THIS to the full path where you cloned SAM2 (see README's Installation
#section, step 3) -- e.g. "/home/you/sam2" or "C:\\Users\\you\\sam2"
SAM2_PATH = "C:\\Users\\samle\\OneDrive\\Desktop\\Archive\\sam2"  #"/path/to/sam2"
SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_PATH = f"{SAM2_PATH}/checkpoints/sam2.1_hiera_large.pt"

#Maps raw Roboflow annotation labels to canonical model classes
REMAP_PATH = "data/remap.json"

#Do you want to save the predicted annotations as an image?
SAVE_MASKS = False
#Do you want to save the predicted annotations in COCO format?
SAVE_COCO = True
#How many pixels in the image file are not part of the actual image
CROP_SPACE = 7130

#Training Variables #################################################################################
#(You don't have to touch these parameters if you don't wish to train the model on new data)
VERSION = 1.2
VERBOSE = True

#Where coco annotations (from roboflow) and images are located
TRAIN_DIR = "yellowfin_segment.v18i.coco-segmentation/train" #data/train"
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
MASK_DATA_PATH = "data/segmentation/maskloader_128_tolerance=0.2_v18.pt"
#Lower tolerance reduces label noise but discards more large proposed masks from training
TOLERANCE = 0.2
#These are the size of the mask inputs that will be used for classification
MASK_SIZE = (128, 128)
#Minimum area needed to accept a proposed mask
MIN_AREA = 1000

#For training the coral classifier
TRAIN_CORAL_FILTER = False
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
#Of that ensembler pool, how much is further held out as the ensemble's own
#out-of-sample set (see CoralFilterEnsembler.train_ensemble). For methods
#"em"/"linear"/"reweight"/"multinomial" this is only used to validate the
#fit afterward; for "adam" it's also used during fitting for validation-
#based early stopping, so its validate() metrics are mildly optimistic.
ENSEMBLE_SPLIT = 0.1
#Seed for MASK_TRANSFORM_AUGMENT wherever it's used outside of CNN training
#itself (live inference in CoralFilter.predict/CoralFilterEnsembler.predict,
#and ensemble-logit extraction in extract_submodel_logits) -- makes an
#otherwise-random transform reproducible from run to run (see transforms.seeded_rng).
MASK_TRANSFORM_AUGMENT_SEED = 42
#Number of independent random EM initializations used to fit the ensemble's
#alpha/beta (see EMEnsembleOptimizer.fit in classifier.py) -- the best of these
#by weighted log-likelihood is kept
N_STARTS = 30
#Which combiner CoralFilterEnsembler.train_ensemble uses to fit the submodels
#together -- see classifier.py for each optimizer's full definition:
#  "em"          EMEnsembleOptimizer: generative multinomial-logit mixture, fit via EM
#  "linear"      LinearStackingOptimizer: closed-form ridge-regularized linear stacking
#  "adam"        AdamEnsembleOptimizer: gradient ascent on "em"'s objective via torch.optim.Adam
#  "reweight"    ClassReweightingOptimizer: Hadamard reweight-and-renormalize, fit via EM/MM
#  "multinomial" MultinomialRegressionOptimizer: per-submodel calibration only (no mixture
#                weights), exactly concave -- single Newton solve
#  "nn"          NeuralNetEnsembleOptimizer: small feedforward net on submodel probabilities,
#                fit via torch.optim.Adam
ENSEMBLE_METHOD = "reweight"
#Every ENSEMBLE_METHOD value CoralFilterEnsembler._make_ensemble_model supports
#-- used by scripts/retrain_ensemble.py to fit and persist all six combiners
#(not just ENSEMBLE_METHOD) in one run, and by scripts/generate_filter_reports.py
#to report on all of them together in model_performance.txt
ALL_ENSEMBLE_METHODS = ("em", "linear", "adam", "reweight", "multinomial", "nn")
#Ablation study switch: set to a 1-indexed submodel number (matching
#model_<i>.pth, e.g. 5 -> model_5.pth) to make CoralFilterEnsembler.predict()
#also compute that one submodel's own softmax(logits) alongside the ensemble
#prediction (stashed as self.last_ablation_proba; see
#CoralSegmenter.predict/self.last_ablation_result and train.py's eval()).
#None skips the ablation computation entirely.
ABLATION_SUBMODEL = 4
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
SAVE_IMG = False
FIG_SIZE = (16, 9)
#Data from yellowfin to be merged with prediction metric data
METADATA = "data/metadata/Day3_Photo_MetaData_sr4.xlsx"
#Minimum distance (in meters, matching the metadata's UTM coordinates) required between any
#train image and any test image. Images closer than this to an image in the opposite split
#are folded into the train set so the train/test independence assumption isn't violated by
#overlapping/adjacent photos. Set to None to disable this check.
SPATIAL_RADIUS = 2.0