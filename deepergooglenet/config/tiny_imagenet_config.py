from os import path

TRAIN_IMAGES = "../../datasets_dl4cv/tiny-imagenet-200/train"
VAL_IMAGES = "../../datasets_dl4cv/tiny-imagenet-200/val/images"

VAL_MAPPINGS = "../../datasets_dl4cv/tiny-imagenet-200/val/val_annotations.txt"

# paths to the WordNet hierarchy files
WORDNET_IDS = "../../datasets_dl4cv/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "../../datasets_dl4cv/tiny-imagenet-200/words.txt"

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES #there is no access to the test labels

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../../datasets_dl4cv/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = ".../../datasets_dl4cv/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "../../datasets_dl4cv/tiny-imagenet-200/hdf5/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "../../dl4cv_models/nn_models/tiny-imagenet-200/tiny-image-net-200-mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "../../dl4cv_models/nn_models/tiny-imagenet-200"
MODEL_PATH = path.sep.join([OUTPUT_PATH,
"checkpoints", "epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,
"deepergooglenet_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH,
"deepergooglenet_tinyimagenet.json"])
