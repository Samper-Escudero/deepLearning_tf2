# define the paths to the images directory
IMAGES_PATH = "../datasets/kaggle_dogs_vs_cats/train"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

MODEL_PATH = "output/alexnet_dogs_vs_cats.model"
DATASET_MEAN = "output/dogs_vs_cats_mean.json"
OUTPUT_PATH = "output"
