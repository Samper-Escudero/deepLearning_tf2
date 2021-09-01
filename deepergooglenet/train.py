import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from dl4cv.preprocessing import ImageToArrayPreprocessor
from dl4cv.preprocessing import SimplePreprocessor
from dl4cv.preprocessing import MeanPreprocessor
from dl4cv.callbacks import EpochCheckpoint
from dl4cv.callbacks import TrainingMonitor
from dl4cv_io import HDF5DatasetWriter
from nn.conv import DeeperGoogleNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import argparse
import json

p = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
help="epoch to restart training at")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainlGen = HDF5DatasetWriter(config.TRAIN_HDF5, 64, aug = aug, preprocessors =[sp, mp, iap], classes = config.NUM_CLASSES)
valGen = HDF5DatasetWriter(config.VAL_HDF5, 64, aug = aug, preprocessors =[sp, mp, iap], classes = config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = DeeperGoogleNet.build(width=64, height=64, depth = 3, classes = config.NUM_CLASSES, reg = 0.0002)
    opt = Adam(1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print("[INFO] loading {}....". format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.learning_rate)))
    K.set_value(model.optimizer.learning_rate, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.learning_rate)))

callbacks =[
    EpochCheckpoint(args["checkpoints"], every = 5, startAt = args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH, startAt=args["start_epoch"])]

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=10,
    max_queue_size=10,
    callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
