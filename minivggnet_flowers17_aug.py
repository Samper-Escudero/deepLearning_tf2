from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import AspectAwarePreprocessor
from datasets import SimpleDatasetLoader
from nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils import plotMyNet

# image ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels)=sdl.load(imagePaths, verbose = 500)
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                            shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                fill_mode="nearest")

print("[INFO] compiling model...")
opt = SGD(0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")

# steps per peoch determines controls the number of batches per epoch
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32, epochs=100, verbose = 1)

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), target_names=classNames))

plotMyNet(H.history, epochs = 100)
