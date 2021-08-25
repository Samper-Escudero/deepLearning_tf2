from callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from utils import myArgParser
import os
import numpy as np
import argparse
from sklearn.metrics import classification_report


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
help="path to the output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


# define the callbacks to be passed to the model during training
print("[INFO] Compiling model...")
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov = True)
model = MiniVGGNet.build(width=32, height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#construct the set of callbacks
figPath = os.path.sep.join([args["output"],"{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"],"{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath,jsonPath=jsonPath)]

print("[INFO] Training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),
            batch_size=64, epochs=40, callbacks=callbacks, verbose=1)


print("[INFO] Evaluating network...")
predictions=model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
