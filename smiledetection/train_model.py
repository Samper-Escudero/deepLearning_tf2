from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
# also need to export python path
# export PYTHONPATH='~/''
from dl4cv.nn.conv import LeNet
from imutils import paths
from dl4cv.utils import plotMyNet
from dl4cv.utils import myArgParser
import numpy as np
import cv2
import os
import imutils
import argparse

args = myArgParser(dataset=True)
data=[]
labels=[]
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label =="positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels),2)

# handle dta imbalance
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
weightDict = dict(zip([0, 1],classWeight))
# print("[INFO] class weight: {}".format(weightDict))
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

print("[INFO] compiling model ...")
model = LeNet.build(width=28, height=28, depth = 1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), class_weight = weightDict, batch_size=64, epochs=15, verbose=1)

print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

plotMyNet(H.history, epochs=15)
