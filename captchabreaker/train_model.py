from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from nn.conv import LeNet
from utils.catchahelper import preprocess
from imutils import paths
from utils import plotMyNet
import numpy as np
import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True, help="path to input dataset")
ap.add_argument("-m", "--model",required=True, help="path to output model")
args = vars(ap.parse_args())

data=[]
labels=[]

for imagePath in path.list_images(args["dataset"]):

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    label=imagePath.split(os.path.sep)[-2]
    labels.append(label)
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

print("[INFO] Compiling model...")
model = LeNet.build(widht=28, height=28, depth=1,classes=9)
opt =SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=32, epochs = 15, verbose=1)

print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

print("[INFO] saving model...")
model.save(args["model"])

plotMyNet(H.history, epochs = 15)
