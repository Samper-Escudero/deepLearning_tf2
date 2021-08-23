from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from utils import plotMyNet
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from imutils import paths
import argparse

print("[INFO] loading CIFAR-10 dataset...")
((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert labels into vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

labelNames = ["airplanes","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

print("[INFO] compiling model...")
opt = SGD(learning_rate=0.01)
model = ShallowNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=32, epochs=40, verbose=1)

# Evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plotMyNet(H, epochs = 40, save=False)
