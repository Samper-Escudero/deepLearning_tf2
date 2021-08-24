from nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import numpy as np
from utils import orderImShape
from utils import plotMyNet

print("[INFO] loading MNIST...")
((trainX,trainY),(testX,testY))=mnist.load_data()

trainX = orderImShape(trainX,K).astype("float")/255.0
testX = orderImShape(testX,K).astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] Compiling model...")
opt = SGD(learning_rate=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes = 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY), batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

plotMyNet(H.history,epochs = 20)
