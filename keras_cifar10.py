from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plotMyNet(H, save = False):
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
  plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  if save:
    plt.savefig(args["output"])

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] accessing CIFAR-10...")
((trainX, trainY),(testX,testY))=cifar10.load_data()

# Flattening
trainX = trainX.reshape((trainX.shape[0], 32*32*3))
testX = testX.reshape((testX.shape[0],32*32*3))

# rescale
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# label tags
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# define model
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10,activation="softmax"))

# train
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs = 100, batch_size=32)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names = [str(x) for x in lb.classes_]))

plotMyNet(H, save = True)
