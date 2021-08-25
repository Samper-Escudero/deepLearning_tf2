from sklearn.preprocessing import LabelBinarizer
from nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import os
from utils import myArgParser

args = myArgParser()

print("[INFO] loading CIFAR-10...")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] Compiling model")
opt=SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32,depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# callback to save only the best model to disk
fname = os.path.sep.join([args["model"],"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname,monitor="val_loss",mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

H = model.fit(trainX, trainY,validation_data=(testX,testY),batch_size=64,epochs=40,callbacks=callbacks, verbose=2)
