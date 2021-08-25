from sklearn.preprocessing import LabelBinarizer
from nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from utils import myArgParser

args = myArgParser()

print("[INFO] Loading CIFAR-10 dataset")
((trainX,trainY),(testX,testY))=cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] Compiling model")
opt = SGD(learning_rate=0.01,momentum=0.9,nesterov=True, decay=0.01/40)
model = MiniVGGNet.build(height=32,width=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

checkpoint=ModelCheckpoint(args["model"],monitor="val_loss", save_best_only=True, verbose=1)
callbacks=[checkpoint]

print("[INFO] Training model...")
H = model.fit(trainX,trainY, validation_data=(testX,testY), batch_size=64, epochs=40,callbacks=callback, verbose=2)
