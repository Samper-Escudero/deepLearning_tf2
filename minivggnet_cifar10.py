from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

from nn.conv import MiniVGGNet
from utils import myArgParser
from utils import plotMyNet

args = myArgParser()
print("[INFO] Loading CIFAR-10 dataset...")
((trainX,trainY), (testX,testY))= cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

if args["loadM"]:
    print("[INFO] Loading model...")
    model = load_model(args["model"]+".hdf5")
    H=np.load(args["model"]+'.npy',allow_pickle='TRUE').item()
else:
    print("[INFO] Compiling model...")
    opt = SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov = True)
    model = MiniVGGNet.build(width=32, height=32,depth=3,classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training network...")
    H = model.fit(trainX,trainY,validation_data=(testX,testY),
                batch_size=64, epochs=40, verbose=1)

print("[INFO] Evaluating network...")
predictions=model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

if args["loadM"]:
    plotMyNet(H)
else:
    plotMyNet(H.history)
