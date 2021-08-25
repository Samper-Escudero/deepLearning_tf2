from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv MiniVGGNet
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from utils import myArgParser
from utils import plotMyNet
from tensorflow.keras.models import load_model

def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    alpha = initAlpha*(factor**np.floor((1+epoch)/dropEvery))
    return float(alpha)

args = myArgParser()

print("[INFO] loading CIFAR-10...")
((trainX,trainY),(testX,testY))=cifar10.load_data()
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
    # define the callbacks to be passed to the model during training
    callbacks = [LearningRateScheduler(step_decay)]
    print("[INFO] Compiling model...")
    opt = SGD(learning_rate=0.01, momentum=0.9, nesterov = True)
    model = MiniVGGNet.build(width=32, height=32,depth=3,classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training network...")
    H = model.fit(trainX,trainY,validation_data=(testX,testY),
                batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

print("[INFO] Evaluating network...")
predictions=model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

if args["loadM"]:
    plotMyNet(H, epochs=40)
else:
    plotMyNet(H.history,epochs=40)
