from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from utils import plotMyNet, myArgParser
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths

args = myArgParser

print("[INFO] loading dataset...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels)=sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(trainX,testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels into vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(learning_rate=0.005)
model = ShallowNet.build(width=32,height=32,depth=3,classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=32, epochs=100, verbose=1)

# Evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat","dog","panda"]))

plotMyNet(H, save=False)
