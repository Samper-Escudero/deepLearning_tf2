from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
help="path to models directory")
args = vars(ap.parse_args())

(testX, testy) = cifar10.load_data()[1]
testX = testX.astype("float")/255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# obtain all names ending in .model
modelPaths = os.path.sep.join([args["model"],"*.model"])
# create a list with the result
modelPaths = list(glob.glob(modelPaths))
models = []

for(i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i+1,len(modelPaths)))
    models.append(load_model(modelPath))

print("[INFO] evaluating ensemble...")
predictions =[]

for model in models:
    predictions.append(model.predict(testX,batch_size=64))

predictions = np.average(predictions.argmax(axis = 0))
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)
