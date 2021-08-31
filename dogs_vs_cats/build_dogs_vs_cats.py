from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from preprocessing import AspectAwarePreprocessor
from dl4cv_io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([],[],[])

for(dType, paths, labels, outputPath) in datasets:
    # loop over the datasets and create a writer for each one
    print("[INFO]  building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar()," ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = len(paths), widgets = widgets).start()

    for(i,path) in enumerate(zip(paths, labels)):
        # loop over each image on the dataset and its labels
        #
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # in the training dataset r,g and b pixels are to be extracted so that
        # mean substraction normalisation can be applied
        if dType =="train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image],[label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] serializing images...")
# create a dictionary with the average rgb values and save it on json
D ={"R": np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f.open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
