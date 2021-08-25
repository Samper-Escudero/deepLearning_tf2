from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from nn.conv import LeNet
from utils.catchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True, help="path to input dataset")
ap.add_argument("-m", "--model",required=True, help="path to output model")
args = vars(ap.parse_args())

data=[]
labels=[]

for imagePath in path.list_images(args["dataset"]):

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    label=imagePath.split(os.path.sep)[-2]
    labels.append(label)
