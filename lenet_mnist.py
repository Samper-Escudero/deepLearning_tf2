from nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import numpy as np
from utils import orderImageShape

print("[INFO] loading MNIST...")
((trainX,trainY),(testX,testY))=mnist.load_data()

trainX = orderImageShape(trainX,K)
testX = orderImageShape(testX,K)
