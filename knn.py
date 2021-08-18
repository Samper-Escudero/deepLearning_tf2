from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
import argparse
