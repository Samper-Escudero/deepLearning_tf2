from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required = True, help = "path to input dataset")
ap.add_argument("-k","--neighbors", type=int, default=1, help="no. nearest neighbors considered for classification")
ap.add_argument("-j", "--jobs",type=int,default=-1,help="jobs for k-NN distance (-i uses all available cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# intialise preprocessor, load dataset and reshape images
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths,verbose = 500)
data = data.reshape((data.shape[0],3072))

# show memory consumption of the images loaded
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Divide the dataset for training and test (75-25)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")

# class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform',
# algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)

# algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’

# weights{‘uniform’, ‘distance’} or callable, default=’uniform’
# distance = weight points by the inverse of their distance. in this case,
# closer neighbors of a query point will have a greater influence than neighbors which are further away.

# p: int, default=2 (metric: Minkowski (1) or Euclidean (2) )

model = KNeighborsClassifier(n_neighbors = args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX), target_names=le.classes_))
