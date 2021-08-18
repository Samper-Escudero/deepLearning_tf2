import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
            self.preprocessors = preprocessors
            # If preprocessors are not set, a list is set so that they can be
            # sequentialy set later on
            if self.preprocessors is None:
                self.preprocessors = []

    def load(self, imagePaths, verbose = -1):
        data = []
        labels = []

        # loop over the input images
        # Assuming that path is /path/to/dataset/class/image.jpg
        # where class is the label of the picture
        for (i,imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # preprocess the image using the preprocessors available
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i>0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple
        return (np.array(data), np.array(labels))
