import matplotlib.pyplot as plt
import numpy as np
import argparse

def plotMyNet(H, args="plottedNN.png", epochs = 100, save = False):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    if save:
      plt.savefig(args["output"])
    else:
      plt.show()
def myArgParser(dataset=False):
    ap = argparse.ArgumentParser()
    if dataset:
        ap.add_argument("-d","--dataset", required = True,
                        help="path to input dataset")
    ap.add_argument("-m", "--model", required=True,
                    help="path to model")
    ap.add_argument("-lm", "--loadM", default=False,type=bool,
                    help="train a new model(False) or load(True)")
    args = vars(ap.parse_args())

    return args
def orderImShape(trainX, K):
    if K.image_data_format()=="channels_first":
      trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1],trainX.shape[2]))
    else:
      trainX = trainX.reshape((trainX.shape[0], trainX.shape[1],trainX.shape[2],1))
    return trainX
