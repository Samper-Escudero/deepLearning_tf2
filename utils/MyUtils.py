import matplotlib.pyplot as plt
import numpy as np
import argparse

def plotMyNet(H, args="plottedNN.png", epochs = 100, save = False):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    if save:
      plt.savefig(args["output"])
    else:
      plt.show()
def myArgParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset", required = True,
                    help="path to input dataset")
    args = vars(ap.parse_args())
    return args
