from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class ShallowNet:
    # This type of method takes neither a self nor a cls parameter
    # (but of course it’s free to accept an arbitrary number of
    # other parameters). Therefore a static method can neither
    #  modify object state nor class state.
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape =(height,width,depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # Conv layers --> CONV=>RELU
        model.add(Conv2D(32,(3,3),padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier and output layers
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
