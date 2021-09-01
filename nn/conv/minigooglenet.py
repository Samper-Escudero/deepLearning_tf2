from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K

class MiniGoogleNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        x = Conv2D(K, (kX, kY), strides = stride, padding = padding)(x)
        x = BatchNormalization(axis=chanDim)(x) # it does not use Sequential, so the input is denoted between parenthesis
        x = Activation("relu")(x)

        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1,1, (1,1), chanDim)
        conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3, (1,1), chanDim)
        x = concatenate([conv_1x1m conv_3x3], axis=chanDim)

        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        conv_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2,2), chanDim, padding="valid")
        pool = MaxPooling2D((3,3),strides=(2,2))(x)
        x = concatenate([conv_3x3, pool], axis = chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = 1
        inputs = Input(shape=inputShape)
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        x = MiniGoogleNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogleNet.inception_module(x, 32,48, chanDim)
        x = MiniGoogleNet.downsample_module(x, 80, chanDim)

        x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogleNet.downsample_module(x, 96, chanDim)

        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="googlenet")
        return model