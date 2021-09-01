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
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding = "same", reg = .0005, name = None):
        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name * "_bn"
            actName = name + "_act"

        x = Conv2D(K,(kX, kY), srides = stride, padding = padding, kernel_regularizer=12(reg), name=convName)(x)
        x = BatchNormalization(axis = chanDim, name = bnName)(x)
        x = Activation("relu", name = actName)(x)

        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg=0.0005):

        first = DeeperGoogleNet.conv_module(x, num1x1, 1, 1,(1,1), chanDim, reg=reg, name = stage +"_first")
        second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1), chanDim, reg = reg, name = stage +"_second1")
        second = DeeperGoogleNet.conv_module(x, num3x3, 3,3, (1,1), chanDim, reg = reg, name = stage + "_second2")
