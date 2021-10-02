from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# this network was introduced by Szegedy et al. in their 2014 paper, Going Deeper With Convolutions
# The architectura makes use of a network in network or micro-architecture to construct the macro-architecture
# The inception module was introduced here a building block that fits into a Convolutional Neural Network enabling
# it to learn CONV layers with multiple filters

# Two aspects are notworthy from this implementation:
#       * The model is tiny compared to previous sequential networks (AlexNet or VGGNet)
#       * The authors obtain such dramatic drop in network architecture while still increasing the depth overall networks
#         This is accomplished by removing FC layers and using global average pooling instead (recall that most weights are in the FC layers)

# This construction based on micro-architectures inspired later variants such as Residual modules (ResNet), the Fire Module (SqueezeNet)


# The idea behind inception module:
#       * It can be hard to decide size of the filter for a given CONV layer
#           why not learn 5x5, 3x3 and 1x1, computing them in parallel and then
#           concatenate the resulting feature maps along the channel dimensions?
#           The next layers receives these concatenated, mixed filters and performs the same process
#           This process taken as a whole enables GoogleNet to learn both local features(small convs) and abstracted features (larger convs)
#       * By learning multiple filter sizes, the module can be turned into a multi-level feature extractor
#               - 5 x 5 can learn abstract features
#               - 1 x 1 is by definition local
#               - 3 x 3 is a balance between the previous

# Four branches in the inception module:
#       1. 1 x 1 local features from input
#       2. 1 x 1 convolution (not only to learn local features but to dimensionality reduction) and 3 x 3.
#               Preceding 3 x 3 and 5 x 5 with 1 x 1 is a good practice to reduce computation. The nº of 1 x1 in this branch is always smaller than nº 3 x 3
#       3. 1 x 1 convolution to reduce and 5 x 5
#       4. Pool projection branch --> 3 x 3 max pooling with a stride of 1 x 1
#               Over the years, models performing pooling demonstrated obtaining higher accuracy
#               Even though Springenberg et al.Striving for Simplicity: The All Convolutional Net
#               showed that POOL can be replaced by CONV to reduce volume size
#               Szegedy et al. added POOL based on hypothesis it was required so CNNs perform properly.
#               The output is then fed into a series of 1 x 1 convs to learn local features

#       Finally, all four branches are concatenated along the channel dimension. Special care is taken during implementation
#       to ensure the output of each branch has the same volume size.
#       In practice, it is common to stack various Inception layers before reducing dim with POOL

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding = "same", reg = .0005, name = None):
        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name * "_bn"
            actName = name + "_act"

        x = Conv2D(K,(kX, kY), srides = stride, padding = padding, kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis = chanDim, name = bnName)(x)
        x = Activation("relu", name = actName)(x)

        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj, chanDim, stage, reg=0.0005):

        first = DeeperGoogleNet.conv_module(x, num1x1, 1, 1,(1,1), chanDim, reg=reg, name = stage +"_first")
        second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1, (1, 1), chanDim, reg = reg, name = stage +"_second1")
        second = DeeperGoogleNet.conv_module(x, num3x3, 3,3, (1,1), chanDim, reg = reg, name = stage + "_second2")

        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1, (1, 1), chanDim, reg=reg, name=stage + "_third1")
        third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5, (1, 1), chanDim, reg=reg, name=stage + "_third2")

        fourth = MaxPooling2D((3, 3), strides=(1, 1),padding="same", name=stage + "_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj,1, 1, (1, 1), chanDim, reg=reg, name=stage + "_fourth")

        x = concatenate([first, second, third, fourth], axis=chanDim, name=stage + "_mixed")

        return x

    @staticmethod
    def buld(width, height, depth, classes, reg=0.0005):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim=1

        inputs = Input(shape=inputShape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chanDim, reg=reg, name="block1")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
        name="pool2")(x)

        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        model = Model(inputs, x, name="googlenet")

        return model
