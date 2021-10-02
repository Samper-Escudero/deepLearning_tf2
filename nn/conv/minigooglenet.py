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
from tensorflow.keras import backend as K

# chanDim is the channel dimensions which is either "channels last" or "channels first"



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

class MiniGoogleNet:

    #   GoogleNet implemented the inception module to work with 224x224x3 images and obtain state-of-the-art accuracy.
    #   Nonetheless, the module can be simplified to work with smaller datasets where fewer parameters are required.
    #   Miniception --> Zhang et al.’s 2017 publication, Understanding Deep Learning Requires Re-Thinking Generalization
    #
    #   Three modules are used in the MiniGoogleNet:
    #       1. (left) A conv module responsible for convolution, batch normalisation and activation
    #       2. (middle) Two sets of convolutions (1 for 1x1  filters and other for 3x3 filters), then concatenates the result
    #               No dimensionality reduction before 3x3 as input volumes are already smaller due to the dataset size
    #       3.(right) A downsample module which applies both a conv and max pooling layer to reduce dimensionality,
    #               then concatenates them

    #  These modules are combined to build the MiniGoogleNet. Note that authors placed the batch normalisation BEFORE the
    #  activation (pressumable because Szegedy did too). In contrast to what is now recommended.


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
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

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
        chanDim = -1
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
