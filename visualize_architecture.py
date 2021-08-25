from nn.conv import LeNet
from tensorflow.keras.utils import plot_model

model = LeNet.build(28,28,1,10)
# will write None where the batch_size goes, given that is not know at this moment
plot_model(model, to_file="output/lenet.png", show_shapes=True)
