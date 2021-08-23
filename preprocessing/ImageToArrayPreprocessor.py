from tensorflow.keras.preprocessing.image import img_to_array

# applies the correction of the channels depending on the backend selected in keras
class ImageToArrayPreprocessor:
  def __init__(self,dataFormat=None):
    self.dataFormat = dataFormat

  def preprocess(self,image):
    return img_to_array(image,data_format=self.dataFormat)
