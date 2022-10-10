"""
importing MobileNetV2
"""
from keras.applications.mobilenet_v2 import MobileNetV2
from images import imread
import numpy as np

# imagenet - значит, что используется предобученная модель на большом объеме данных
model = MobileNetV2(weights='imagenet')

"""
Loading the image
"""
data = np.empty((1, 1754, 1240, 3))
data[0] = imread('../images/img_0.png')