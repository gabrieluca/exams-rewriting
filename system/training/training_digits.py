##### MODULES #####

import numpy as np 
import cv2

# Glob module for filename utilities (https://www.youtube.com/watch?v=OZ6cNezon2Y)
import glob

# Regular expression utilities (https://www.youtube.com/watch?v=sZyAn2TW7GY)
import re

import os

# Utilitie for image (https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
from PIL import Image

# Keras utilities #
from keras.models import Sequential                     # https://keras.io/getting-started/sequential-model-guide/
from keras.layers import Dense                          # https://youtu.be/49IOTCzoWQg
from keras.layers import Dropout                        # https://youtu.be/NhZVe50QwPM & https://youtu.be/ARq74QuavAo
from keras.layers import Flatten                        # https://www.youtube.com/watch?v=mFAIBMbACMA
from keras.layers.convolutional import Conv2D           # https://keras.io/layers/convolutional/     
from keras.layers.convolutional import MaxPooling2D     # https://youtu.be/ZjM_XQa5s6s
from keras.utils import np_utils
from keras import backend as K                          # https://www.youtube.com/watch?v=okCAqDxCXVk        

# Needs to check the necessity of this line of code:
K.set_image_dim_ordering('th')

# fix random seed for reproducibility\n
seed = 7
np.random.seed(seed)


'''
Read all images in the Training DataBase
And sort their name with respect to the alphabetical order.
Based on the sorted name, create image list.
Note: these lines will be modified when the UI get ready
'''
types = '*.png'
image_list = []
path = "DataBase/images/"

for files in types:
    image_list.extend(glob.glob(os.path.join(path, files)))

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

image_list.sort(key=natural_keys)


'''
In the next step, we must extract letters in each image.
For it:
. convert rgb image to binary one
. Remove borders and noise


'''
Num = 0
data = []

for i in range(1, len(image_list)):

    image = cv2.imread(image_list[i])

    # Converts to Grayscale
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image (https://www.youtube.com/watch?v=6pX3II2eVs0)
    res, img = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # FloodFills to remove borders
    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(img, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(img, None, (0, 0), 0)

    # Morphological Ops (https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
