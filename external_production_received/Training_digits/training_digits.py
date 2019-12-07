import numpy as np
import cv2
import glob
import re
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


K.set_image_dim_ordering('th')

# fix random seed for reproducibility\n
seed = 7
np.random.seed(seed)


types = '*.png'
image_list = []
path = "images/"


for files in types:
    image_list.extend(glob.glob(os.path.join(path, files)))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


image_list.sort(key=natural_keys)

Num = 0

data = []

for i in range(1, len(image_list)):

    image = cv2.imread(image_list[i])

    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold image
    res, img = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(img, None, (0, 0), 255)

    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(img, None, (0, 0), 0)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 120

    # your answer image
    temp = np.zeros(output.shape, dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for j in range(0, nb_components):
        if sizes[j] >= min_size:
            temp[output == j + 1] = 255

    # find contours
    _, contours, hierarchy = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for k, ctr in enumerate(sorted_contours):

        Num += 1

        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        letter = temp[y:y+h, x:x+w]

        roi = cv2.resize(letter, (14, 28), cv2.INTER_AREA)

        data.append([roi])

        # Image.fromarray(roi).save("character_pattern/" + str(Num) + ".png")

train_label_data = open("labels/train_labels.txt", "r")
trainList = []

for line in train_label_data:
    trainList.append(line)
labels = [x for v in trainList for x in v.rstrip().split(" ")]



test_split = 0.1
idx = int(len(data) * (1 - test_split))
X_train, y_train = np.array(data), np.array(labels)
X_test, y_test = np.array(data[idx:]), np.array(labels[idx:])


# --------------------------- Training handwritten digits using mnist data set -------------------
# reshape to be [samples][pixels][width][height]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def create_model():
    # create model
    models = Sequential()
    models.add(Conv2D(30, (5, 5), input_shape=(1, 28, 14), activation='relu'))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(15, (3, 3), activation='relu'))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Dropout(0.2))
    models.add(Flatten())
    models.add(Dense(128, activation='relu'))
    models.add(Dense(50, activation='relu'))
    models.add(Dense(num_classes, activation='softmax'))
    # Compile model
    models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return models


# build the model
model = create_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

# Exciting! we have trained our model
# Saving model weights for later use
model.save("model.h5")
print("model weights saved in model.h5 file")

# Saving model information in .json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("model saved as model.json file")

