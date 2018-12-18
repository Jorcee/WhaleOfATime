import cv2
import numpy as np
import pandas as pd
import tensorflow as ts
from keras import backend as K
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

img_size = 64

image_path = "./smallimages/"
save_path = "./smallimages/"
print("Loading Mapping info")
map = pd.read_csv("train.csv")
ids = map["Id"]
image_names = map["Image"]
unique_ids = set(ids)
num_ids = len(unique_ids)
print("Mapping Info Loaded")
print("Loading Images")
images = []
count = 1
for image in image_names:
    img = cv2.resize(cv2.imread(image_path + image, 0),(img_size,img_size))
    print(count)
    count +=1
    images.append(img)
    #cv2.imwrite(save_path + image, img)

cv2.imshow("test",images[np.random.randint(0,high=len(images))])
cv2.waitKey(0)
cv2.destroyAllWindows()
images = np.array(images)
images = images / 255.0
images = np.reshape(images,(-1, img_size, img_size, 1))
print("Images Loaded")
print("Building Train and Test Sets")
ids = np.array(ids)
print(images.shape)

input_shape = (img_size,img_size,1)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(ids)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels = onehot_encoder.fit_transform(integer_encoded)
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=.3)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
print("Train and Tests sets built")
print("Building Model")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_ids, activation='softmax'))

print("Model Built")
print("Compiling Model")
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
print("Model Compiled")
print("Training Model")
batch_size = 50
num_epoch = 1
# model_log = model.fit(train_x, train_y,
#           batch_size=batch_size,
#           epochs=num_epoch,
#           verbose=1,
#           validation_data=(test_x, test_y))
model_log = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1)
print("Model Trained")
print("Testing Model")
score = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
