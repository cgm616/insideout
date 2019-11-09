import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Input, MaxPooling2D, Conv2D, Flatten
from keras import optimizers

# Read the dataset into a dataframe
raw_data_file_name = "dataset/fer2013.csv"
raw_data = pd.read_csv(raw_data_file_name)

# Separate out the training data from the testing data
train_data = raw_data[raw_data["Usage"] == "Training"]

first_test_data = raw_data[raw_data["Usage"] == "PrivateTest"]
first_test_data.reset_index(inplace=True)

second_test_data = raw_data[raw_data["Usage"] == "PublicTest"]
second_test_data.reset_index(inplace=True)

# Create the expected vectors for each image
train_expected = keras.utils.to_categorical(train_data["emotion"], num_classes=7, dtype='int32')
first_test_expected = keras.utils.to_categorical(first_test_data["emotion"], num_classes=7, dtype='int32')
second_test_expected = keras.utils.to_categorical(second_test_data["emotion"], num_classes=7, dtype='int32')

# This function will transform the space-separated pixel strings into arrays of floats
def process_pixels(array_input):
    output = np.empty([int(len(array_input)), 2304])
    for index, item in enumerate(output):
        item[:] = array_input[index].split(" ")
    output /= 255
    return output

# Perform the conversion on each of the sets of data and reshape them into the shape we need
train_pixels = process_pixels(train_data["pixels"])
train_pixels = train_pixels.reshape(train_pixels.shape[0], 48, 48, 1)

first_test_pixels = process_pixels(first_test_data["pixels"])
first_test_pixels = first_test_pixels.reshape(first_test_pixels.shape[0], 48, 48, 1)

second_test_pixels = process_pixels(second_test_data["pixels"])
second_test_pixels = second_test_pixels.reshape(second_test_pixels.shape[0], 48, 48, 1)

# Start building the Convolutional Neural Network with input shape (?, 48, 48, 1) and output (?, 7)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = (48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

# Using the adamax optimizer, set up the regression
adamax = optimizers.Adamax()
model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])

# Train the model on the training data
model.fit(train_pixels, train_expected, validation_data=(train_pixels, train_expected), epochs=10, batch_size=64)

# Evaluate the performance of the model on all of the training data and print
train_score = model.evaluate(train_pixels, train_expected, batch_size=32)
print("train score: {}".format(train_score))

# Evaluate the performance of the model on some of the testing data and print
test_score = model.evaluate(first_test_pixels, first_test_expected, batch_size=32)
print("test score: {}".format(test_score))

# Save the model to disk for loading by the app
model.save('model/cnn.h5')