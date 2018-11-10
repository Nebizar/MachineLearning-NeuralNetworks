# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:43:00 2018

@author: Krzysztof Pasiewicz
"""
# Convolutional Neural Network

# Imporing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Building the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = .2, zoom_range = .2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch = 8000, nb_epoch = 25, validation_data = test_set, validation_steps = 2000)
