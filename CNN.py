#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import random
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tqdm import tqdm


# In[3]:


data_filename = "Downloads/kagglecatsanddogs_5340/PetImages" # data file location with dogs and cats pictures
categories = ["Dog", "Cat"] # two categories


# In[4]:


image_size = 100 # shape data into a smaller size
training_data = []
def create_training_data():
    for category in categories: 
        path = os.path.join(data_filename, category)  # create file path to dogs and cats
        categories_num = categories.index(category)  # classification 0 = dog and 1 = cat
        for img in tqdm(os.listdir(path)):  # loop through each image of dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert image to array and read in
                # grayscale the image so the data is smaller
                new_array = cv2.resize(img_array, (image_size, image_size))  # resize data
                training_data.append([new_array, categories_num])  # add to training data
            except Exception as e:  # exception handling
                pass

create_training_data() # call function to create training data

print(len(training_data)) # check the size


# In[5]:


random.shuffle(training_data) # shuffle data to make training more efficient
X = []
Y = [] # initialize two pickle 

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close() # X pickle

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close() # Y pickle


# In[6]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in) # load in X pickle

pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in) # load in Y pickle

X = X/255.0 # normalize data for image with value from 0-255

model = Sequential() # use a sequential model for single input

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:])) # first hidden layer with 64 neurons
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) 

model.add(Conv2D(64, (3, 3))) # second hidden layer with 64 neurons
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())  # converts 3D feature maps to 1D vectors

model.add(Dense(64))

model.add(Dense(1)) 
model.add(Activation('sigmoid')) # use sigmoid as activation function

model.compile(loss = 'binary_crossentropy', # compute loss
              optimizer = 'adam', # adam algorithm
              metrics = ['accuracy'])

X = np.array(X)
Y = np.array(Y) # convert to array to fit in

model.fit(X, Y, batch_size = 32, epochs = 20, validation_split = 0.1) # train model 20 times
# use batch size of 32 to save memory and validation split of 0.1 to save some samples to validate
model.save('dogs_cats_classifier.model') # save trained model 


# In[7]:


def prepare(filepath): # function to prepare prediction image
    image_size = 100 # resize the image
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image and convert to grayscale
    new_array = cv2.resize(img_array, (image_size, image_size))  # resize image to the same size
    return new_array.reshape(-1, image_size, image_size, 1)  # return to the training model


# In[8]:


model = tf.keras.models.load_model("dogs_cats.model") # load model
prediction = model.predict([prepare("Pictures/Saved Pictures/Bulldog_02.jpg")]) # load prediction file
print(categories[int(prediction[0][0])]) # output prediction
