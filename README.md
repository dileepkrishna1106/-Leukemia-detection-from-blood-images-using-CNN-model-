import pandas as pd
import numpy as np
import seaborn as sns

import os
def count_images_in_folder(folder_path):
    total_images = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                total_images += 1
    return total_images

folder_path = "D:/Desktop/REKONEX/Blood cell Cancer [ALL]"
total_images = count_images_in_folder(folder_path)
print("Total images in folder:", total_images)
import os

def count_images_in_folder(folder_path):
    total_images = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                total_images += 1
    return total_images

folder_path = "D:/Desktop/REKONEX/Blood cell Cancer [ALL]"
total_images = count_images_in_folder(folder_path)
print("Total images in folder:", total_images)

train_path = "D:/Desktop/REKONEX/Blood cell Cancer [ALL]"
valid_path = "D:/Desktop/REKONEX/Blood cell Cancer [ALL]"

num_train_samples = 3228
num_val_samples =  3228
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

from numpy.random import seed
seed(101)
import tensorflow as tf
tf.random.set_seed(101)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten


import pandas as pd
import numpy as np
#import keras
#from keras import backend as K

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

train_batches = datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size
)

# Check the output shape of the labels in the first batch
print(train_batches)
print(train_batches[0][1].shape)

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

train_batches = datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size
)

valid_batches = datagen.flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size
)

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=1,
    shuffle=False
)


import tensorflow as tf
mobile = tf.keras.applications.mobilenet.MobileNet()# How many layers does MobileNet have?
len(mobile.layers)

from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

# Assuming 'mobile' is your pre-trained model
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 7 corresponds to the number of classes
x = Dropout(0.25)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.
model = Model(inputs=mobile.input, outputs=predictions)
model.summary()
# We need to choose how many layers we actually want to be trained.

# Here we are freezing the weights of all layers except the
# last 23 layers in the new model.
# The last 23 layers of the model will be trained.

for layer in model.layers[:-23]:
    layer.trainable = False
# Define Top2 and Top3 Accuracy

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',top_3_accuracy])
print(valid_batches.class_indices)
#Add weights to try to make the model more sensitive to melanoma

class_weights={
    0: 1.0, 
    1: 1.0, 
    2: 1.0, 
    3: 1.0
    
    
}

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# ... (Your existing code)

# Define the model checkpoint callback to save the model
#hyper prmeter tuning
checkpoint = ModelCheckpoint('your_model.keras',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

# Other callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-6, mode='min', verbose=1)

class Top3AccuracyCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_val_pred = self.model.predict(x_val)
        #top3_acc = top_3_accuracy(y_val, y_val_pred).numpy()
        #logs['val_top_3_accuracy'] = top3_acc
        #print(f'val_top_3_accuracy: {top3_acc}')

# Use the custom callback in your callbacks list
#top3_accuracy_callback = Top3AccuracyCallback(validation_data=(valid_batches, valid_batches.labels))
#callbacks_list = [checkpoint, reduce_lr, top3_accuracy_callback]
train_steps = int(train_steps)
val_steps = int(val_steps)
# Then, use callbacks_list in model.fit
history = model.fit(train_batches, steps_per_epoch=train_steps,
                    class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=5, verbose=1,
                    #callbacks=callbacks_list
                   )

# Save the final model after training
model.save('bloodwisepnormalmodel.h5')

import tensorflow as tf

def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
keras_model = tf.keras.models.load_model('bloodwisepnormalmodel.h5', custom_objects={'top_3_accuracy': top_3_accuracy})
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open('BloodWise.tflite', 'wb') as f:
    f.write(tflite_model)
s
