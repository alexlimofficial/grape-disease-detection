import os
from tensorflow.keras import optimizers


# Dataset config
img_height = 224
img_width = 224
num_channels = 3
input_shape = (img_height, img_width, num_channels)
num_classes = 4

# Train config
batch_size = 32
epochs = 30
learning_rate = 2e-5


# Config for Vanilla CNN
vanilla = {
    'optimizer': optimizers.Adam(learning_rate),
    'activation': 'relu',
    'outact': 'softmax',
    'loss': 'categorical_crossentropy'
}

vgg = {
    'optimizer': optimizers.Adam(learning_rate),
    'activation': 'relu',
    'outact': 'softmax',
    'loss': 'categorical_crossentropy'
}

inception = {
    'optimizer': optimizers.Adam(learning_rate),
    'activation': 'relu',
    'outact': 'softmax',
    'loss': 'categorical_crossentropy'
}

mobilenet = {
    'optimizer': optimizers.Adam(learning_rate),
    'activation': 'relu',
    'outact': 'softmax',
    'loss': 'categorical_crossentropy'
}

nasnet = {
    'optimizer': optimizers.Adam(learning_rate),
    'activation': 'relu',
    'outact': 'softmax',
    'loss': 'categorical_crossentropy'
}