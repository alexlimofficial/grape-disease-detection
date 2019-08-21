"""
Dataset class.

Author: Alex Lim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.set_random_seed(100)


class Dataset:
    """Dataset class which constructs the necessary ImageDataGenerator 
    pipelines to feed into TensorFlow/Keras models.
    """
    def __init__(self, datapath):
        self.datapath = datapath
        self.data = {}

        self.create_dataset()

    def create_dataset(self):
        """Creates the dataset generators from the dataset to
        be used to feed batches of images to a model.
        """
        train_dir = os.path.join(self.datapath, 'train')
        val_dir = os.path.join(self.datapath, 'validation')
        test_dir = os.path.join(self.datapath, 'test')

        assert os.listdir(train_dir) == os.listdir(val_dir) == os.listdir(test_dir), \
            'Train/Validation/Test directories must contain same class folders.'

        classes = os.listdir(train_dir)

        total_train = 0
        total_val = 0
        total_test = 0

        for label in classes:
            num_train = len(os.listdir(os.path.join(train_dir, label)))
            num_val = len(os.listdir(os.path.join(val_dir, label)))
            num_test = len(os.listdir(os.path.join(test_dir, label)))
            total_train += num_train
            total_val += num_val
            total_test += num_test

        self.data['num_train'] = total_train
        self.data['num_val'] = total_val
        self.data['num_test'] = total_test

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(config.img_height, config.img_width),
            color_mode='rgb',
            batch_size=config.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        validation_generator = test_datagen.flow_from_directory(
            directory=val_dir,
            target_size=(config.img_height, config.img_width),
            color_mode='rgb',
            batch_size=config.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        valtest_generator = test_datagen.flow_from_directory(
            directory=val_dir,
            target_size=(config.img_height, config.img_width),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=(config.img_height, config.img_width),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )

        self.data['train'] = train_generator
        self.data['val'] = validation_generator
        self.data['valtest'] = valtest_generator
        self.data['test'] = test_generator
