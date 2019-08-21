"""
VGG16 model.

Author: Alex Lim
"""
import os
import sys
import config
from config import vgg
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16 as VGG
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from utils import evalutils


class VGG16:
    def __init__(self):
        """Implementation of VGG16 classification model."""
        self.name = 'VGG16'
        self.model = None
        self.history = None
        self.build_model()

    def build_model(self):
        conv_base = VGG(
            include_top=False,
            weights='imagenet',
            input_shape=config.input_shape
        )
        conv_base.trainable = False
        print(conv_base.summary())
        inputs = Input(shape=config.input_shape)
        x = conv_base(inputs)
        x = Flatten()(x)
        x = Dense(units=256, activation=vgg['activation'])(x)
        x = Dropout(rate=0.5)(x)
        outputs = Dense(units=config.num_classes, activation=vgg['outact'])(x)
        self.model = Model(inputs=inputs, outputs=outputs)

    def summary(self):
        print(self.model.summary())

    def train(self, data):
        self.model.compile(
            loss=vgg['loss'],
            optimizer=vgg['optimizer'],
            metrics=['acc']
        )
        self.history = self.model.fit_generator(
            data['train'],
            steps_per_epoch=data['num_train']//config.batch_size,
            epochs=config.epochs,
            validation_data=data['val'],
            validation_steps=data['num_val']//config.batch_size
        )

    def evaluate(self, saved_model, generator, num_instances, name):
        model = load_model(filepath=saved_model)
        evalutils.get_metrics(
            model, generator=generator, steps=num_instances, name=name)

    def save(self, name):
        if not os.path.exists('./trained_models'):
            os.makedirs('./trained_models')
        savepath = os.path.join('./trained_models', name)
        self.model.save(savepath)
        print('Saved model as {}'.format(savepath))
        return savepath

    def load(self, saved_model):
        self.model = load_model(saved_model)
