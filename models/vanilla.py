"""
Vanilla CNN model.

Author: Alex Lim
"""
import os
import sys
import config
from config import vanilla
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from utils import evalutils


class VanillaCNN:
    """Implementation of vanilla CNN classification model."""
    def __init__(self):
        self.name = 'VanillaCNN'
        self.model = None
        self.history = None
        self.build_model()

    def build_model(self):
        inputs = Input(shape=config.input_shape)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation=vanilla['activation'], \
                    input_shape=config.input_shape)(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation=vanilla['activation'])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation=vanilla['activation'])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation=vanilla['activation'])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=128, activation=vanilla['activation'])(x)
        x = Dropout(rate=0.5)(x)
        outputs = Dense(units=config.num_classes, activation=vanilla['outact'])(x)
        self.model = Model(inputs=inputs, outputs=outputs)

    def summary(self):
        print(self.model.summary())

    def train(self, data):
        self.model.compile(
            loss=vanilla['loss'],
            optimizer=vanilla['optimizer'],
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

