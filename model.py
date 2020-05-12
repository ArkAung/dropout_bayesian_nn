import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import *

from sklearn.metrics import classification_report

import os
import numpy as np
import random

from typing import Tuple


class Model:
    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.model = self._build_model()

    @staticmethod
    def _set_seeds(seed):
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self) -> Model:
        inp = Input(shape=self.input_shape)

        x = Conv2D(32, (3, 3), activation='relu')(inp)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.4)(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)

        out = Dense(5, activation='softmax')(x)

        model = Model(inp, out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train_model(self, train_gen, val_gen: , batch_size: int, epochs: int) -> Model:
        self._set_seeds(1337)

        es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)

        train_steps = train_gen.samples / batch_size
        val_steps = val_gen.samples / batch_size

        self.model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                       validation_data=val_gen, validation_steps=val_steps,
                       callbacks=[es], verbose=2)

    def get_classification_report(self):
        print(classification_report([map_labels[label] for label in label_test],
                                    [map_labels[label] for label in pred_test]))