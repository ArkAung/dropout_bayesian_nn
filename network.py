import os
import random
from typing import Tuple
from dataset import Dataset

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model


class Network:
    def __init__(self, input_shape: Tuple[int, int, int], dropout_rate: float, num_classes: int):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
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
        x = Dropout(self.dropout_rate)(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(self.dropout_rate)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)

        out = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inp, out)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model

    def train_model(self, train_dataset: Dataset, val_dataset: Dataset, epochs: int) -> None:
        self._set_seeds(1337)

        es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=7)

        train_steps = train_dataset.datagenerator.samples / train_dataset.datagenerator.batch_size
        val_steps = val_dataset.datagenerator.samples / val_dataset.datagenerator.batch_size

        self.model.fit(train_dataset.datagenerator, steps_per_epoch=train_steps, epochs=epochs,
                       validation_data=val_dataset.datagenerator, validation_steps=val_steps,
                       callbacks=[es], verbose=1)

    def get_predictions(self, test_dataset: Dataset) -> np.ndarray:
        return self.model.predict(test_dataset)
