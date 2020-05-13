"""
Dataset class which includes:
    - Preparing data generator with data augmentations
    - Creating data iterator from images in directory

Dataset class has additional attribute which can hold label mappings and
an attribute which the user can specify the classes that should be included
in training.


Author: Arkar Min Aung
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from typing import List, Dict, Tuple
import enum
import numpy as np


class DatasetType(enum.Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class Dataset:
    def __init__(self, path: str, target_size: Tuple[int, int], dataset_type: DatasetType,
                 batch_size: int, data_generator: ImageDataGenerator = None,
                 class_filter: List[str] = None,
                 label_mapping: Dict[int, str] = None):
        self.path = path
        if dataset_type == DatasetType.TRAIN:
            self.dataset_subset = 'training'
            self.dataset_shuffle = True
        elif dataset_type == DatasetType.VAL:
            self.dataset_subset = 'validation'
            self.dataset_shuffle = False
        elif dataset_type == DatasetType.TEST:
            self.dataset_subset = None
            self.dataset_shuffle = False
        self.dataset_type = dataset_type
        self.label_mapping = label_mapping
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_filter = class_filter
        if data_generator is not None:
            self.data_generator = data_generator
        else:
            self.data_generator = self._get_data_generator()
        self.data_iterator = self._get_data_iterator()

    def _get_data_generator(self) -> ImageDataGenerator:
        if self.dataset_type == DatasetType.TRAIN:
            return ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest',
                                validation_split=0.2
                                )
        else:
            return ImageDataGenerator(rescale=1./255)

    def _get_data_iterator(self) -> DirectoryIterator:
        return self.data_generator.flow_from_directory(self.path,
                                                       target_size=self.target_size,
                                                       class_mode='categorical',
                                                       subset=self.dataset_subset,
                                                       shuffle=self.dataset_shuffle,
                                                       batch_size=self.batch_size,
                                                       seed=1337,
                                                       classes=self.class_filter)

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect all data and label by exhausting the data iterator
        and returns a single array of data and a single array of labels
        :return: ndarray of images, ndarray of labels
        """
        test_num = self.data_iterator.samples

        data, label = [], []
        for i in range((test_num // self.batch_size) + 1):
            x, y = self.data_iterator.next()
            data.append(x)
            label.append(y)

        data = np.vstack(data)
        label = np.argmax(np.vstack(label), axis=1)
        return data, label
