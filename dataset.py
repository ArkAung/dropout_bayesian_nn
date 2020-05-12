from tensorflow.keras.preprocessing.image import ImageDataGenerator

from typing import List, Dict


class Dataset:
    def __init__(self, path, data_shape, train: bool,
                 class_filter: List[str] = None, label_mapping: Dict[int, str] = None):
        self.path = path
        if train:
            self.dataset_subset = 'training'
            self.dataset_shuffle = True
        else:
            self.dataset_subset = 'validation'
            self.dataset_shuffle = False
        self.label_mapping = label_mapping
        self.data_shape = data_shape
        self.class_filter = class_filter

    @staticmethod
    def _prepare_generator(train=True):
        if train:
            return ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest',
                                validation_split=0.1
            )
        else:
            return ImageDataGenerator(rescale=1./255)

    def get_datagenerator(self, train=True):
        prepared_generator = self._prepare_generator(train)
        return prepared_generator.flow_from_directory(self.path,
                                                      target_size=self.data_shape,
                                                      class_mode='categorical',
                                                      subset=self.dataset_subset,
                                                      shuffle=self.dataset_shuffle,
                                                      seed=1337,
                                                      classes=self.class_filter)
