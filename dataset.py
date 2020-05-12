from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import List, Dict, Tuple


class Dataset:
    TRAIN = 1
    VAL = 2
    TEST = 3

    def __init__(self, path: str, target_size: Tuple[int, int], dataset_type, batch_size: int,
                 class_filter: List[str] = None, label_mapping: Dict[int, str] = None):
        self.path = path
        if dataset_type == Dataset.TRAIN:
            self.dataset_subset = 'training'
            self.dataset_shuffle = True
        elif dataset_type == Dataset.VAL:
            self.dataset_subset = 'validation'
            self.dataset_shuffle = False
        elif dataset_type == Dataset.TEST:
            self.dataset_subset = None
            self.dataset_shuffle = False
        self.dataset_type = dataset_type
        self.label_mapping = label_mapping
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_filter = class_filter
        self.datagenerator = self._get_datagenerator()

    def _prepare_generator(self):
        if self.dataset_type == Dataset.TRAIN:
            return ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=40,
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

    def _get_datagenerator(self):
        prepared_generator = self._prepare_generator()
        return prepared_generator.flow_from_directory(self.path,
                                                      target_size=self.target_size,
                                                      class_mode='categorical',
                                                      subset=self.dataset_subset,
                                                      shuffle=self.dataset_shuffle,
                                                      batch_size=self.batch_size,
                                                      seed=1337,
                                                      classes=self.class_filter)
