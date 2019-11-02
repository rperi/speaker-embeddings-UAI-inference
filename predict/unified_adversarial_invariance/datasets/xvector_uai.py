from unified_adversarial_invariance.datasets.core import DatasetBase
from unified_adversarial_invariance.datasets.core import get_targets
from keras.utils import to_categorical

import os
import numpy as np


class Dataset(DatasetBase):

    def __init__(self, fold_id=None, data_dir=None):

        super(Dataset, self).__init__(fold_id=fold_id)

        self.data_dir = data_dir
        self.nb_classes = None
        self.__load_data()

    def __load_data(self):
        self.data = {}
        self.labels = {}
        for split in ['train', 'val']:
            self.data[split] = np.load(os.path.join(self.data_dir, '{}_data.npy'.format(split)))
            self.labels[split] = np.load(os.path.join(self.data_dir, '{}_labels.npy'.format(split)))
            if split == 'train':
                self.nb_classes = np.unique(self.labels[split]).shape[0]

    def get_data(self, split, bias=True, y_only=False, z_only=False,
                 embedding_dim_1=None, embedding_dim_2=None):
        mu = 0
        var = 2

        inputs = self.data[split]
        noise = np.random.normal(mu, var, [inputs.shape[0],inputs.shape[1]])

        labels = self.labels[split] #to_categorical(self.labels[split], self.nb_classes)

        targets = get_targets(inputs, labels, embedding_dim_1=embedding_dim_1, embedding_dim_2=embedding_dim_2)
        return inputs , targets
