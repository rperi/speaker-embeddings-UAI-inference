from unified_adversarial_invariance.datasets.core import DatasetBase
from unified_adversarial_invariance.datasets.core import get_targets
from keras.utils import to_categorical
#from . import mnist_rot
import numpy
import pandas as pd
import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FREQ_DIM = 80

class Dataset(DatasetBase):
    def __init__(self, fold_id=None):
        super(Dataset, self).__init__()
        self.train_size = None
        self.val_size = None
        self.nb_classes = None
        self.speaker_map = None
        self.generator_steps = {
            'train': None,
            'valid': None
        }

    # Map labels from speaker ids to integers
    def __prepare_data(self):
        df = pd.read_csv(self.train_file)
        uniq_spkrs = numpy.unique(df['speaker'].values)
        self.train_size = df['speaker'].values.shape[0]
        self.nb_classes = uniq_spkrs.shape[0]
        self.speaker_map = {sp: idx for idx, sp in enumerate(uniq_spkrs)}

        self.generator_steps = {
            'train': int(self.train_size / self.batch_size),
            'valid': int(self.val_size/self.batch_size)
        }

    def __get_data(self, path_id, label, start):
        if isinstance(path_id, str):  # if batch_size==1
            return numpy.load(path_id)[start: start + self.segment_length, :], to_categorical(label, self.nb_classes)
        else:
            dummy = numpy.load(path_id[0])
            freq_dim = dummy.shape[1]
            out_array = numpy.empty((self.batch_size, self.segment_length, freq_dim, 1))
            for idx in range(path_id.shape[0]):
                a = numpy.load(path_id[idx])
                a = a.reshape(a.shape[0], freq_dim, 1)
                out_array[idx, :] = a[start[idx]: start[idx] + self.segment_length, :]
            return out_array, to_categorical(label, self.nb_classes)

    def get_generator(self, split, bias=True, y_only=False, z_only=False,
                      embedding_dim_1=None, embedding_dim_2=None):
        self.__prepare_data()

        if split == 'train':
            df = pd.read_csv(self.train_file)
        else:
            df = pd.read_csv(self.val_file)

        file_names = []
        for file_id in df['file_name'].values:
            sp_id = file_id.split('_')[0]
            file_names.append(os.path.join(self.feat_root, sp_id, file_id))

        sp_ids = [int(self.speaker_map[x]) for x in df['speaker'].values]
        start_offsets = list(df['start'].values)

        ds = tf.data.Dataset.from_tensor_slices((file_names, sp_ids, start_offsets))\
            .repeat()\
            .shuffle(10*self.train_size)\
            .batch(self.batch_size)
        ds = ds.map(lambda path_id, label, start: tf.py_func(self.__get_data, [path_id, label, start], [tf.double, tf.double]))
        d_iterator = ds.make_one_shot_iterator()

        tf.InteractiveSession().close()

        sess = tf.InteractiveSession()

        a = d_iterator.get_next()
        while True:
            inputs, labels = sess.run(a)
            inputs = numpy.transpose(inputs, (0, 2, 1, 3))
            #numpy.reshape(inputs, (self.batch_size, FREQ_DIM, self.segment_length, 1))
            targets = get_targets(inputs, labels, embedding_dim_1=embedding_dim_1, embedding_dim_2=embedding_dim_2)
            yield inputs, targets
