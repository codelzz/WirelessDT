import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class WindowSplitter:
    """
        Window splitter - given a time series data set, the splitter
        can split it based on window size, feature columns
    """
    def __init__(self, input_width, label_width, shift, columns, input_columns=None, label_columns=None):
        """
        :param input_width: input window size
        :param label_width: label(output) window size
        :param shift: control offset between input and output on time axis
        :param columns: all feature column names
        :param input_columns: input column names
        :param label_columns: label column names
        """
        self.input_width, self.label_width = input_width, label_width
        self.shift = shift
        self.columns = columns
        self.input_columns, self.label_columns = input_columns, label_columns
        # the total window size
        self.total_window_size = input_width + shift
        # init column index
        self.init_columns_indices()

    def init_columns_indices(self):
        # all column index
        self.column_indices = {name: i for i, name in enumerate(self.columns)}
        # input columns indices
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        if self.input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(self.input_columns)}
        # label columns indices
        self.label_slice = slice(self.total_window_size - self.label_width, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}

    def __call__(self, data):
        """
        :param data: shape(batch_size，seq_len, feature_dim)
        :return inputs: input series
        :return labels: output series
        """
        # 1.
        inputs, labels = data[:, self.input_slice, :], data[:, self.label_slice, :]
        # 2.
        if self.input_columns is not None:
            inputs = tf.stack([inputs[:, :, self.column_indices[name]] for name in self.input_columns], axis=-1)
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        # 3.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels


class DatasetGenerator(WindowSplitter):
    """
        Dataset Generator - generate time serise dataset
    """
    def __init__(self, data, input_width, label_width, shift,
                 columns, input_columns, label_columns, batch_size,
                 partition=(0.7, 0.2, 0.1), dtype='float32'):
        """
        :param data: the data before split，shape(seq_len, dim)
        :param input_width: input window size (input series size)
        :param label_width: label window size (label series size)
        :param shift: offset
        :param columns: list of column name
        :param input_columns: list of input feature column names
        :param label_columns: list of label feature column names
        :param batch_size: batch size
        :param partition: proportion of training/validation/testing dataset
        """
        super().__init__(input_width=input_width, label_width=label_width, shift=shift,
                         columns=columns, input_columns=input_columns, label_columns=label_columns)
        self.batch_size = batch_size
        self.partition = partition

        data = data.to_numpy().astype(dtype)

        self.full_ds = self.build_dataset(data)
        size = self.full_dataset_size();
   
        train_size = int(partition[0] * size)
        test_size  = int(partition[2] * size)
        
        self.train_ds = self.full_ds.take(train_size)
        self.test_ds = self.full_ds.skip(train_size)
        self.valid_ds = self.test_ds.skip(test_size)
        self.test_ds = self.test_ds.take(test_size)

    def build_dataset(self, data):
        """ Construct Dataset, split input data to training set/ validation set and test set

        Algorithm：
            1. call tensorflow api with batch_size=1 as the first time construct
            2. use map function，split each data point from dataset to input/label series
            3. then re-batch data set to desired size

        :param data: shape(seq_len, feature_dim)
        :return: batched dataset
        """
        # 1.
        dataset = timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                sequence_stride=1, shuffle=True, batch_size=1)
        # check the size of dataset
        # print("dataset size before batch", dataset.cardinality().numpy())
        # 2.
        dataset = dataset.map(self)
        # 3.
        return dataset.unbatch().batch(self.batch_size)
    
    def full_dataset_size(self):
        count = 0
        for item in self.full_ds:
            count += 1
        return count

    def __repr__(self):
        return '\n'.join([
            f'<DatasetGenerator>',
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input column name(s): {self.input_columns}',
            f'Label column name(s): {self.label_columns}',
            f'Batch size: {self.batch_size}',
            f'Partition: Train(%.2f), Validation(%.2f), Test(%.2f)' % self.partition,
        ])