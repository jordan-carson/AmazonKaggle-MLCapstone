import os
import numpy as np, pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
from PIL import Image

from abc import ABC, abstractmethod
import pandas
import datetime

class Preprocessor:
    def __init__(self, train_dir, train_csv, test_dir, img_resize=(64,64), validation_split=0.2):
        """
        Function to process image directories for train, test and training labels.

        Parameters
        ----------
        train_dir, train_csv, test_dir, Optional: img_resize=(64,64), Optional: validation_split2
        Returns
        ----------
        initalizing the preprocessor for loading the image files

        """
        self.train_dir = train_dir
        self.train_csv = train_csv
        self.test_dir = test_dir
        self.img_resize = img_resize
        self.validation_split = validation_split

    def process_images(self, array):
        """
        module to process the images to greyscale and output an np.array
        :return: np.array
        """
        import cv2  
        
        return cv2.resize(array)


class BuildModel:

    def __init__(self, dropout):
        self.dropout = dropout


class MyDense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.units = output_dim
        super(MyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim),
                                      initializer='uniform', trainable=True)
        super(MyDense, self).build(input_shape)

    def call(self, x, **kwargs):
        """This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        """
        y = K.dot(x, self.kernel)
        return y


class AbstractBaseClass(ABC):

    @abstractmethod
    def read_file(self, full_file_path):
        return

    @abstractmethod
    def import_image(self, image_path):
        return

    @abstractmethod
    def write_file(self, folder_path):
        return
    # https://www.python-course.eu/python3_object_oriented_programming.php


class SumList(object):
    def __init__(self, this_list):
        self.mylist = this_list

    def __add__(self, other):
        new_list = [x + y for x, y in zip(self.mylist, other.mylist)]

        return SumList(new_list)

    def __repr__(self):
        return str(self.mylist)


class ReadImageData(AbstractBaseClass):

    @abstractmethod
    def read_file(self, full_file_path):
        try:
            return pandas.read_csv(full_file_path)
        except Exception as err:
            print("Read file method requires a csv." + '\n'
                  + 'Error: ' + str(err))

    @abstractmethod
    def import_image(self, image_path):
        print("Reading a image file from " + image_path)


class WriteFile:

    def __init__(self, filename, writer):
        self.fh = open(filename, 'w')
        self.formatter = writer()

    def write(self, text):
        self.fh.write(self.formatter.format(text))
        self.fh.write('\n')

    def close(self):
        self.fh.close()


class CSVFormatter:
    """module to format csv output"""
    def __init__(self):
        self.delim = ','

    def format(self, this_list):
        new_list = []
        for element in this_list:
            if self.delim in element:
                new_list.append('"{0}'.format(element))
            else:
                new_list.append(element)
        return self.delim.join(new_list)


class LogFormatter:
    """For formating log files"""
    def format(self, this_line):
        dt = datetime.datetime.now()
        date_str = dt.strftime('%Y-%m-%d %H:%M')
        return '{0}  {1}'.format(date_str, this_line)


