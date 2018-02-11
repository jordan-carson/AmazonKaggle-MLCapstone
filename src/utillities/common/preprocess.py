import os
import numpy as np, pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
from PIL import Image


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















