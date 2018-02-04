import os
import numpy as np, pandas as pd


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

    def load_labels(self):
        pd.read_csv(self.train_csv, sep=',')