import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from PIL import Image

from src.utillities.utility import read_file, plot_pictures

MAIN = '/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone'
TRAIN_PATH = 'resources/train-jpg/'
TRAIN_LABELS = 'resources/train_v2.csv'

df_labels = read_file(MAIN, TRAIN_LABELS, filetype='csv')
# print(df_labels.head(10))


labels = df_labels.tags.values

labels_list = []
for lbl in labels:
    labels_list.extend(lbl.split(' '))
labels_set = set(labels_list)
# print(labels_set)

df_train = df_labels.tags.str.get_dummies(' ')
df_train.insert(0, 'image_name', df_labels.image_name)
# print(df_train.head(10))

# print(df_train[list(labels_set)].sum().sort_values().plot(kind='bar'))

ordered_labels = df_train[list(labels_set)].sum().sort_values(ascending=False).index

# pandas.core.indexes.base.Index
# print(type(ordered_labels))


print(plot_pictures('primary', df_train, os.path.join(MAIN, TRAIN_PATH)))


# TODO: create a class for image resize, flatten etc.

