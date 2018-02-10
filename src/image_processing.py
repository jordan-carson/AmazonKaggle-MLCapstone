import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time

MAIN = '/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone'
TRAIN_PATH = 'resources/train-jpg/'
TEST_PATH = 'resources/test-jpg/'
TRAIN_LABELS = 'resources/train_v2.csv'
SUBMISSION_FILE = 'resources/sample_submission_v2.csv'


x_train = []
x_test = []
y_train = []

df_train = pd.read_csv(os.path.join(MAIN, TRAIN_LABELS))
df_test = pd.read_csv(os.path.join(MAIN, SUBMISSION_FILE))

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']

label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}
for f, tags in tqdm(df_train.values[:18000], miniters=1000):
    img = cv2.imread(os.path.join(MAIN, TRAIN_PATH) + '{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)


# for f, tags in tqdm(df_test.values, miniters=1000):
#     img = cv2.imread(os.path.join(MAIN, TEST_PATH) + '{}.jpg'.format(f))
#     x_test.append(cv2.resize(img, (32, 32)))

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255.
# x_test = np.array(x_test, np.float32) / 255.

print(x_train.shape)
print(y_train.shape)