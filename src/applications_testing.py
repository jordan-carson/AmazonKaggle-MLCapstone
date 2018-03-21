import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
import cv2
import tensorflow as tf
from tqdm import tqdm
import time
from src.utillities.utility import init_logger
from src.common.CNN import optimise_f2_thresholds
from src.common.CNN import get_optimal_threshhold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical



init_logger('/Users/jordancarson/Logs', 'image_processing')

start_time = time.time()


MAIN = '/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone'
TRAIN_PATH = 'resources/train-jpg/'
TEST_PATH = 'resources/test-jpg/'
TRAIN_LABELS = 'resources/train_v2.csv'
SUBMISSION_FILE = 'resources/sample_submission_v2.csv'


x_train = []
x_test = []
y_train = []

logging.info('Reading TRAINING LABELS datasource')
df_labels = pd.read_csv(os.path.join(MAIN, TRAIN_LABELS))
logging.info('We read a training dataframe of shape' + str(df_labels.shape))

df_test = pd.read_csv(os.path.join(MAIN, SUBMISSION_FILE))


# flatten = lambda l: [item for sublist in l for item in sublist]

# print(flatten)
# labels = list(set(flatten([l.split(' ') for l in df_labels['tags'].values])))
# print(labels)
labels_list = []
for tag in df_labels.tags.values:
    labels = tag.split(' ')
    for label in labels:
        if label not in labels_list:
            labels_list.append(label)

# print(labels_list)

# labels = ['blow_down',
#           'bare_ground',
#           'conventional_mine',
#           'blooming',
#           'cultivation',
#           'artisinal_mine',
#           'haze',
#           'primary',
#           'slash_burn',
#           'habitation',
#           'clear',
#           'road',
#           'selective_logging',
#           'partly_cloudy',
#           'agriculture',
#           'water',
#           'cloudy']

# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

label_map = {'blow_down': 0,'bare_ground': 1,'conventional_mine': 2,'blooming': 3,'cultivation': 4,'artisinal_mine': 5,
            'haze': 6,'primary': 7,'slash_burn': 8,'habitation': 9,'clear': 10,'road': 11,'selective_logging': 12,
            'partly_cloudy': 13,'agriculture': 14, 'water': 15,'cloudy': 16}

for f, tags in tqdm(df_labels.values[:18000], miniters=1000):
    train_img = cv2.imread(os.path.join(MAIN, TRAIN_PATH) + '{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(train_img, (32, 32)))
    y_train.append(targets)

for f, tags in tqdm(df_test.values[:18000], miniters=1000):
    img = cv2.imread(os.path.join(MAIN, TEST_PATH) + '{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (32, 32)))

y_train = np.array(y_train, np.uint8)

x_train = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.

print("x_test shape: " + str(x_test.shape))
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))


prediction = np.random.rand(18000, 17)
true_label = np.random.rand(18000, 17) > 0.5

start = time.time()
t1 = get_optimal_threshhold(true_label, prediction)
print(time.time() - start)
start = time.time()

t2 = optimise_f2_thresholds(true_label, prediction)
print(time.time() - start)



# from src.common.CNN import create_model_vgg16
#
#
# model = create_model_vgg16((128, 128, 3))
