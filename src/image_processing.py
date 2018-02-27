import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import logging
from src.utillities.utility import init_logger
from src.common.CNN import optimise_f2_thresholds
from src.common.CNN import get_optimal_threshhold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

init_logger('/Users/jordancarson/Logs', 'image_processing')

import cv2
from tqdm import tqdm

from sklearn.metrics import fbeta_score
import time

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
for strtag in df_labels.tags.values:
    labels = strtag.split(' ')
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

# start = time.time()
# t1 = get_optimal_threshhold(true_label, prediction)
# print(time.time() - start)
# start = time.time()
#
# t2 = optimise_f2_thresholds(true_label, prediction)
# print(time.time() - start)


nfolds = 3

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train = []

kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

"""
    module takes 388.4562318325043 to run
"""


for train_index, test_index in kf:
    start_time_model_fitting = time.time()

    X_train = x_train[train_index]
    Y_train = y_train[train_index]
    X_valid = x_train[test_index]
    Y_valid = y_train[test_index]

    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

    model = Sequential()
    model.add(BatchNormalization(input_shape=(32, 32, 3)))
    model.add(Conv2D(8, 1, 1, activation='relu'))
    model.add(Conv2D(16, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=2, nb_epoch=10, callbacks=callbacks,
              shuffle=True)

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=128, verbose=2)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
    print("Optimizing prediction threshold")
    print(optimise_f2_thresholds(Y_valid, p_valid))

    p_test = model.predict(x_train, batch_size=128, verbose=2)
    yfull_train.append(p_test)

    p_test = model.predict(x_test, batch_size=128, verbose=2)
    yfull_test.append(p_test)


result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels_list)
print(result)


# end_time =

print(time.time() - start_time)