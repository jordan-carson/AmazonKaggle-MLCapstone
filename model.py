import numpy as np
import pandas as pd
import os
import logging
import cv2
from tqdm import tqdm
import time
from src.utillities.utility import init_logger
from src.common.CNN import optimise_f2_thresholds
from src.common.CNN import get_optimal_threshhold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
from sklearn.utils import shuffle


init_logger('/Users/jordancarson/Logs', 'image_processing')

start_time = time.time()


MAIN = '/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone'
TRAIN_PATH = 'resources/train-jpg/'
TEST_PATH = 'resources/test-jpg/'
TRAIN_LABELS = 'resources/train_v2.csv'
SUBMISSION_FILE = 'resources/sample_submission_v2.csv'
CWD = os.getcwd()
LABEL_NUMBER = 17
IMAGE_SIZE = 128


def train():
    x_train = []
    y_train = []

    df_train = pd.read_csv(os.path.join(CWD, TRAIN_LABELS))
    df_train = shuffle(df_train, random_state=0)

    for f, tags in tqdm(df_train.values, miniters=500):
        img = cv2.imread(os.path.join(CWD, TRAIN_PATH, '{}.jpg'.format(f)))
        targets = np.zeros(LABEL_NUMBER)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        flipped_img = cv2.flip(img, 1)
        rows, cols, channel = img.shape

        # regular image
        x_train.append(img)
        y_train.append(targets)

        # flipped image
        x_train.append(flipped_img)
        y_train.append(targets)

        for rot_deg in [90, 180, 270]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rot_deg, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            x_train.append(dst)
            y_train.append(targets)

            dst = cv2.warpAffine(flipped_img, M, (cols, rows))
            x_train.append(dst)
            y_train.append(targets)

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.uint8)

    kfold_train(x_train, y_train)

def predict():
    df_test = pd.read_csv(os.path.join(CWD, SUBMISSION_FILE))

    x_test = []

    for f, tags in tqdm(df_test.values, miniters=500):
        img = cv2.imread(os.path.join(CWD, TEST_PATH, '{}.jpg'.format(f)))
        x_test.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))

    x_test = np.array(x_test, np.uint8)

    result = kfold_predict(x_test)

    result = pd.DataFrame(result, columns=labels)

    thres = {'blow_down'        : 0.2,
             'bare_ground'      : 0.138,
             'conventional_mine': 0.1,
             'blooming'         : 0.168,
             'cultivation'      : 0.204,
             'artisinal_mine'   : 0.114,
             'haze'             : 0.204,
             'primary'          : 0.204,
             'slash_burn'       : 0.38,
             'habitation'       : 0.17,
             'clear'            : 0.13,
             'road'             : 0.156,
             'selective_logging': 0.154,
             'partly_cloudy'    : 0.112,
             'agriculture'      : 0.164,
             'water'            : 0.182,
             'cloudy'           : 0.076}

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        pred_tag = []
        for k, v in thres.items():
            if (a[k][i] >= v):
                pred_tag.append(k)
        preds.append(' '.join(pred_tag))

    df_test['tags'] = preds
    df_test.to_csv('sub.csv', index=False)


def amazon_sequential_custom_build(input_shape = (128, 128), weight_path=None):
    """

    :param input_shape:
    :return:
    """
    custom_model = Sequential()
    custom_model.add(BatchNormalization(input_shape=input_shape))

    custom_model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    custom_model.add(Conv2D(32, (3, 3), activation='relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))
    custom_model.add(Dropout(0.25))

    custom_model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    custom_model.add(Conv2D(64, (3, 3), activation='relu'))
    custom_model.add((MaxPooling2D(pool_size=(2, 2))))
    custom_model.add(Dropout(0.25))

    custom_model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    custom_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))
    custom_model.add(Dropout(0.25))

    custom_model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    custom_model.add(Conv2D(256, (3, 3), activation='relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))
    custom_model.add(Dropout(0.25))

    custom_model.add(Flatten())
    custom_model.add(Dense(512, activation='relu'))
    custom_model.add(BatchNormalization())
    custom_model.add(Dropout(0.5))
    custom_model.add(Dense(17, activation='sigmoid'))
    if (weight_path!=None):
        if os.path.isfile(weight_path):
            custom_model.load_weights(weight_path)
    return custom_model


def amazon_model_vgg19(input_shape=(128, 128, 3), weight_path=None):
    from keras.applications.vgg19 import VGG19

    base_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))
    if (weight_path!=None):
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    return model


def amazon_model_vgg16(input_shape=(128, 128, 3), weight_path=None):
    from keras.applications.vgg16 import VGG16

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))
    if (weight_path!=None):
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    return model


def kfold_train(x_train, y_train, n_folds=3, batch_size=128):
    model = amazon_sequential_custom_build()

    kfold = KFold(len(y_train), n_folds=n_folds, shuffle=False, random_state=1)
    num_fold = 0

    for train_index, test_index in kfold:

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, n_folds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        weight_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        epochs_arr = [60, 15, 15]
        learn_rates = [0.001, 0.0001, 0.00001]

        for learn_rate, epochs in zip(learn_rates, epochs_arr):
            optimizer = optimizers.Adam(lr=learn_rate)
            model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                         ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]

            model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                      batch_size=batch_size, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)

        p_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.18, beta=2, average='samples'))


def kfold_predict(x_test, nfolds=3, batch_size=128):
    model = amazon_sequential_custom_build()
    yfull_test = []

    for num_fold in range(1, nfolds+1):
        weight_path = os.path.join('' + 'weights_kfold_' + str(num_fold) + '.h5')

        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        p_test = model.predict(x_test, batch_size=batch_size, verbose=2)
        yfull_test.append(p_test)

    result = np.array(yfull_test[0])
    for i in range(1, nfolds):
        result += np.array(yfull_test[i])
    result /= nfolds

    return result

