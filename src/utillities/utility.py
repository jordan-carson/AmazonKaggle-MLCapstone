import pandas as pd
import os
import socket
import logging
import logging.handlers
import sys
import datetime

import random
from PIL import Image
import matplotlib.pyplot as plt

# df_labels = read_file('/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone/resources', 'train_v2.csv', 'csv')

def plot_pictures(label, df_train, TRAIN_PATH):

    images = df_train[df_train[label] == 1].image_name.values

    fig , ax = plt.subplots(nrows=3, ncols=3, figsize=(8,8))
    ax = ax.flatten()

    for i in range(0,9):
        f = random.choice(images)
        img = Image.open(os.path.join(TRAIN_PATH, f + '.jpg'))
        ax[i].imshow(img)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title("{}s h:{}s w:{}s".format(f, img.height, img.width))
    plt.tight_layout()


def init_logger(log_dir, process_name, loglevel_file=20, loglevel_stdout=40):
    logging.getLogger().setLevel(logging.NOTSET)  # 0
    logging.getLogger().handlers = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    logfile_name = process_name + '_' + str(socket.gethostname()) + '_' + str(os.getenv('Username')) \
                   + '_' + timestamp + '.txt'
    logfile_path = os.path.join(log_dir, logfile_name)

    # create directory if it doesn't already exist
    try:
        os.makedirs(log_dir)
    except:
        pass

    # delete previous version of this logfile if it exists

    try:
        os.remove(logfile_name)
    except:
        pass

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # file handler
    file_h = logging.handlers.RotatingFileHandler(logfile_path, mode='w', maxBytes=50*1024*1024,
                                                  backupCount=2)
    file_h.setLevel(loglevel_file)  # 20 = info
    file_h.setFormatter(formatter)

    # stdout handler
    stdout_h = logging.StreamHandler(stream=sys.stdout)
    stdout_h.setLevel(loglevel_stdout)  # 40 = error
    stdout_h.setFormatter(formatter)
    logging.getLogger().addHandler(stdout_h)

    logging.info('Logging initialised. PID ' + str(os.getpid()))




































