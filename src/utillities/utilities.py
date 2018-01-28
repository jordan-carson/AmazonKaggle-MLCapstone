import pandas as pd
import os
# this contains image_name and tags for training set

class FileNotFoundError:
    pass


def read_file(filepath, filename, filetype):
    "module for reading different files"
    if len(filepath) > 0:
        if len(filename) > 0:
            if len(filetype) > 0:
                if filetype == 'csv':
                    try:
                        df = pd.read_csv(os.path.join(filepath, filename))

                        return df
                    except Exception as err:
                        print(err)


df_labels = read_file('/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone/resources', 'train_v2.csv', 'csv')

