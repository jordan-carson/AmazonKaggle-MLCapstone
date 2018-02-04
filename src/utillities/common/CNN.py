from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score
from keras.models import Model
import numpy as np


def create_model_vgg16(image_dimensions=(128, 128, 3)):
    input_tensor = Input(shape=image_dimensions)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_dimensions)

    bn = BatchNormalization()(input_tensor)
    x = base_model()(bn)
    x = Flatten()(x)
    output = Dense(17, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model


def fbeta(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')


