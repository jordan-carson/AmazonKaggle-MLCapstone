# =======================================
# Configuration File for Capstone Project
# =======================================


optimizers = {
    'sgd': 'SGD',
    'rmsprop': 'RMSprop',
    'adagrad': 'Adagrad',
    'adadelta': 'Adadelta',
    'adam': 'Adam',
    'adamax': 'Adamax',
    'nadam': 'Nadam',
    'tfoptimizer': 'TFOptimizer'
}

loss = {
    'mse': 'mean_squared_error',
    'mae': 'mean_absolute_error',
    'categorical_crossentorpy': 'categorical_crossentropy'
}

# for _, i in optimizers.items():
#     print(i)
