import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import TensorBoard

def main():
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    except:
        X_train, X_test, y_train, y_test = create_train_test_datasets()

        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

    try:
        model = load_model('models/2-dense')
    except:
        model = create_model('2-dense', X_train, X_test, y_train, y_test)

    predict_and_show(model, random.choice(X_test))

def create_train_test_datasets():
    ''' Create train-test datasets.

    Returns
    ----------
    `X_train`: train dataset.
    `X_test`: test dataset.
    `y_train`: train labels.
    `y_test`: test labels.
    '''

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = normalize(X_train, axis = 1)
    X_test = normalize(X_test, axis = 1)

    return X_train, X_test, y_train, y_test

def create_model(model_name, X_train, X_test, y_train, y_test):
    ''' Create, compile and fit a model.

    Parameters
    ----------
    `model_name`: name of the model.
    `X_train`: train dataset.
    `X_test`: test dataset.
    `y_train`: train labels.
    `y_test`: test labels.

    Returns
    ----------
    The model.
    '''

    model = Sequential()

    # transform into 1D data
    model.add(Flatten())

    # hidden layers
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))

    # output layer
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs = 5, validation_data = (X_test, y_test), callbacks = [
        TensorBoard(log_dir = 'logs/' + model_name)
    ])

    model.save('models/' + model_name)

    return model

def predict_and_show(model, sample):
    ''' Show the sample's image and predict its class.

    Parameters
    ----------
    `model`: the model.
    `sample`: an image array.
    '''

    prediction_probs = model.predict(np.array([sample]))[0]
    prediction = np.argmax(prediction_probs)

    print(f'My guess is... \033[1m{prediction}\033[0;0m.')

    plt.imshow(sample, cmap = 'binary')
    plt.show()

if __name__ == "__main__":
    main()
