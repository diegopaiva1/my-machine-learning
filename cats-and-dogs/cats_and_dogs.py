import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

CATEGORIES = ['Cat', 'Dog']

def main(argv):
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    except Exception:
        X_train, X_test, y_train, y_test = create_train_test_datasets()

        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)

    try:
        model = load_model(argv[1])
    except Exception:
        conv_layers = [1, 2, 3]
        dense_layers = [0, 1, 2]
        units = [32, 64, 128]

        for c in conv_layers:
            for d in dense_layers:
                for u in units:
                    model_name = f'{c}-conv-{u}-units-{d}-dense'
                    model = create_model(model_name, X_train, X_test, y_train, y_test, c, d, u)

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

    img_width = 50
    img_height = 50

    X = []
    y = []

    for label, category in enumerate(CATEGORIES):
        path = os.path.join('PetImages/', category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_width, img_height))
                X.append(img_array)
                y.append(label)
            except Exception:
                pass # ignore broken images

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, img_width, img_height, 1)
    X_test = np.array(X_test).reshape(-1, img_width, img_height, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

def create_model(model_name, X_train, X_test, y_train, y_test, conv_layers, dense_layers, units):
    ''' Create, compile and fit a model.

    Parameters
    ----------
    `model_name`: name of the model.
    `X_train`: train dataset.
    `X_test`: test dataset.
    `y_train`: train labels.
    `y_test`: test labels.
    `conv_layers`: number of convolutional layers.
    `dense_layers`: number of dense layers.
    `units`: number of neurons in each layer.

    Returns
    ----------
    The model.
    '''

    print('<<<<<< Creating model ' + model_name + ' >>>>>>')

    model = Sequential()

    model.add(Conv2D(units, (3, 3), activation = 'relu', input_shape = X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    for c in range(conv_layers - 1):
        model.add(Conv2D(units, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())

    for d in range(dense_layers):
        model.add(Dense(units, activation = 'relu'))
        model.add(Dropout(0.20))

    # Output layer
    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), callbacks = [
        TensorBoard(log_dir = 'logs/' + model_name)
    ])

    model.save('models/' + model_name)

    return model

def predict_and_show(model, sample):
    ''' Show the sample's image and predict its class.

    Parameters
    ----------
    `model`: the model.
    `sample`: a cv2 image array.
    '''

    prediction_probs = model.predict(np.array([sample]))[0]
    label = CATEGORIES[np.argmax(prediction_probs)]
    print(f'{max(prediction_probs) * 100.0:.2f}% chance of being a {label}.')

    plt.imshow(sample, cmap = 'binary')
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
