import os
import struct
import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.keras as keras
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def load_mnist(path, kind='train'):
    '''Load MNIST data'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        '''magic is the magic number that specifies the file protocol
           n is the number of items from the file buffer. > signals
           big endian ordering, and I is an unsigned integer.'''
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8) #Class labels (integers 0-9)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)

    return images, labels

X_train, y_train = load_mnist('./mnist/', kind='train')
print('Rows: %d, Columns: %d' %(X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('./mnist/', kind='t10k')
print('Rows: %d, Columns: %d' %(X_test.shape[0], X_test.shape[1]))

## mean centering and normalization:
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

np.random.seed(123)
tf.set_random_seed(123)

y_train_onehot = keras.utils.to_categorical(y_train)

print('\nFirst 3 labels: ', y_train[:3])
print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])

def create_model():
    model= keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'))

    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'))

    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='softmax'))

    sgd_optimizer = keras.optimizers.SGD(lr=0.0005, decay=1e-7, momentum=.8)
    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
    return model

model = create_model()
history = model.fit(X_train_centered, y_train_onehot, batch_size=64, epochs=50,
                    verbose=1, validation_split=0.1)

y_train_pred = model.predict_classes(X_train_centered, verbose=0)

y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('\nTraining accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('\nTest accuracy: %.2f%%' % (test_acc * 100))
print('\n')
