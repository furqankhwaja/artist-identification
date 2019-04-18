import os
import cv2
import numpy as np
import multiprocessing
from datetime import datetime
# import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tnrange, tqdm
from numpy.random import shuffle

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import Sequence


class MyGenerator(Sequence):
    def __init__(self, list_files, list_labels, batch_size=8, im_height=256, im_width=256, train=True):
        self.list_files = list_files
        self.list_labels = list_labels
        self.batch_size = batch_size
        self.im_height = im_height
        self.im_width = im_width
        self.train = train

    def __len__(self):
        '''
        Returns the number of batches per epoch
        '''
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, idx):
        '''
        Gets data for each batch
        '''
        batch_x = self.list_files[idx *
                                  self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.list_labels[idx *
                                   self.batch_size: (idx + 1) * self.batch_size]
        if self.train:
            # Augment data
            preprocessed_batch = np.array(
                [self.preprocess(file_name) for file_name in batch_x])
            noisy_batch = np.array([self.add_noise(image)
                                    for image in preprocessed_batch])
            flip_up_batch = np.array([self.flip_up_down(image)
                                      for image in preprocessed_batch])
            flip_hor_batch = np.array([self.flip_hor(image)
                                       for image in preprocessed_batch])
            batch_images = np.concatenate(
                (preprocessed_batch, noisy_batch, flip_up_batch, flip_hor_batch))
            batch_labels = np.concatenate(
                (np.array(batch_y), np.array(batch_y), np.array(batch_y), np.array(batch_y)))
            shuffle_files(batch_images, batch_labels)
        else:
            # No data augmentation
            batch_images = np.array(
                [self.preprocess(file_name) for file_name in batch_x])
            batch_labels = np.array(batch_y)

        return batch_images, batch_labels

    def preprocess(self, file):
        '''
        Load and preprocess image
        Resize to (im_height, im_width)
        '''
        img = load_img(file)
        x_img = img_to_array(img) / 255
        x_img = cv2.resize(x_img, (self.im_height, self.im_width))
        return x_img

    def add_noise(self, image):
        '''
        Add gaussian noise to image
        '''
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    def flip_up_down(self, image):
        '''
        Flip array in the up/down direction
        '''
        return np.flipud(image)

    def flip_hor(self, image):
        '''
        Flip array in the left/right direction
        '''
        return np.fliplr(image)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    '''
    1 Convolutional block

    Arguments:
        input_tensor {tensor} -- input to convolutional block
        n_filters {int} -- number of filters for Conv2D

    Keyword Arguments:
        kernel_size {int} -- kernel size of filters (default: {3})
        batchnorm {bool} -- Batch Normalization flag (default: {True})
    '''
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def my_model(input_img, n_filters=8, dropout=0.2, batchnorm=True):
    '''
    Model definition

    Arguments:
        input_img {numpy array} -- input image

    Keyword Arguments:
        n_filters {int} -- number of filters for conv2d_block (default: {8})
        dropout {float} -- Dropout rate (default: {0.2})
        batchnorm {bool} -- Batch Normalization flag (default: {True})

    Returns:
        Keras model
    '''
    c1 = conv2d_block(input_img, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    flat1 = Flatten()(p4)
    dense = Dense(1024, activation='sigmoid')(flat1)
    dense = Dense(256, activation='sigmoid')(dense)
    outputs = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def list_all_files(path):
    '''
    Gets path names for all paintings

    Arguments:
        path  -- Source directory containing all paintings

    Returns:
        list_files -- List containing the file/path names of paintings
    '''
    list_files = []
    list_artists = []
    for artist in next(os.walk(path))[1]:
        print('Getting list of {} images...'.format(artist))
        file_path = path + artist + '/'
        ids = next(os.walk(file_path))[2]
        for _, id_ in tqdm(enumerate(ids), total=len(ids)):
            list_files.append(file_path + id_)
            list_artists.append(artist)
    return list_files, list_artists


def shuffle_files(list_files, list_artists):
    '''
    Randomly shuffles the list of file names and list of artists in same order

    Arguments:
        list_files -- List containing the file names of paintings
        list_artists -- List containing the artist names of paintings
    '''
    rng_state = np.random.get_state()
    shuffle(list_files)
    np.random.set_state(rng_state)
    shuffle(list_artists)


def create_labels(list_artists):
    '''
    Creates labels for artist names
    Picasso = 1
    Van Gogh = 0

    Arguments:
        list_artists -- List containing the artist names of paintings

    Returns:
        list_labels -- List containing the artist labels of paintings
    '''
    list_labels = np.zeros(np.shape(list_artists))
    list_labels[np.array(list_artists) == 'Picasso'] = 1
    return list_labels


def create_datasets(list_files, list_labels, num_train_samples, num_valid_samples):
    '''
    Creates the train, validation and test datasets

    Arguments:
        list_files -- List containing the file names of paintings
        list_labels -- List containing the corresponding artist labels of paintings
        num_train_samples -- Number of training samples to be taken
        num_valid_samples -- Number of validation samples to be taken

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test -- the train, validation
            and test file names and corresponding artist labels
    '''
    X_train = list_files[:num_train_samples]
    X_valid = list_files[num_train_samples:num_train_samples+num_valid_samples]
    X_test = list_files[num_train_samples + num_valid_samples:]
    y_train = list_labels[:num_train_samples]
    y_valid = list_labels[num_train_samples:num_train_samples +
                          num_valid_samples]
    y_test = list_labels[num_train_samples + num_valid_samples:]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":

    num_workers = multiprocessing.cpu_count()
    src_dir = './artist_dataset/'
    num_train_samples = 1440
    num_valid_samples = 144
    im_height = 256
    im_width = 256
    num_epochs = 15
    num_filters = 8
    batch_size = 8
    dropout_rate = 0.10
    bool_multiprocessing = True
    Train = False

    # Create model
    input_img = Input((im_width, im_height, 3), name='img')
    model = my_model(input_img, n_filters=num_filters,
                     dropout=dropout_rate, batchnorm=True)
    model.compile(optimizer=Adam(), loss='binary_crossentropy',
                  metrics=["accuracy"])
    # model.summary()

    if Train:
        # Preprocessing
        list_files, list_artists = list_all_files(src_dir)
        shuffle_files(list_files, list_artists)
        list_labels = create_labels(list_artists)
        X_train, X_valid, X_test, y_train, y_valid, y_test = create_datasets(
            list_files, list_labels, num_train_samples, num_valid_samples)

        # Save datasets
        np.save('X_train.npy', X_train)
        np.save('X_valid.npy', X_valid)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_valid.npy', y_valid)
        np.save('y_test.npy', y_test)

        # Create data generator objects
        training_generator = MyGenerator(
            X_train, y_train, batch_size=batch_size, im_height=im_height, im_width=im_width, train=True)
        validation_generator = MyGenerator(
            X_valid, y_valid, batch_size=batch_size, im_height=im_height, im_width=im_width, train=False)
        test_generator = MyGenerator(
            X_test, y_test, batch_size=batch_size, im_height=im_height, im_width=im_width, train=False)

        # Callbacks
        callbacks = [EarlyStopping(patience=5, verbose=1),
                     ReduceLROnPlateau(factor=0.1, patience=3,
                                       min_lr=0.000001, verbose=1),
                     ModelCheckpoint('model_best.hdf5', verbose=1,
                                     save_best_only=True, save_weights_only=True)]

        # Train model
        print('\nTraining model...\n')
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=bool_multiprocessing,
                            workers=num_workers, epochs=num_epochs, callbacks=callbacks, verbose=1)

        print('\nTraining Completed...\n')

    else:
        # Load datasets
        X_train = np.load('X_train.npy')
        X_valid = np.load('X_valid.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_valid = np.load('y_valid.npy')
        y_test = np.load('y_test.npy')

        # Create data generator objects
        training_generator = MyGenerator(
            X_train, y_train, batch_size=batch_size, im_height=im_height, im_width=im_width, train=False)
        validation_generator = MyGenerator(
            X_valid, y_valid, batch_size=batch_size, im_height=im_height, im_width=im_width, train=False)
        test_generator = MyGenerator(
            X_test, y_test, batch_size=batch_size, im_height=im_height, im_width=im_width, train=False)

        # Load model weights
        model.load_weights('model_best.hdf5')

        # Predict
        print('\nPredicting on Training Data...')
        train_scores = model.evaluate_generator(training_generator, verbose=1)
        print('Predicting on Validation Data...')
        valid_scores = model.evaluate_generator(
            validation_generator, verbose=1)
        print('Predicting on Test Data...')
        test_scores = model.evaluate_generator(test_generator, verbose=1)

        # Display results
        print('\nTrain accuracy = {} %'.format(train_scores[1]*100))
        print('Validation accuracy = {} %'.format(valid_scores[1]*100))
        print('Test accuracy = {} %\n'.format(test_scores[1]*100))
