import numpy as np
import keras as ks
from scipy import signal
import pandas as pd

from read_data import read_data_and_labels, read_species_list

if __name__ == '__main__':
    # Read relevant data from files
    data = read_data_and_labels()
    species = read_species_list()

    # Obtain data dimensions
    number_of_signals = len(data)
    number_of_species = len(species)

    # Convert signals to numpy matrices for spectrogram conversion
    train_signals = data['signal'].as_matrix()
    sample_freqs = data['sample_rate'].as_matrix()

    # Convert all signals to spectrograms
    conversion_results = np.array([signal.spectrogram(xs, fs) for xs, fs in zip(train_signals, sample_freqs)])

    # Store result in original DataFrame
    data['sample_freqs'] = pd.Series(conversion_results[:, 0], index=data.index)
    data['segment times'] = pd.Series(conversion_results[:, 1], index=data.index)
    data['spectrograms'] = pd.Series(conversion_results[:, 2], index=data.index)

    # Normalize labels to match NN output
    train_labels_norm = np.zeros(shape=(number_of_signals, number_of_species))
    for i in range(number_of_signals):
        labels = data['labels'].iloc[i]
        for j in range(len(labels)):
            train_labels_norm[i][j] = 1

    # Obtain spectrogram dimensions
    spectrogram_shape = data['spectrograms'].iloc[0].shape
    print(spectrogram_shape)

    # Define dummy model
    model = ks.Sequential()
    model.add(ks.layers.Dense(64, activation='relu', input_shape=spectrogram_shape))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(number_of_species, activation='relu'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Convert spectrograms to numpy matrix of correct dimensions TODO -- more efficiently!!!
    spectrograms = data['spectrograms'].as_matrix()
    fit_data = np.zeros(shape=(number_of_signals, spectrogram_shape[0], spectrogram_shape[1]))
    for i in range(number_of_signals):
        for j in range(spectrogram_shape[0]):
            for k in range(spectrogram_shape[1]):
                fit_data[i][j][k] = spectrograms[i][j][k]

    print(fit_data.shape)

    # Fit the model on the data
    model.fit(fit_data, train_labels_norm, epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate the model



    # model = ks.Sequential()
    #
    # model.add(ks.layers.Dense(64, activation='relu', input_shape=(311, 513)))
    #
    # model.add(ks.layers.Flatten())
    #
    # model.add(ks.layers.Dense(19, activation='relu'))

    # model.add(ks.layers.Dense(number_of_species, activation='relu', input_shape=(311, 19)))

    # model.add(ks.layers.Lambda(lambda x: print(x)))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])
    #
    # print(train_labels_norm.shape)
    # model.fit(stfts, train_labels_norm, epochs=5, batch_size=32, validation_split=0.1)
    #
    # print(model.summary())

