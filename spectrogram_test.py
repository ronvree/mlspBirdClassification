import tensorflow as tf
import keras as ks
import numpy as np

from scipy import signal

from read_data import read_data_and_labels, read_species_list

train_data, train_labels, test_data, test_labels = read_data_and_labels()

species = read_species_list()
number_of_species = len(species)

train_signals = train_data['signal'].as_matrix()
sample_freqs = train_data['sample_rate'].as_matrix()

number_of_signals = len(train_signals)
signal_length = len(train_signals[0])

input_shape = (number_of_signals, signal_length)

signals = tf.placeholder(tf.float32, shape=input_shape)

stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,
                               fft_length=1024)

# Normalize labels
train_labels_norm = np.zeros(shape=(len(train_labels), number_of_species))
for i in range(len(train_labels)):
    labels = train_labels['labels'].iloc[i]
    for j in range(len(labels)):
        train_labels_norm[i][j] = 1

spectrograms = np.array([signal.spectrogram(xs, fs) for xs, fs in zip(train_signals, sample_freqs)])

print(spectrograms)

with tf.Session() as session:
    input_matrix = np.zeros(shape=input_shape)
    for i in range(number_of_signals):
        series = train_signals[i]
        for j in range(signal_length):
            input_matrix[i][j] = series[j]

    print(input_matrix.shape)
    session.run(stfts, feed_dict={signals: input_matrix})
    print(stfts)

    model = ks.Sequential()

    model.add(ks.layers.Dense(64, activation='relu', input_shape=(311, 513)))

    model.add(ks.layers.Flatten())

    model.add(ks.layers.Dense(number_of_species, activation='relu'))

    # model.add(ks.layers.Dense(number_of_species, activation='relu', input_shape=(311, 19)))

    # model.add(ks.layers.Lambda(lambda x: print(x)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    print(train_labels_norm.shape)
    model.fit(stfts, train_labels_norm, epochs=5, batch_size=32, validation_split=0.1)

    print(model.summary())


