import tensorflow as tf
import keras as ks
import numpy as np

from read_data import read_data_and_labels

train_data, train_labels, test_data, test_labels = read_data_and_labels()

number_of_signals = len(train_data)
signal_length = len(train_data['signal'].iloc[0])

signals = tf.placeholder(tf.float32, shape=(number_of_signals, signal_length))

stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,
                               fft_length=1024)

with tf.Session() as session:
    train_signals = train_data['signal'].as_matrix()
    input_matrix = np.zeros(shape=(number_of_signals, signal_length))
    for i in range(number_of_signals):
        series = train_signals[i]
        for j in range(signal_length):
            input_matrix[i][j] = series[j]

    print(input_matrix.shape)

    session.run(stfts, feed_dict={signals: input_matrix})

    print(stfts)

