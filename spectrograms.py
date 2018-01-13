import numpy as np
from scipy import signal
from read_data import read_data_and_labels


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = read_data_and_labels()

    train_signals = train_data['signal'].as_matrix()
    sample_freqs = train_data['sample_rate'].as_matrix()

    spectrograms = np.array([signal.spectrogram(xs, fs) for xs, fs in zip(train_signals, sample_freqs)])


