import numpy as np
from scipy import signal
import pandas as pd

from preprocessing import gaussian_filter, apply_thresholding
from read_data import read_data_and_labels


def read_data_as_spectrograms():
    """
    :return: A Pandas DataFrame with columns:
                red_id - Unique id of signal
                sample_rate - Sample rate of signal
                signal - The actual signal
                labels - Labels corresponding to the signal
                sample_freqs - Sample frequencies as result of signal conversion to spectrogram
                segment_times - Segment times as result of signal conversion to spectrogram
                spectrograms - Spectrogram corresponding to the signal
    """
    # Read relevant data from files
    data = read_data_and_labels()
    # Convert signals to numpy matrices for spectrogram conversion
    train_signals = data['signal'].as_matrix()
    sample_freqs = data['sample_rate'].as_matrix()
    # Convert all signals to spectrograms
    conversion_results = np.array([signal.spectrogram(xs, fs)
                                   for xs, fs in zip(train_signals, sample_freqs)])
    # Store result in original DataFrame
    data['sample_freqs'] = pd.Series(conversion_results[:, 0], index=data.index)
    data['segment_times'] = pd.Series(conversion_results[:, 1], index=data.index)
    data['spectrograms'] = pd.Series(conversion_results[:, 2], index=data.index)

    # data['spectrograms'] = data['spectrograms'].apply(lambda x: gaussian_filter(x))
    # data['spectrograms'] = data['spectrograms'].apply(lambda x: apply_thresholding(x))

    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    spec_data = read_data_as_spectrograms()

    rec_id = 4

    spectrogram = spec_data['spectrograms'].iloc[rec_id]
    sample_freq = spec_data['sample_freqs'].iloc[rec_id]
    segment_time = spec_data['segment_times'].iloc[rec_id]

    plt.pcolormesh(segment_time, sample_freq, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

