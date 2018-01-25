import numpy as np
from scipy.ndimage import gaussian_filter as _gaussian_filter
from skimage.filters import threshold_otsu

from data_info import get_number_of_species

'''
    Filters
'''


def gaussian_filter(x):
    """
    Apply a gaussian filter on a spectrogram
    :param x: The spectrogram
    :return: a filtered spectrogram
    """
    return _gaussian_filter(x, 3)


def apply_thresholding(x):
    """
    Apply otsu thresholding to a spectrogram
    :param x: The spectrogram
    :return: a filtered spectrogram
    """
    return x > threshold_otsu(x)


'''
    Normalization
'''


def normalize_data(data):
    """
    Normalize the input data to an acceptable range
    :param data: The data to be normalized
    :return:
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def normalize_labels(labels):
    """
    Transform all multi-class labels to vectors of 1 and 0 (not one-hot)
    :param labels: The labels to be normalized
    :return: a numpy matrix of normalized labels
    """
    number_of_labels = len(labels)
    number_of_species = get_number_of_species()
    labels_norm = np.zeros(shape=(number_of_labels, number_of_species))
    for i in range(number_of_labels):
        for label in labels[i]:
            labels_norm[i][label] = 1
    return labels_norm


'''
    Other
'''


def copy_to_matrix(data):
    """
    Copy a numpy array of regular arrays to a numpy matrix of corresponding shape
    :param data: A numpy array containing regular arrays of the same sizes
    :return: A numpy matrix with all the data and correct shape
    """
    spectrogram_shape = data[0].shape
    fit_data = np.zeros(shape=(len(data), spectrogram_shape[0], spectrogram_shape[1]))
    for i in range(len(data)):
        for j in range(spectrogram_shape[0]):
            for k in range(spectrogram_shape[1]):
                fit_data[i][j][k] = data[i][j][k]
    return fit_data


def filter_to_matrix(data):
    """
    Copy the data to a new numpy matrix while applying filters
    :param data: The data to be filtered and copied
    :return: a new matrix containing the filtered data
    """
    spectrogram_shape = data[0].shape
    fit_data = np.zeros(shape=(len(data), spectrogram_shape[0], spectrogram_shape[1]))
    for i in range(len(data)):
        # filtered_spectrogram = apply_thresholding(gaussian_filter(data[i]))
        filtered_spectrogram = gaussian_filter(data[i])
        for j in range(filtered_spectrogram.shape[0]):
            for k in range(filtered_spectrogram.shape[1]):
                fit_data[i][j][k] = filtered_spectrogram[j][k]
    return fit_data


if __name__ == '__main__':

    ys = [[1, 2],
          [4, 5],
          [4, 1, 2],
          [],
          [1]
          ]

    print(normalize_labels(ys))
