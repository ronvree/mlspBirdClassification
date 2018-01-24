import numpy as np

from data_info import get_number_of_species


def normalize_labels(labels):
    number_of_labels = len(labels)
    number_of_species = get_number_of_species()
    labels_norm = np.zeros(shape=(number_of_labels, number_of_species))
    for i in range(number_of_labels):
        for label in labels[i]:
            labels_norm[i][label] = 1
    return labels_norm


def copy_to_matrix(data):
    spectrogram_shape = data[0].shape
    fit_data = np.zeros(shape=(len(data), spectrogram_shape[0], spectrogram_shape[1]))
    for i in range(len(data)):
        for j in range(spectrogram_shape[0]):
            for k in range(spectrogram_shape[1]):
                fit_data[i][j][k] = data[i][j][k]
    return fit_data


if __name__ == '__main__':

    ys = [[1, 2],
          [4, 5],
          [4, 1, 2],
          [],
          [1]
          ]

    print(normalize_labels(ys))
