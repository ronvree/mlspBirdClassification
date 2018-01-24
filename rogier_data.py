import csv

import numpy as np
import soundfile as sf
from scipy import signal


def get_file_names():
    with open('data/mlsp_contest_dataset/essential_data/rec_id2filename.txt') as csvfile:
        reader = csv.DictReader(csvfile)
        rec_id2filename = dict()
        for row in reader:
            rec_id2filename[row['rec_id']] = row['filename']
        return rec_id2filename


def get_training_ids():
    with open('data/mlsp_contest_dataset/essential_data/CVfolds_2.txt', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        cv_folds = dict()
        for row in reader:
            if row['fold'] in cv_folds:
                cv_folds[row['fold']].append(row['rec_id'])
            else:
                cv_folds[row['fold']] = [row['rec_id']]
        return cv_folds['0']


def get_test_ids():
    with open('data/mlsp_contest_dataset/essential_data/CVfolds_2.txt', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        cv_folds = dict()
        for row in reader:
            if row['fold'] in cv_folds:
                cv_folds[row['fold']].append(row['rec_id'])
            else:
                cv_folds[row['fold']] = [row['rec_id']]
        return cv_folds['1']


def get_labels():
    with open('data/mlsp_contest_dataset/essential_data/bag_labels.txt', 'r') as file:
        result = dict()
        file.readline()
        for x in file:
            if '?' not in x:
                x = x.rstrip('\n')
                s = x.split(',')
                data = np.zeros((19,))
                if len(s) > 1:

                    birds = s[1:]
                    for b in birds:
                        try:
                            id = int(b)
                            data[id] = 1.
                        except ValueError:
                            continue
                result[s[0]] = data
        return result


def get_roc_labels():
    with open('data/mlsp_contest_dataset/essential_data/bag_labels.txt', 'r') as file:
        result = dict()
        file.readline()
        for x in file:
            if '?' not in x:
                x = x.rstrip('\n')
                s = x.split(',')
                result[s[0]] = s[1:]
        return result


def get_training_filenames():
    filenames = get_file_names()
    training_ids = get_training_ids()
    return [filenames.get(x) for x in training_ids]


def get_test_filenames():
    filenames = get_file_names()
    test_ids = get_test_ids()
    return [filenames.get(x) for x in test_ids]


def preprocess_wav(wav, sr):
    f, t, Sxx = signal.spectrogram(wav, sr)

    # print(Sxx.shape)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # quit()
    Sxx = np.reshape(Sxx, (Sxx.shape[0], Sxx.shape[1], 1))
    return Sxx

    # return np.transpose(np.array(Sxx))


def get_training_data():
    result = []
    train_files = get_training_filenames()
    for file in train_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav')
        result.append(preprocess_wav(wav, samplerate))

    return np.array(result)

def get_training_locations():
    locs = []
    train_files = get_training_filenames()
    for file in train_files:
        loc = file.split('_')[0][2:]
        locs.append(loc)
    return np.array(locs)

def get_1d_training_data(append_loc=False):
    features = get_training_data()
    f_shape = features.shape
    new_features = np.reshape(features, (f_shape[0], f_shape[1] * f_shape[2]))
    if append_loc:
        locations = get_training_locations()
        new_features = np.resize(new_features, (f_shape[0], f_shape[1] * f_shape[2] + 1))
        for l in range(len(locations)):
            new_features[l][f_shape[1] * f_shape[2]] = locations[l]
    return new_features

def get_2d_training_data():
    features = get_training_data()
    return np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))

def get_3d_training_data():
    return get_training_data()


def get_test_data():
    result = []
    test_files = get_test_filenames()
    for file in test_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav')
        result.append(preprocess_wav(wav, samplerate))
    return np.array(result)

def get_test_locations():
    locs = []
    test_files = get_test_filenames()
    for file in test_files:
        loc = file.split('_')[0][2:]
        locs.append(loc)
    return np.array(locs)

def get_1d_test_data(append_loc=False):
    features = get_test_data()
    f_shape = features.shape
    new_features = np.reshape(features, (f_shape[0], f_shape[1] * f_shape[2]))
    if append_loc:
        locations = get_test_locations()
        new_features = np.resize(new_features, (f_shape[0], f_shape[1] * f_shape[2] + 1))
        for l in range(len(locations)):
            new_features[l][f_shape[1] * f_shape[2]] = locations[l]
    return new_features

def get_2d_test_data():
    features = get_test_data()
    return np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))

def get_3d_test_data():
    return get_test_data()







def get_training_labels():
    labels = get_labels()
    training_ids = get_training_ids()
    return np.array([labels.get(x) for x in training_ids])


def get_test_labels():
    labels = get_labels()
    test_ids = get_test_ids()
    return np.array([labels.get(x) for x in test_ids])


def get_roc_test_labels():
    labels = get_roc_labels()
    test_ids = get_test_ids()
    return np.array([labels.get(x) for x in test_ids])


def get_num_classes():
    return 19


def training_label_variation():
    labels = get_training_labels()
    zeros = 0
    ones = 0
    for sample in labels:
        for label in sample:
            if (label == 1.):
                ones += 1
            else:
                zeros += 1
    return ones, zeros


def test_label_variation():
    labels = get_test_labels()
    zeros = 0
    ones = 0
    for sample in labels:
        for label in sample:
            if label == 1.:
                ones += 1
            else:
                zeros += 1
    return ones, zeros


train_l = training_label_variation()
print(train_l, sum(train_l))

test_l = test_label_variation()
print(test_l, sum(test_l))