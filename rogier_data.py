import csv

import numpy as np
import scipy
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from skimage.filters.thresholding import threshold_otsu, threshold_local, threshold_li, threshold_minimum, \
    threshold_mean, threshold_niblack, try_all_threshold

from skimage.filters import rank

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



def subtract_spectrogram(a: np.array, b):
    c = np.zeros(a.shape)
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            if a[x][y] > b[x][y]:
                c[x][y] = a[x][y] - b[x][y]
    return c

def preprocess_wav(wav, sr, name, plot=False, time=False, plotEnd = False):
    f, t, Sxx = signal.spectrogram(wav, sr)
    to_remove =[]
    newF = []
    # newSxx = newSxx > threshold_otsu(newSxx)
    # print(Sxx.shape)
    # print(f.shape)
    for freqI in range(len(f)):
        if f[freqI] <= 1600:
            to_remove.append(freqI)
        else:
            newF.append(f[freqI])

    # print(to_remove)
    newSxx = np.delete(Sxx, to_remove, axis=0)
    # print(newSxx.shape)
    # newSxx = np.zeros(shape=Sxx.shape)
    # for x in range(Sxx.shape[0]):
    #     for y in range(Sxx.shape[1]):
    #         if f[x] > 1500:
    #             newSxx[x, y] = Sxx[x,y]
    if plot:
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(name)
        plt.savefig("1.png")
        plt.clf()
        plt.pcolormesh(t, newF, newSxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(name)
        plt.savefig("2.png")

    newSxx = gaussian_filter(newSxx, 3)


    if plot:
        plt.pcolormesh(t, newF, newSxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(name)
        plt.savefig("3.png")
    # newSxx = newSxx > np.percentile(newSxx, 95)


    # newSxx = rank.gradient(newSxx, disk(5))
    # newSxx = scipy.ndimage.binary_closing(scipy.ndimage.binary_opening(newSxx))
    # newSxx = newSxx > np.percentile(newSxx, 90)
    if plot or plotEnd:
        plt.pcolormesh(t, newF, newSxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(name)
        plt.savefig("4.png")
    if time:
        newSxx = np.transpose(newSxx)
    newSxx = np.reshape(newSxx, (newSxx.shape[0], newSxx.shape[1], 1))
    return newSxx

    # return np.transpose(np.array(Sxx))


def get_training_data(time=False):
    result = []
    train_files = get_training_filenames()
    i =0
    for file in train_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav')
        result.append(preprocess_wav(wav, samplerate, file, time=time))
        i += 1
        # if (i == 1):
        #     quit()

    return np.array(result)

get_training_data()

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
    features = get_training_data(time=True)
    return np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))

def get_3d_training_data():
    return get_training_data()


def get_test_data(time=False):
    result = []
    test_files = get_test_filenames()
    i = 0
    for file in test_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav')
        result.append(preprocess_wav(wav, samplerate, file, time=time, plot=i==2))
        i += 1
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
    features = get_test_data(time=True)
    return np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))

def get_3d_test_data():
    return get_test_data()




def get_training_labels():
    labels = get_labels()
    training_ids = get_training_ids()
    return np.array([labels.get(x) for x in training_ids])


def get_class_weights():
    labels = get_training_labels()
    labels = labels.flatten()
    unique = np.unique(labels)
    weights = compute_class_weight('balanced', unique, labels)
    class_weights = dict()
    for k in range(len(unique)):
        class_weights[unique[k]] = weights[k]
    return class_weights



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

# get_test_data()
# get_training_data()

# train_l = training_label_variation()
# print(train_l, sum(train_l))
#
# test_l = test_label_variation()
# print(test_l, sum(test_l))