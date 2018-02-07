import csv
import os

import librosa
import numpy as np
from librosa.feature import mfcc, melspectrogram, delta
from sklearn.utils.class_weight import compute_class_weight

SINGLE = False
EXTRA_DATA = True
FROM_FILE = True


def write_balanced_data():
    data_path = "/Users/Rogier/Downloads/balanced_data"
    label_path = data_path + "/labels.txt"
    samples_path = data_path + "/samples/"
    label_dict = dict()
    single_dict = dict()
    print("Reading label file....")
    with open(label_path, 'r') as label_file:
        for x in label_file:
            if '?' not in x:
                x = x.rstrip('\n')
                s = x.split(',')
                data = np.zeros((19,))
                single_class = 0
                if len(s) > 1:

                    birds = s[1:]
                    for b in birds:
                        try:
                            id = int(b)
                            data[id] = 1.
                            # if id == 8:
                            single_class = 1
                        except ValueError:
                            continue
                label_dict[s[0]] = data
                single_dict[s[0]] = single_class
    features = []
    multi_labels = []
    single_labels = []
    locs = []
    print("Processing sound files", end='', flush=True)
    i = 0
    seen = []
    num_keys = len(label_dict.keys())
    for file_name in os.listdir(samples_path):
        rec_id = file_name.split("_")[0]
        wav, samplerate = librosa.load(samples_path + file_name, sr=None, duration=10)
        spectro = preprocess_wav(wav, samplerate, name=file_name)
        features.append(spectro)
        m_label = label_dict.get(rec_id)
        if m_label is None:
            print(rec_id)
        multi_labels.append(label_dict.get(rec_id))
        single_labels.append(single_dict.get(rec_id))
        loc = file_name.split('_')[1][2:]
        locs.append(loc)
        if int(i * 100 / num_keys) % 10 == 0:
            per = int(i * 100 / num_keys)
            if per not in seen:
                seen.append(per)
                print("...{}%".format(per), end='', flush=True)
        i += 1
    print("...Done!")

    np_f = np.array(features)
    np_m = np.array(multi_labels)
    np_s = np.array(single_labels)
    np_l = np.array(locs)
    print(np_f.shape, np_l.shape, np_m.shape, np_s.shape)
    np.savez_compressed("sound_data", features=np_f, locations=np_l, multi_labels=np_m, single_labels=np_s)


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
        single_dict = dict()
        file.readline()
        for x in file:
            if '?' not in x:
                x = x.rstrip('\n')
                s = x.split(',')
                data = np.zeros((19,))
                single_class = 0
                if len(s) > 1:

                    birds = s[1:]
                    for b in birds:
                        try:
                            id = int(b)
                            data[id] = 1.
                            # if id == 8:
                            single_class = 1
                        except ValueError:
                            continue
                result[s[0]] = data
                single_dict[s[0]] = single_class
        return result, single_dict


def get_training_labels():
    labels, _ = get_labels()
    training_ids = get_training_ids()
    return np.array([labels.get(x) for x in training_ids])


def get_label_distribution(labels):
    birds = np.zeros(19)
    nothing = 0
    for sample in labels:
        birds += sample
        if sum(sample) == 0:
            nothing += 1

    # birds = np.append(birds, nothing)
    return birds, nothing


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


def preprocess_wav(wav, sr, name, label=None, plot=False, shift=0, plotEnd=False):
    S = melspectrogram(wav, sr=sr, n_mels=40)
    # print(np.var(S))
    S = np.roll(S, shift * 20, axis=1)
    melmfcc = mfcc(S=librosa.power_to_db(S), n_mfcc=60)

    return np.transpose(librosa.power_to_db(S))

    # return np.transpose(np.array(Sxx))


def piczak_preprocessing(wav, sr, shift=0):
    # resampled_wav = librosa.resample(y=wav,orig_sr=sr, target_sr=22050)
    spectrogram = melspectrogram(y=wav, sr=sr, n_mels=60, n_fft=1024)
    spectrogram = np.roll(spectrogram, shift * 20, axis=0)
    logspec = librosa.logamplitude(spectrogram)
    deltas = delta(logspec)
    return np.stack((logspec, deltas), axis=-1)


def feature_label(multi_label, single_label):
    if SINGLE:
        return single_label
    return multi_label


def get_training_data():
    if FROM_FILE:
        data = np.load('sound_data.npz')
        return data['features'], data['locations'], data['multi_labels'], data['single_labels']
    x = []
    locs = []
    y = []
    y1 = []
    files = get_file_names()
    ids = get_training_ids()
    labels, single_labels = get_labels()
    training_labels = get_training_labels()
    label_info, empty = get_label_distribution(training_labels)

    for id_i in range(len(ids)):
        if id_i % 100 == 0:
            print("At sample nr.{}".format(id_i))
        rec_id = ids[id_i]
        # wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + files.get(rec_id) + '.wav')
        wav_labels = labels.get(rec_id)
        single = single_labels.get(rec_id)
        # print(wav_labels)
        wav, samplerate = librosa.load(
            'data/mlsp_contest_dataset/essential_data/src_wavs/' + files.get(rec_id) + '.wav', sr=None, duration=10)
        spectro = preprocess_wav(wav, samplerate, files.get(rec_id), label=str(wav_labels))
        x.append(spectro)
        y.append(feature_label(wav_labels, single))
        y1.append(single)
        loc = files.get(rec_id).split('_')[0][2:]
        locs.append(loc)

        for w in range(len(wav_labels)):
            if wav_labels[w] == 1. and EXTRA_DATA:
                occurrences = label_info[w]
                copies = int(empty / occurrences) - 1
                for c in range(copies):
                    extra_feature = preprocess_wav(wav, samplerate, files.get(rec_id), label=str(wav_labels),
                                                   shift=(c + 1))
                    # extra_feature = piczak_preprocessing(wav, samplerate, shift=(c+1))
                    x.append(extra_feature)
                    # x.append(np.roll(spectro, (c + 1) * 20, axis=0))
                    y.append(feature_label(wav_labels, single))
                    y1.append(single)
                    locs.append(loc)
    return np.array(x), np.array(locs), np.array(y), np.array(y1)


def get_training_locations():
    locs = []
    train_files = get_training_filenames()
    for file in train_files:
        loc = file.split('_')[0][2:]
        locs.append(loc)
    return np.array(locs)


def get_1d_training_data(append_loc=False):
    features, labels, locations = get_training_data()
    f_shape = features.shape
    new_features = features.reshape((f_shape[0], f_shape[1] * f_shape[2]))
    if append_loc:
        new_features = np.resize(new_features, (f_shape[0], f_shape[1] * f_shape[2] + 1))
        for l in range(len(locations)):
            new_features[l][f_shape[1] * f_shape[2]] = locations[l]
    return new_features, labels


def get_2d_training_data():
    return get_training_data()


def get_3d_training_data():
    features, locs, labels, single_labels = get_training_data()
    if len(features.shape) == 4:
        return features, labels
    return np.reshape(features, newshape=(
    features.shape[0], features.shape[1], features.shape[2], 1)), labels, locs, single_labels


def get_test_data():
    result = []
    test_files = get_test_filenames()
    i = 0
    for file in test_files:
        # wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav')
        wav, samplerate = librosa.load(
            'data/mlsp_contest_dataset/essential_data/src_wavs/' + file + '.wav', sr=16000)
        # feature = piczak_preprocessing(wav, samplerate)
        feature = preprocess_wav(wav, samplerate, file)
        result.append(feature)
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
    return get_test_data()


def get_3d_test_data():
    features = get_test_data()
    if len(features.shape) == 4:
        return features
    return np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2], 1))


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
    labels, single_labels = get_labels()
    test_ids = get_test_ids()
    # return np.array([labels.get(x) for x in test_ids])
    return np.array([feature_label(labels.get(x), single_labels.get(x)) for x in test_ids]), np.array(
        [single_labels.get(x) for x in test_ids])


def get_num_classes():
    if SINGLE:
        return 1
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

# write_balanced_data()

# get_test_data()
# get_training_data()

# train_l = training_label_variation()
# print(train_l, sum(train_l))
#
# test_l = test_label_variation()
# print(test_l, sum(test_l))

# get_training_label_information()
