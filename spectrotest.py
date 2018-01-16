import csv

import keras
import numpy as np
from sklearn import metrics
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.io.wavfile


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
    f, t, Sxx = signal.spectrogram(wav, sr, window=signal.get_window('hamming', 512))
    wav = np.zeros((f.shape[0], t.shape[0], 3))
    for freqI in range(f.shape[0]):
        for timeI in range(t.shape[0]):
            wav[freqI][timeI] = np.array([f[freqI], t[timeI], Sxx[freqI][timeI]])
    return wav

    # return np.transpose(np.array(Sxx))

def get_training_data():
    result = []
    train_files = get_training_filenames()
    for file in train_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/'+ file + '.wav')

        result.append(preprocess_wav(wav, samplerate))
    return np.array(result)

def get_test_data():
    result = []
    test_files = get_test_filenames()
    for file in test_files:
        wav, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/'+ file + '.wav')

        result.append(preprocess_wav(wav, samplerate))
    return np.array(result)

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

#


labels = get_training_labels()

print(labels)

features = get_training_data()

new_features = []
new_labels = []

for wav in range(labels.shape[0]):
    feat = features[wav]

    for b in range(labels.shape[1]):
        if labels[wav][b] > 0:
            lab = np.zeros(labels[wav].shape)
            lab[b] = 1.
            new_labels.append(lab)
            new_features.append(feat)

new_features = np.array(new_features)
new_labels = np.array(new_labels)


input_shape = (new_features.shape[1], new_features.shape[2], new_features.shape[3])
num_classes = 19
print(new_features.shape, "Features")
print(new_labels.shape, "Labels")

print("Done extracting data")


#
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Reshape, LSTM, Conv2D, MaxPooling2D, Dropout

# model = Sequential([
#     LSTM(1000, input_shape=(new_features.shape[1], new_features.shape[2])),
#     Activation('relu'),
#     Dense(800),
#     Activation('relu'),
# Dense(600),
#     Activation('relu'),
# Dense(400),
#     Activation('relu'),
# Dense(200),
#     Activation('relu'),
#
#     Dense(19, activation='sigmoid')
# ])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(new_features, new_labels, epochs=5, batch_size=32, validation_split=0.1)



test_labels = get_test_labels()


test_features = get_test_data()

predictions = model.predict(test_features)
print(predictions[1])
true_labels = get_roc_test_labels()
print(true_labels.shape)
roc = metrics.roc_auc_score(test_labels, predictions)
print("AUC: {}".format(roc))

