import csv
import numpy as np
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
    with open('data/mlsp_contest_dataset/essential_data/CVfolds_2.txt') as csvfile:
        reader = csv.DictReader(csvfile)
        cv_folds = dict()
        for row in reader:
            if row['fold'] in cv_folds:
                cv_folds[row['fold']].append(row['rec_id'])
            else:
                cv_folds[row['fold']] = [row['rec_id']]
        return cv_folds['0']

def get_training_filenames():
    filenames = get_file_names()
    training_ids = get_training_ids()
    return [filenames.get(x) for x in training_ids]

def get_training_data():
    result = []
    train_files = get_training_filenames()
    # for f in trainfiles:
    data, samplerate = sf.read('data/mlsp_contest_dataset/essential_data/src_wavs/'+ train_files[0] + '.wav')
    f, t, Sxx = signal.spectrogram(data, samplerate)

    print(f.shape)
    print(t.shape)
    print(Sxx.shape)


print(get_training_data())

#
#
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# from keras.models import Sequential
#
# model = Sequential()
# from keras.layers import Dense
#
# model.add(Dense(units=64, activation='relu', input_dim=100))
# model.add(Dense(units=10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
