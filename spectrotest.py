import csv
import keras
import numpy as np
import sklearn
from sklearn import metrics
import soundfile as sf
from scipy import signal
from scipy.io.wavfile import read
import scipy.io.wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from sklearn.ensemble import RandomForestClassifier

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


features = get_training_data()

# features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))
# np.save("features2d1", features)
# features = np.load('features2d1.npy')





input_shape = (features.shape[1], features.shape[2], features.shape[3])
# input_shape = (features.shape[1], features.shape[2])
num_classes = 19
print(features.shape, "Features")
print(labels.shape, "Labels")
# print("Input Shape", input_shape)

print("Done extracting data")



from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Reshape, LSTM, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, \
    BatchNormalization


# classif = RandomForestClassifier(n_estimators=500, criterion='entropy',
#                                      random_state=np.random.RandomState(0))
# classif.fit(features, labels)

model = Sequential()

model.add(Conv2D(16, (5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(2))
model.add(Conv2D(19, (4,4), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(1, 37)))
model.add(Flatten())
model.add(Dense(19, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.1)



test_labels = get_test_labels()


test_features = get_test_data()
# test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1] * test_features.shape[2]))

predictions = model.predict(test_features)
# preds = np.array(classif.predict_proba(test_features))
# print(preds.shape)
# print(test_labels.shape)
#
# predictions = np.zeros(shape=(test_labels.shape[0], test_labels.shape[1]))

# for p in range(predictions.shape[0]):
#     for bird in range(predictions.shape[1]):
#         predictions[p][bird] = preds[bird][p][1]
    # print(str(test_labels[p]) + " : " + str(predictions[p]))

# print(predictions)
# fpr, tpr, thresholds = metrics.roc_curve(test_labels, predictions, pos_label=1)
# auc = metrics.auc(fpr,tpr)
roc = metrics.roc_auc_score(test_labels,predictions, average='micro')
print("AUC: {}".format(roc))
