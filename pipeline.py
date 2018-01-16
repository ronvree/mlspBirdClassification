import numpy as np
import keras as ks

from data_info import get_number_of_species
from evaluation import Evaluation
from spectrograms import read_data_as_spectrograms
from sklearn.utils import shuffle
from sklearn import metrics


# Read relevant data from files
data = read_data_as_spectrograms()
# Shuffle data
data = shuffle(data)

# Obtain data dimensions
number_of_signals = len(data)
number_of_species = get_number_of_species()
spectrogram_shape = data['spectrograms'].iloc[0].shape

# Convert spectrograms to numpy matrix of correct dimensions TODO -- more efficiently!!!
spectrograms = data['spectrograms'].as_matrix()
fit_data = np.zeros(shape=(number_of_signals, spectrogram_shape[0], spectrogram_shape[1]))
for i in range(number_of_signals):
    for j in range(spectrogram_shape[0]):
        for k in range(spectrogram_shape[1]):
            fit_data[i][j][k] = spectrograms[i][j][k]

print(fit_data.shape)
# Normalize labels to match NN output
labels_norm = np.zeros(shape=(number_of_signals, number_of_species))
for i in range(number_of_signals):
    labels = data['labels'].iloc[i]
    for j in range(len(labels)):
        labels_norm[i][j] = 1

# Split data in train/test sets
train_frac = 0.6
split_index = int(train_frac * number_of_signals)
train_data, test_data = fit_data[:split_index], fit_data[split_index:]
train_labels, test_labels = labels_norm[:split_index], labels_norm[split_index:]

# Define dummy model
model = ks.Sequential()
model.add(ks.layers.Dense(64, activation='relu', input_shape=spectrogram_shape))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(number_of_species, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Fit the model on the data
model.fit(train_data, train_labels, epochs=50, batch_size=32)

# Evaluate the model

# print(result)

predictions = model.predict_on_batch(test_data)

predictions = np.round(predictions)

roc = metrics.roc_auc_score(test_labels.ravel(), predictions.ravel())

Evaluation(test_labels, predictions)

print('Area under ROC curve: {}'.format(roc))
