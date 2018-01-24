import numpy as np

from data_info import get_number_of_species
from models import DummyNetwork, DummyConvNetwork
from spectrograms import read_data_as_spectrograms
from sklearn.utils import shuffle
from sklearn import metrics


# Read relevant data from files
data = shuffle(read_data_as_spectrograms())
# TODO -- check if data itself is not shuffled

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

fit_data = fit_data.reshape(fit_data.shape + (1,))

print(fit_data.shape)
# print(fit_data.shape + (1,))

# Normalize labels to match NN output
labels_norm = np.zeros(shape=(number_of_signals, number_of_species))
for i in range(number_of_signals):
    labels = data['labels'].iloc[i]
    for j in range(len(labels)):
        labels_norm[i][j] = 1

# Split data in train/test sets
train_frac = 0.6  # TODO -- use evaluation protocol
split_index = int(train_frac * number_of_signals)
train_data, test_data = fit_data[:split_index], fit_data[split_index:]
train_labels, test_labels = labels_norm[:split_index], labels_norm[split_index:]

# Define dummy model
print(fit_data.shape[1:3] + (1,))
model = DummyConvNetwork(fit_data.shape[1:3] + (1,))

# Fit the model on the data
model.fit(train_data, train_labels)

# Evaluate the model
predictions = model.predict(test_data)

predictions = np.round(predictions)

roc = metrics.roc_auc_score(test_labels.ravel(), predictions.ravel())

print('Area under ROC curve: {}'.format(roc))
