import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from evaluation_protocols import hold_out_validation, k_fold_cross_validation
from models import DummyNetwork, DummyConvNetwork, MaximMilakovCNN, RandomForest
from spectrograms import read_data_as_spectrograms
from sklearn.utils import shuffle
from sklearn import metrics


# Read relevant data from files
complete_data = shuffle(read_data_as_spectrograms())

Xs = complete_data['spectrograms'].as_matrix()
ys = complete_data['labels'].as_matrix()

mean = np.mean(Xs)
std = np.std(Xs)

filters = [lambda x: gaussian_filter(x, 3),  # Gaussian filter
           # lambda x: x > threshold_otsu(x),  # Threshold
           lambda x: (x - mean) / std,  # Input normalization
           ]

for f in filters:
    for i in range(len(Xs)):
        Xs[i] = f(Xs[i])

performance_metrics = [lambda x, y: metrics.roc_auc_score(x, y, average='micro')]
# performance_metrics = [metrics.accuracy_score]

# model = RandomForest()
model = DummyNetwork(Xs[0].shape)
# model = MaximMilakovCNN(Xs[0].shape)

# performance = hold_out_validation(Xs, ys, 0.7, model, performance_metrics)
performance = k_fold_cross_validation(Xs, ys, 4, model, performance_metrics)

print(performance)
