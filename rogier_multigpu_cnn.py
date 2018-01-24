from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import multi_gpu_model
from sklearn import metrics
from rogier_cnn import get_3d_input_shape, nn_model

import rogier_data as data

labels = data.get_training_labels()


features, _ = data.get_training_data()


input_shape = get_3d_input_shape(features.shape)

num_classes = data.get_num_classes()

gpus = 2

print(features.shape, "Features")
print(labels.shape, "Labels")
print("Input Shape", input_shape)

print("Done extracting data")


model = multi_gpu_model(nn_model(input_shape), gpus)

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

model.fit(features, labels, epochs=200, batch_size=16, validation_split=0.1)

test_labels = data.get_test_labels()


test_features, _ = data.get_test_data()

predictions = model.predict(test_features)
roc = metrics.roc_auc_score(test_labels,predictions, average='micro')
print("AUC: {}".format(roc))