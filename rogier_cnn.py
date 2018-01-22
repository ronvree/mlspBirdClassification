from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn import metrics

import rogier_data as data

labels = data.get_training_labels()


features = data.get_training_data()


def get_3d_input_shape(features_cnn):
    return (features_cnn.shape[1], features_cnn.shape[2], features_cnn.shape[3])

def get_2d_input_shape(features_cnn):
    return (features_cnn.shape[1], features_cnn.shape[2])


input_shape = get_3d_input_shape(features)

num_classes = data.get_num_classes()

print(features.shape, "Features")
print(labels.shape, "Labels")
print("Input Shape", input_shape)

print("Done extracting data")


def get_cnn_model(input_shape_cnn):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(16, (5,5), activation='relu', input_shape=input_shape_cnn))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(32, (5,5), activation='relu'))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(64, (5,5), activation='relu'))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(128, (5,5), activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(19, (4,4), activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(MaxPooling2D(pool_size=(1, 37)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(19, activation='sigmoid'))
    return cnn_model


model = get_cnn_model(input_shape)

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.1)

test_labels = data.get_test_labels()


test_features = data.get_test_data()

predictions = model.predict(test_features)
roc = metrics.roc_auc_score(test_labels,predictions, average='micro')
print("AUC: {}".format(roc))