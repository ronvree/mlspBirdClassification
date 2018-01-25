import keras
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model
from sklearn import metrics
import rogier_data as data


def get_3d_input_shape(cnn_shape):
    return (cnn_shape[1], cnn_shape[2], cnn_shape[3])


def get_2d_input_shape(nn_shape):
    return (nn_shape[1], nn_shape[2])


def get_1d_input_shape(nn_shape):
    return (nn_shape[1],)


def get_cnn_model(input_shape_cnn):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(16, (5, 5), activation='relu', input_shape=input_shape_cnn))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(64, (5, 5), activation='relu'))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(128, (5, 5), activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(MaxPooling2D(2))
    cnn_model.add(Conv2D(19, (4, 4), activation='relu'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(MaxPooling2D(pool_size=(1, 37)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(19, activation='sigmoid'))
    return cnn_model


def nn_model(input_shape_nn):
    nn_model = Sequential()

    nn_model.add(Dense(100, activation='relu', input_shape=input_shape_nn))
    nn_model.add(Dense(5, activation='relu'))
    nn_model.add(Dense(1, activation='relu'))
    nn_model.add(Flatten())
    nn_model.add(Dense(19, activation='sigmoid'))
    return nn_model


def flat_nn_model(input_shape_nn):
    nn_model = Sequential()

    nn_model.add(Dense(10, activation='relu', input_shape=input_shape_nn))
    nn_model.add(Dense(500, activation='relu'))
    nn_model.add(Dense(400, activation='relu'))
    nn_model.add(Dense(300, activation='relu'))
    nn_model.add(Dense(200, activation='relu'))
    nn_model.add(Dense(100, activation='relu'))

    nn_model.add(Dense(19, activation='sigmoid'))
    return nn_model


def lstm_nn_model(input_shape_nn):
    nn_model = Sequential()
    # nn_model.add(Conv1D(40, 10, activation='sigmoid', input_shape=input_shape_nn))
    nn_model.add(LSTM(1024, activation='sigmoid', input_shape=input_shape_nn))
    nn_model.add(Dense(19, activation='sigmoid'))
    return nn_model


def lstm_nn_location_model(shape_main, shape_aux):
    main_input = Input(shape=shape_main, batch_shape=(1, shape_main[0], shape_main[1]), name='main_input')
    lstm_out = LSTM(1024, activation='sigmoid', stateful=True)(main_input)
    auxiliary_output = Dense(19, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=shape_aux, batch_shape=(1, shape_aux[0]), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.1)(x)
    main_output = Dense(data.get_num_classes(), activation='sigmoid', name='main_output')(x)
    return Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])


labels = data.get_training_labels()

features = data.get_2d_training_data()
locations = data.get_training_locations()
locations = np.reshape(locations, newshape=(locations.shape[0], 1))

input_shape = get_2d_input_shape(features.shape)
input_shape_aux = (locations.shape[1],)

num_classes = data.get_num_classes()

print(features.shape, "Features")
print(labels.shape, "Labels")
print("Input Shape", input_shape)

print("Done extracting data")

# model = lstm_nn_model(input_shape)
model = multi_gpu_model(lstm_nn_location_model(input_shape, input_shape_aux),gpus=2)

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'], loss_weights=[1., 0.2])

print(model.summary())
model.fit([features, locations], [labels, labels], epochs=100, batch_size=1, validation_split=0.1,
          verbose=2)

test_labels = data.get_test_labels()

test_features = data.get_2d_test_data()
test_locations = data.get_test_locations()
test_locations = np.reshape(test_locations, newshape=(test_locations.shape[0], 1))

predictions = model.predict([test_features, test_locations], batch_size=1)
for p in predictions[0]:
    print(p)
roc = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
print("AUC: {}".format(roc))
