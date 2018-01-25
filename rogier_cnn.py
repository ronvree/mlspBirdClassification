import keras
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM, GRU
from keras.models import Sequential
from keras.utils import plot_model
from sklearn import metrics
import rogier_data as data
import time

def get_3d_input_shape(cnn_shape):
    return (cnn_shape[1], cnn_shape[2], cnn_shape[3])


def get_2d_input_shape(nn_shape):
    return (nn_shape[1], nn_shape[2])


def get_1d_input_shape(nn_shape):
    return (nn_shape[1],)


def birds_correct(predictions: np.array, labels: np.array):
    correct = 0
    for l in range(labels.shape[0]):
        p = np.round(predictions[l])
        # print(p)
        if np.array_equal(p, labels[l]):
            print("Prediction {} correct".format(l))
            print(predictions[l])
            print(labels[l])
            correct += 1
    print("Got {} correct predictions".format(correct))


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

def run_cnn_model():
    labels = data.get_training_labels()

    features = data.get_3d_training_data()

    input_shape = get_3d_input_shape(features.shape)

    num_classes = data.get_num_classes()

    print(features.shape, "Features")
    print(labels.shape, "Labels")
    print("Input Shape", input_shape)

    print("Done extracting data")

    # model = lstm_nn_model(input_shape)
    model = get_cnn_model(input_shape)

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'], loss_weights=[1., 0.2])

    print(model.summary())
    model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.1,
              verbose=2)

    test_labels = data.get_test_labels()

    test_features = data.get_2d_test_data()
    test_locations = data.get_test_locations()
    test_locations = np.reshape(test_locations, newshape=(test_locations.shape[0], 1))

    predictions = model.predict([test_features, test_locations], batch_size=32)
    for p in predictions[0]:
        print(p)
    roc = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
    print("AUC: {}".format(roc))


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


def lstm_nn_model(input_shape_nn, num_classes):
    nn_model = Sequential()
    # nn_model.add(Conv1D(40, 10, activation='sigmoid', input_shape=input_shape_nn))
    nn_model.add(GRU(512, activation='sigmoid', input_shape=input_shape_nn))
    nn_model.add(Dense(num_classes, activation='sigmoid'))
    return nn_model

def run_rnn_model():
    print("Begin extracting data")
    t1 = time.time()
    labels = data.get_training_labels()
    print("Got labels with shape " + str(labels.shape))
    features = data.get_2d_training_data()
    print("Got features with shape " + str(features.shape))

    input_shape = get_2d_input_shape(features.shape)

    num_classes = data.get_num_classes()

    test_labels = data.get_test_labels()
    print("Got -test- labels with shape " + str(test_labels.shape))
    test_features = data.get_2d_test_data()
    print("Got -test- features with shape " + str(test_features.shape))

    print("Done extracting data - {}s -".format(int(time.time() - t1)))

    model = lstm_nn_model(input_shape, num_classes)

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=[keras.metrics.binary_accuracy])

    print(model.summary())
    model.fit(features, labels, epochs=5, batch_size=32, validation_split=0.1, verbose=2)

    predictions = model.predict(test_features, batch_size=32)
    birds_correct(predictions, labels)
    roc = metrics.roc_auc_score(test_labels, predictions, average='micro')
    print("AUC: {}".format(roc))


def lstm_nn_location_model(shape_main, shape_aux, output_nodes):
    main_input = Input(shape=shape_main, name='main_input')
    lstm_out = LSTM(256, activation='sigmoid')(main_input)
    auxiliary_output = Dense(19, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = Input(shape=shape_aux, name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = Dense(40, activation='relu')(x)
    x = Dense(40, activation='relu')(x)
    main_output = Dense(output_nodes, activation='sigmoid', name='main_output')(x)
    return Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

def run_lstm_loc_model():
    print("Begin extracting data")
    t1 = time.time()
    labels = data.get_training_labels()
    print("Got labels with shape " + str(labels.shape))
    features = data.get_2d_training_data()
    print("Got features with shape " + str(features.shape))
    locations = data.get_training_locations()
    locations = np.reshape(locations, newshape=(locations.shape[0], 1))
    print("Got location features with shape " + str(locations.shape))

    input_shape = get_2d_input_shape(features.shape)
    input_shape_aux = (locations.shape[1],)

    num_classes = data.get_num_classes()

    test_labels = data.get_test_labels()
    print("Got -test- labels with shape " + str(test_labels.shape))
    test_features = data.get_2d_test_data()
    print("Got -test- features with shape " + str(test_features.shape))
    test_locations = data.get_test_locations()
    test_locations = np.reshape(test_locations, newshape=(test_locations.shape[0], 1))
    print("Got -test- location features with shape " + str(test_locations.shape))

    print("Done extracting data - {}s -".format(int(time.time()-t1)))

    model = lstm_nn_location_model(input_shape, input_shape_aux, num_classes)

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=[keras.metrics.binary_accuracy], loss_weights=[1., 0.2])
    plot_model(model, "rnn.png")

    print(model.summary())
    model.fit([features, locations], [labels, labels], epochs=5, batch_size=32, validation_split=0.05,
              verbose=2)


    predictions = model.predict([test_features, test_locations], batch_size=32)
    birds_correct(predictions[0], test_labels)
    roc = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
    print("AUC: {}".format(roc))

if __name__ == '__main__':

    run_lstm_loc_model()
    # run_rnn_model()