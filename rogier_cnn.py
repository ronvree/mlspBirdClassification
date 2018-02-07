import time

import keras
import numpy as np
from keras import Input, Model
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM, GRU, AveragePooling2D, GlobalMaxPooling1D, \
    Reshape, BatchNormalization, Concatenate, GlobalMaxPooling2D
from keras.metrics import binary_accuracy
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


def birds_correct(predictions: np.array, labels: np.array):
    correct = 0
    for l in range(labels.shape[0]):
        p = np.round(predictions[l])
        # print(p)
        if np.array_equal(p, labels[l]):
            print("Prediction {} correct with sum: {}".format(l, sum(p)))
            # print(predictions[l])
            # print(labels[l])
            correct += 1
    print("Got {} correct predictions".format(correct))


class AucCallback(Callback):
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.model.validation_data[0])
        roc = metrics.roc_auc_score(self.model.validation_data[1][0], y_pred[0], average='micro')
        roc_do = metrics.roc_auc_score(self.model.validation_data[1][1], y_pred[1], average='micro')
        print('mo_auc: {} - do_auc: {} '.format(str(round(roc, 4)), str(round(roc_do, 4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def milakov_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(1, 5))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(1, 2))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(1, 2))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(1, 2))
    model.add(Dropout(0.1))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def piczak_model(input_shape_cnn, num_classes):
    model = Sequential()
    model.add(Conv2D(80, (57, 60), activation='relu', input_shape=input_shape_cnn))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(Conv2D(80, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def detection_model(input_shape, num_classes, map_size=96):
    model = Sequential()
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 5)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Reshape((313, 96)))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def detection_with_location_model(is_sound, is_location, num_classes, dr=0.25, maps=96, pool_sizes=(5, 2, 2, 2),
                                  gru_units=256):
    sound_input = Input(shape=is_sound, name='si')
    conv_1 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[0]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(sound_input)))))

    conv_2 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[1]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(conv_1)))))

    conv_3 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[2]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(conv_2)))))

    conv_4 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[3]))(
        BatchNormalization()((Conv2D(maps, (3, 3), padding='same', activation='relu')(conv_3)))))

    reshape = Reshape((313, maps))(conv_4)
    gru_1 = Dropout(dr)(GRU(gru_units, return_sequences=True)(reshape))

    gru_2 = Dropout(dr)(GRU(gru_units, return_sequences=True)(gru_1))
    time_pooling = GlobalMaxPooling1D()(gru_2)
    detection_output = Dense(1, activation='sigmoid', name='do')(time_pooling)

    location_input = Input(shape=is_location, name='li')

    concat = Concatenate()([detection_output, location_input])
    hidden_layer_1 = Dropout(dr)(Dense(200, activation='relu')(concat))
    hidden_layer_2 = Dropout(dr)(Dense(200, activation='relu')(hidden_layer_1))
    output_layer = Dense(num_classes, activation='sigmoid', name='mo')(hidden_layer_2)

    return Model(inputs=[sound_input, location_input], outputs=[output_layer, detection_output])


def classification_detection_with_location_model(is_sound, is_location, num_classes, dr=0.25, maps=96, c_maps=96,
                                                 pool_sizes=(5, 2, 2, 2), gru_units=256):
    sound_input = Input(shape=is_sound, name='si')
    conv_1 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[0]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(sound_input)))))

    conv_2 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[1]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(conv_1)))))

    conv_3 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[2]))(
        BatchNormalization()((Conv2D(maps, (5, 5), padding='same', activation='relu')(conv_2)))))

    conv_4 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[3]))(
        BatchNormalization()((Conv2D(maps, (3, 3), padding='same', activation='relu')(conv_3)))))

    reshape = Reshape((313, maps))(conv_4)
    gru_1 = Dropout(dr)(GRU(gru_units, return_sequences=True)(reshape))

    gru_2 = Dropout(dr)(GRU(gru_units, return_sequences=True)(gru_1))
    time_pooling = GlobalMaxPooling1D()(gru_2)
    detection_output = Dense(1, activation='sigmoid', name='do')(time_pooling)

    c_conv_1 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[0]))(
        BatchNormalization()(Conv2D(c_maps, (3, 3),padding='same', activation='relu')(Conv2D(c_maps, (5, 5),padding='same', activation='relu')(sound_input)))))

    c_conv_2 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[1]))(
        BatchNormalization()(Conv2D(c_maps, (3, 3),padding='same', activation='relu')(Conv2D(c_maps, (5, 5),padding='same', activation='relu')(
            c_conv_1)))))

    c_conv_3 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[2]))(
        BatchNormalization()(Conv2D(c_maps, (3, 3),padding='same', activation='relu')(Conv2D(c_maps, (5, 5),padding='same', activation='relu')(
            c_conv_2)))))

    c_conv_4 = Dropout(dr)(MaxPooling2D(pool_size=(1, pool_sizes[3]))(
        BatchNormalization()(Conv2D(c_maps, (3, 3),padding='same', activation='relu')(Conv2D(c_maps, (3, 3),padding='same', activation='relu')(
            c_conv_3)))))

    c_reshape = Reshape((313, maps))(c_conv_4)

    c_time_pooling = GlobalMaxPooling1D()(c_reshape)

    classification_output = Dense(num_classes, activation='sigmoid', name='co')(c_time_pooling)

    location_input = Input(shape=is_location, name='li')

    concat = Concatenate()([detection_output, location_input])
    hidden_layer_1 = Dropout(dr)(Dense(200, activation='relu')(concat))
    hidden_layer_2 = Dropout(dr)(Dense(200, activation='relu')(hidden_layer_1))
    output_layer = Dense(num_classes, activation='sigmoid', name='mo')(hidden_layer_2)

    return Model(inputs=[sound_input, location_input],
                 outputs=[output_layer, detection_output, classification_output])


def get_cnn_model(input_shape_cnn, num_classes):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_cnn))
    cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((1, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((1, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((1, 2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(AveragePooling2D(48, 1))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(500, activation='relu'))
    cnn_model.add(Dense(500, activation='relu'))
    # cnn_model.add(Dropout(0.5))
    # cnn_model.add(Dense(500, activation='relu'))
    # cnn_model.add(Dropout(0.5))
    # cnn_model.add(Dense(500, activation='relu'))

    cnn_model.add(Dense(num_classes, activation='sigmoid'))
    return cnn_model


def run_cnn_model(multi_gpu=False):
    print("Begin extracting data")
    t1 = time.time()
    features, labels, locations, single_labels = data.get_3d_training_data()
    print("Got labels with shape " + str(labels.shape))
    print("Got sound features with shape " + str(features.shape))
    print("Got location features with shape " + str(locations.shape))
    input_shape = get_3d_input_shape(features.shape)
    input_shape_l = (1,)

    num_classes = data.get_num_classes()

    test_labels, test_single_labels = data.get_test_labels()
    print("Got -test- labels with shape " + str(test_labels.shape))
    test_features = data.get_3d_test_data()

    print("Got -test- sound features with shape " + str(test_features.shape))

    test_locations = data.get_test_locations()

    print("Got -test- location features with shape " + str(test_locations.shape))

    print("Done extracting data - {}s -".format(int(time.time() - t1)))

    # model = detection_model(input_shape, num_classes)
    model = classification_detection_with_location_model(input_shape, input_shape_l, num_classes)

    if multi_gpu:
        model = multi_gpu_model(model, 2)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[binary_accuracy])
    print(model.summary())

    model.fit([features, locations], [labels, single_labels, labels], epochs=10, batch_size=32, validation_split=0.1,
              verbose=1)
    #
    one_feature = test_features[0:10]
    print(one_feature.shape)
    conv_spec = model.predict([one_feature, test_locations[0:10]])[0]
    [print(x, y) for (x, y) in zip(conv_spec, test_labels[0:10])]

    predictions = model.predict([test_features, test_locations])
    print(model.evaluate([test_features, test_locations], [test_labels, test_single_labels, test_labels]))
    # birds_correct(predictions, test_labels)
    roc_mo = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
    roc_do = metrics.roc_auc_score(test_single_labels, predictions[1], average='micro')
    roc_co = metrics.roc_auc_score(test_labels, predictions[2], average='micro')
    print("AUC Detection output      : {}".format(roc_do))
    print("AUC Classification output : {}".format(roc_co))
    print("AUC Main output           : {}".format(roc_mo))


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
    # nn_model.add(Dense(200, activation='sigmoid', input_shape=input_shape_nn))
    # nn_model.add(Dense(200, activation='sigmoid'))
    # nn_model.add(Dense(200, activation='sigmoid'))
    # nn_model.add(Dense(200, activation='sigmoid'))
    nn_model.add(LSTM(200, input_shape=input_shape_nn))
    nn_model.add(Dense(num_classes, activation='sigmoid'))
    return nn_model


def run_rnn_model():
    print("Begin extracting data")
    t1 = time.time()

    features, labels, _ = data.get_2d_training_data()
    print("Got labels with shape " + str(labels.shape))
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
    model.fit(features, labels, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    predictions = model.predict(test_features, batch_size=32)
    print(model.evaluate(test_features, test_labels))
    print(list(zip(predictions, test_labels)))
    birds_correct(predictions, test_labels)
    roc = metrics.roc_auc_score(test_labels, predictions, average='micro')
    print("AUC: {}".format(roc))


def lstm_nn_location_model(shape_main, shape_aux, output_nodes):
    main_input = Input(shape=shape_main, name='main_input')
    lstm_out = LSTM(256)(main_input)
    # lstm_out = LSTM(128, )(x)
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

    features, labels, locations = data.get_2d_training_data()
    print("Got labels with shape " + str(labels.shape))
    print("Got features with shape " + str(features.shape))
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

    print("Done extracting data - {}s -".format(int(time.time() - t1)))

    model = lstm_nn_location_model(input_shape, input_shape_aux, num_classes)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[keras.metrics.binary_accuracy], loss_weights=[1., 0.2])

    print(model.summary())
    model.fit([features, locations], [labels, labels], epochs=5, batch_size=32, validation_split=0.05,
              verbose=1)

    predictions = model.predict([test_features, test_locations], batch_size=32)
    birds_correct(predictions[0], test_labels)
    roc = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
    print("AUC: {}".format(roc))


def grid_search_params(multi_gpu):
    print("Begin extracting data")
    t1 = time.time()
    features, labels, locations, single_labels = data.get_3d_training_data()
    print("Got labels with shape " + str(labels.shape))
    print("Got sound features with shape " + str(features.shape))
    print("Got location features with shape " + str(locations.shape))
    input_shape = get_3d_input_shape(features.shape)
    input_shape_l = (1,)

    num_classes = data.get_num_classes()

    test_labels, test_single_labels = data.get_test_labels()
    print("Got -test- labels with shape " + str(test_labels.shape))
    test_features = data.get_3d_test_data()

    print("Got -test- sound features with shape " + str(test_features.shape))

    test_locations = data.get_test_locations()

    print("Got -test- location features with shape " + str(test_locations.shape))

    print("Done extracting data - {}s -".format(int(time.time() - t1)))

    # model = detection_model(input_shape, num_classes)
    print("Begin grid search")
    feature_mappings = [50, 70, 90, 110]
    gru_unit_sizes = [32, 64, 128, 256]
    for d_map in feature_mappings:
        for c_map in feature_mappings:
            for gru in gru_unit_sizes:
                model = classification_detection_with_location_model(input_shape, input_shape_l, num_classes,
                                                                     maps=d_map, c_maps=c_map, gru_units=gru)

                if multi_gpu:
                    model = multi_gpu_model(model, 2)

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=[binary_accuracy])

                model.fit([features, locations], [labels, single_labels, labels], epochs=10,
                          batch_size=32, validation_split=0.2, verbose=2)
                predictions = model.predict([test_features, test_locations])

                roc_mo = metrics.roc_auc_score(test_labels, predictions[0], average='micro')
                roc_do = metrics.roc_auc_score(test_single_labels, predictions[1], average='micro')
                roc_co = metrics.roc_auc_score(test_labels, predictions[2], average='micro')
                print("Model with ({}, {}, {})".format(d_map, c_map, gru))
                print("AUC Detection output      : {}".format(roc_do))
                print("AUC Classification output : {}".format(roc_co))
                print("AUC Main output           : {}".format(roc_mo))
                quit()


if __name__ == '__main__':
    # run_lstm_loc_model()
    # run_rnn_model()
    # run_cnn_model()
    #
    # pred = np.load('predictions.npy')
    # for p in pred:
    #     print(p)
    #     print(sum(p))
    #
    grid_search_params(multi_gpu=True)
