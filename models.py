import keras as ks
from sklearn.ensemble import RandomForestClassifier

from data_info import get_number_of_species
from preprocessing import normalize_labels, copy_to_matrix


class Model:

    def fit(self, train_data, train_labels):
        raise Exception('Unimplemented!')

    def predict(self, samples):
        raise Exception('Unimplemented!')

    def pre_process_data(self, data, labels):
        return copy_to_matrix(data), normalize_labels(labels)


class DummyNetwork(Model):

    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Dense(input_shape[0], activation='sigmoid', input_shape=input_shape))
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(get_number_of_species(), activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_labels):
        return self.model.fit(train_data, train_labels, epochs=1, batch_size=32)

    def predict(self, samples):
        return self.model.predict_on_batch(samples)


class DummyConvNetwork(Model):

    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape + (1,)))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))

        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(64, activation='sigmoid'))
        self.model.add(ks.layers.Dense(get_number_of_species(), activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_labels):
        return self.model.fit(train_data, train_labels, epochs=5, batch_size=32)

    def predict(self, samples):
        return self.model.predict_on_batch(samples)

    def pre_process_data(self, data, labels):
        data, labels = super(DummyConvNetwork, self).pre_process_data(data, labels)
        return data.reshape(data.shape + (1,)), labels


class RandomForest(Model):

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=500, criterion='gini')

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)

    def pre_process_data(self, data, labels):
        return data, labels


class MaximMilakovCNN(Model):

    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape + (1,)))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(32, kernel_size=(5, 5), activation='relu'))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(128, kernel_size=(5, 5), activation='relu'))
        self.model.add(ks.layers.MaxPooling2D((2, 2)))
        self.model.add(ks.layers.Conv2D(get_number_of_species(), kernel_size=(4, 4), activation='relu'))
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(get_number_of_species(), activation='tanh'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_labels):
        return self.model.fit(train_data, train_labels, epochs=5, batch_size=32)

    def predict(self, samples):
        return self.model.predict_on_batch(samples)

    def pre_process_data(self, data, labels):
        data, labels = super(MaximMilakovCNN, self).pre_process_data(data, labels)
        return data.reshape(data.shape + (1,)), labels






