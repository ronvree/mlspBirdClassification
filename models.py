import keras as ks

from data_info import get_number_of_species


class Model:

    def fit(self, train_data, train_labels):
        raise Exception('Unimplemented!')

    def predict(self, samples):
        raise Exception('Unimplemented!')


class DummyNetwork(Model):

    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Dense(64, activation='sigmoid', input_shape=input_shape))
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(get_number_of_species(), activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_labels):
        return self.model.fit(train_data, train_labels, epochs=50, batch_size=32)

    def predict(self, samples):
        return self.model.predict_on_batch(samples)

