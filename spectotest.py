import wave
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.io.wavfile


import soundfile as sf
data, samplerate = sf.read('src_wavs/PC1_20090513_070000_0030.wav')
f, t, Sxx = signal.spectrogram(data, samplerate)
plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])