from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(256, input_shape=(50, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(79, activation='softmax'))
