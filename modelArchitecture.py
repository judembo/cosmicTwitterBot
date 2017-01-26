from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.recurrent import LSTM

def getModel(seqlen, outDim):
    model = Sequential()
    model.add(LSTM(256, input_shape=(seqlen, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(outDim, activation='softmax'))

    return model
