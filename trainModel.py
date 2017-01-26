import numpy as np
import sys
import json
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from modelArchitecture import *

if __name__=='__main__':
    textFile = sys.argv[1]
    seqlen = 50     # length of character input sequence to the model

    text = ''
    with open(textFile, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    charDict = dict(zip(chars, range(len(chars))))
    with open('charToInt.json', 'w') as f:
        json.dump(charDict, f, sort_keys=True, indent=4)
        
    encodedText = [charDict[c] for c in list(text)]

    
    # generate input and target data
    dataX = []
    dataY = []
    for i in range(0, len(encodedText) - seqlen, 1):
        dataX.append(encodedText[i:i+seqlen])
        dataY.append(encodedText[i+seqlen])
    x = np.reshape(dataX, (len(dataX), seqlen, 1))
    x = x/float(len(chars))
    y = to_categorical(dataY)

    model = getModel(seqlen, len(chars))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x, y, nb_epoch=25, batch_size=100, callbacks=callbacks_list)


    
          

        

