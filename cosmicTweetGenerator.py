import sys
import tweepy
import json
import numpy as np
from time import sleep
from keras.models import load_model
from credentials import *
from modelArchitecture import *

def generateTweet(model, pattern, inToChar, eosMarkers):
    pattern = seed[:]
    endIndex = 0
    result = ""
    for i in range(140):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(intToChar))
        prediction = model.predict(x, verbose=0)
        if pattern[-1] == 1 or pattern[-1] == 0: # whitespace or newline
            index = np.random.choice(range(79),p=prediction[0])
        else:
            index = np.argmax(prediction)
        result += intToChar[index]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        if result[-1] in eosMarkers:
            endIndex = i

    if endIndex > 0:
        # keep text only until the last end-of-sentence marker that was generated
        return result[:endIndex+1]
    else:
        # if there is no end-of-sentence marker, at least only keep entire words
        endIndex = result.rfind(' ')
        return result[:endIndex]

if __name__=='__main__':
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    seedsFile = sys.argv[1]
    weightsFile = 'weights.hdf5'
    charDictFile = 'charToInt.json'

    seqlen = 50     # length of character input sequence to the model
    eosMarkers = {'.', '!', '?'}    # end-of-sentence markers

    # read char-to-int dictionary and build int-to-char dictionary
    with open(charDictFile, 'r') as f:
        charToInt = json.load(f)
    intToChar = dict([(i,c) for (c,i) in charToInt.items()])

    # load seeds and build
    seeds = []
    with open(seedsFile, 'r') as f:
        for line in f:
            seed = [charToInt[c] for c in list(line)]
            seeds.append(seed[-50:])

    # load model weights and compile
    model.load_weights(weightsFile)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    for i in range(30):
        try:
            # generate seed
            start = np.random.randint(0, len(seeds)-1)
            seed = seeds[start]
            tweet = generateTweet(model, seed, intToChar, eosMarkers)
            api.update_status(status=tweet)
        except tweepy.TweepError as e:
            print(e.reason)
        sleep(600)
        
        
