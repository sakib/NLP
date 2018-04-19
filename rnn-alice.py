#!/home/ec2-user/anaconda3/bin/python
import numpy, sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
train = True
filename = "data/wonderland.txt"
raw_text = open(filename).read().lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("#chars: {}\t#vocab: {}".format(n_chars, n_vocab))

# prepare the dataset of input to output pairs encoded as integers
seq_len = 100
dataX, dataY = [], []
for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i+seq_len]
    seq_out = raw_text[i+seq_len]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("#patterns: {}".format(n_patterns))

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_len, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
if train:
    print('training...')
    #define the checkpoint
    filepath = "results/wonderland-weights-improvement-{epoch:02d}-{loss:4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
else:
    #load network weights
    model.load_weights('results/wonderland-weights-improvement-50-1.2-bigger.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print('seed: \"{}\"'.format([int_to_char[val] for val in pattern]))
    #generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[val] for val in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print('Done')

