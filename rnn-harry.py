#!/home/ec2-user/anaconda3/bin/python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from rnn_utils import generate_text, load_data

# constants
SEQ_LENGTH = 50
HIDDEN_DIM = 500
LAYER_NUM = 2
BATCH_SIZE = 50
DROPOUT_RATE = 0.3
GENERATE_LENGTH = 500
MAX_EPOCH = 100
WEIGHTS = 'weights/hp/checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, 1)
TRAIN = True

# parse the data
print('\nloading data...')
#files = [open('data/hp/hp{}.txt'.format(i), 'r') for i in range(1, 8, 1) if i != 3]
files = [open('data/hp/hp1.txt', 'r')]
X, y, VOCAB_SIZE, ix_to_char = load_data(files, SEQ_LENGTH)

# build the model, lstm, but can replace with gru or simplernn
print('\nbuilding model...')
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# generate stuff before training to see it being bad
print('\npre-training results...')
generate_text(model, 100, VOCAB_SIZE, ix_to_char)

if WEIGHTS == '':
    epochs = 0
else:
    epochs = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
    print('\nloading weights from epoch {}...'.format(epochs))
    model.load_weights(WEIGHTS)

# Training if there is no trained weights specified
if TRAIN or WEIGHTS == '':
    print('training...')
    while True:
        print('\n\nepoch: {}\n'.format(epochs))
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
        epochs += 1
        print('generating text...')
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
        #if epochs % 10 == 0:
        if epochs % 1 == 0:
            print('saving weights to file...')
            model.save_weights('weights/hp/checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, epochs))
        if epochs == MAX_EPOCH:
            break

# Else, loading the trained weights and performing generation only
elif WEIGHTS != '':
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
