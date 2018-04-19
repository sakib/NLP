#!/home/ec2-user/anaconda3/bin/python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation

# constants
SEQ_LENGTH = 100
HIDDEN_DIM = 700
LAYER_NUM = 2
BATCH_SIZE = 64
DROPOUT_RATE = 0.3
GENERATE_LENGTH = 1000
NUM_EPOCHS = 100

# function to generate text from model
def generate_text(model, length=GENERATE_LENGTH):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

# parse the data
files = [open('data/hp/hp{}.txt'.format(i), 'r') for i in range(1, 8, 1) if i != 3]
data = '\n'.join([f.read().lower() for f in files])
chars = list(set(data))
VOCAB_SIZE = len(chars)
print('#chars: {}\t#vocab: {}'.format(len(data), VOCAB_SIZE))

# create char to int mappings
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

# format data for input
X = np.zeros((int(len(data)/SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((int(len(data)/SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))
print(X.shape, y.shape)

# one hot vectors input [0:100], one hot vectors output [1:101]
for i in range(0, int(len(data)/SEQ_LENGTH)):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

# build the model, lstm, but can replace with gru or simplernn
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# run the epochs
nb_epoch = 0
while True:
    print('\n\n')
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    print('generated text at epoch {}: \n{}'.format(nb_epoch, generate_text(model)))
    if nb_epoch % 10 == 0:
        model.save_weights('results/hp/checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
    if nb_epoch == NUM_EPOCHS:
        break
