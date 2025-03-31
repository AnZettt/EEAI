import numpy as np
import os
from utils import *

data_directory = 'data'
sequence_length = 24
embedding_size = 100
maximal_epochs = 50
batch_size = 64


if not os.path.exists('sequences.txt'):

    RQs = load(data_directory + '/clean.txt')
    RQs = RQs.replace('\n', ' endofrq ')
    tokens = RQs.split()

    print('Total Tokens:', len(tokens))
    print('Total Unique Tokens:', len(set(tokens)))

    length = sequence_length + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i - length:i]
        line = ' '.join(seq)
        sequences.append(line)


    save(sequences, 'sequences.txt')

    print('Total Sequences: ', len(sequences))

doc = load('sequences.txt')
lines = doc.split('\n')

from keras._tf_keras.keras.preprocessing.text import Tokenizer
from pickle import dump, load

if not os.path.exists('tokenizer.pkl'):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

tokenizer = load(open('tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dropout, Dense, GRU, Embedding, BatchNormalization

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
model.add(GRU(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()

sequences = np.array(sequences)
print(sequences.shape)
X, y = sequences[:,:-1], sequences[:,-1]

print(X.shape)
print(y.shape)

if os.path.exists('model.h5'):
    from keras._tf_keras.keras.models import load_model
    model = load_model('model.h5')

model.fit(X, y, epochs=maximal_epochs, batch_size=batch_size)
model.save('model.h5')
