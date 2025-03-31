from utils import *
import random
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

sequences_length = 24
batch_size = 64
conf_thresh = 0.6

doc = load('sequences.txt')
lines = doc.split('\n')

from pickle import load

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = load(f)

sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

model = load_model('model.h5')


result = []
in_text = lines[random.randint(0, len(lines))].split()
in_text[len(in_text) - 1] = 'endofrq'
in_text = ' '.join(in_text)
print('\n--------SAMPLE-------')
print('----------Seed---------\n',in_text)
print('-------Generated-------')
for _ in range(10):
    new_comment = ''
    while True:
        if len(new_comment.split()) >= sequences_length:
            in_text += ' endofcomment'
            break
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=sequences_length, truncating='pre')
        yhat_probs = model.predict(encoded, verbose=0)[0]
        yhat = np.random.choice(len(yhat_probs), 1, p=yhat_probs)
        out_word = ''
        for word,index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
        if out_word == 'endofcomment':
            break
        else:
            new_comment += ' ' + out_word
    print('-'+new_comment)
    result.append(new_comment)
print('----------Done---------')
