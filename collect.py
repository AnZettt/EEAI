import pandas as pd
import os, string
from utils import save

Data_Directory = 'data'

data = []

for file in os.listdir(Data_Directory):
    print(file)
    data.append(pd.read_csv(os.path.join(Data_Directory + '/' + file)))

data = pd.concat(data)

RQ = data['ResearchQuestion']

print('Total RQ: ', len(RQ))

RQ = set(RQ)

print('Total unique RQ: ', len(RQ))

clean = []

for i in RQ:
    tokens = i.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    clean.append(' '.join(tokens))

save(clean, Data_Directory + '/clean.txt')