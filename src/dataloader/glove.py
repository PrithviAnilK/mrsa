# load the GloVe vectors in a dictionary:
# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
from tqdm import tqdm 
import numpy as np
GLOVE_PATH = 'D:\\datasets\\GLOVE\\glove.840B.300d.txt'

embeddings_index = {}
f = open(GLOVE_PATH, encoding="utf-8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    # coefs = np.asarray(values[1:], dtype='float32')
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))