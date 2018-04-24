# coding: utf-8

from nltk import word_tokenize,  pos_tag
import gensim
import numpy as np
import os
import pip
import pandas as pd

max_len = 20
word_shape = (300)

#importing the trained embedding model
try:
    emb_model = gensim.models.Word2Vec.load('joke_embedding')
    print ('Model is loaded ')
except FileNotFoundError:
    print ('File not found, trying to download it, please wait for few seconds')
    try:
        import wget
    except ImportError:
        pip.main(['install', 'wget'])
        import wget
    # os.makedirs('models')
    filename = wget.download('https://github.com/sanketgujar/Joke-Generation/blob/master/joke_embedding')
    emb_model = gensim.models.Word2Vec.load('joke_embedding')



def hidden_state_initializer(sent,hidden_state_size = 300):
    """
    input : sentence , hidden_state_size to be returned 
    output: embedded hidden state (averged noun embedding in the sentence)
    call : hidden_state_initializer("Hello how are you")
    """
    nouns = [token for token, pos in pos_tag(word_tokenize(sent)) if pos.startswith('N')]
    e_ = np.zeros((1,hidden_state_size))
    for i in range(len(nouns)):
        e_ += emb_model[nouns[i]]
    return (e_ / len(e_))


def get_word(embedded_vec):
	return np.array(emb_model.most_similar(embedded_vec.reshape(1,300)))

def get_word_embedding(word):
    """
    input: word
    output : word embedding
    """
    try:
        try:
            return emb_model[word]
        except KeyError:
            return np.zeros(word_shape)
    except NameError:
        return em[str(word)]

def get_sent_embedding(sent):
    sent_vec = []
    tokens  = word_tokenize(sent)
    for i in range(len(tokens)):
        sent_vec.append(get_word_embedding(tokens[i])[:300])
    while (len(sent_vec) < max_len):
        sent_vec.append(np.zeros(word_shape))
    return np.array(sent_vec[:20])

df = pd.read_csv('/home/sanket/WPI_Spring18/DL/project/data/shortjokes.csv')
x  = df['Joke'].values.tolist()
print ('Jokes file is loaded')



# print (get_word_embedding('stop.'))
print ('starting to generate jokes embedding, this will take a while')
jokes_embedded = []
for i in range(len(x)):
    jokes_embedded.append(np.concatenate([get_sent_embedding(x[i]), 
                                          hidden_state_initializer(x[i])],
                                          axis = 0))

print ('***jokes generated** Access from joke_embedded list ')
