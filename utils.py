# coding: utf-8

from nltk import word_tokenize,  pos_tag
import gensim
import numpy as np
import os
import pip


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
        return emb_model[word]
    except NameError:
        return em[str(word)]