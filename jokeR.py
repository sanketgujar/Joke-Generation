import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import *

def data_loader(path):
    # data file is n_jokes x 21 x 300
    # first 20 columns are jokes, final column is the topics
    # data will contain the embeddings, at this stage I will not know what word goes with what embedding

    data = np.load(path)
    jokes = data[:, :20, :]
    topics = data[:, -1, :]
    return jokes, topics


class RNN(nn.Module):
    def __init__(self, generate=False):
        super(RNN, self).__init__()

        self.input_dim = 300
        self.hidden_dim = 300
        self.output_dim = 300
        self.sequence_length = 20
        self.n_layers = 1

        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.attention = torch.nn.Linear(self.input_dim, self.sequence_length)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers)
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, sequence, topic):
        self.hidden = self.init_hidden(topic)
        sequence = self.input(sequence)
        output_seq, hidden_seq = self.lstm(sequence, self.hidden)
        output = self.output(output_seq)

        if generate == True:
            output = self.softmax(output)

        return output, hidden_seq

    def init_hidden(self, topic):
        return (Variable(topic),
                Variable(topic))


def get_batches(jokes, topics):
    for i in np.arange(0, jokes.shape[1], self.batch_size):
        joke_batch = jokes[:, i:i+self.batch_size, :]
        topic_batch = topics[i:i+self.batch_size]
        yield (joke_batch, topic_batch)


def train():
    NUM_EPOCH = 100
    BATCH_SIZE = 16
    criterion = nn.CosineEmbeddingLoss()

    for e in range(NUM_EPOCHS):
        for batch in get_batches(jokes, topics):
            joke_batch, topic_batch = batch
    
            joke_batch, topic_batch = Variable(torch.FloatTensor(joke_batch)), Variable(torch.FloatTensor(topic_batch))
            outputs, hidden = rnn(joke_batch, topic_batch)
            loss = criterion(outputs, joke_batch)
            loss.backward()
            print("current_loss:", rnn.loss.data)


def generate():
    NUM_GEN = 30
    SEQ_LENGTH = 20

    input = Variable(torch.rand(1, 300))

    for i in range(NUM_GEN):
        for j in range(SEQ_LENGTH):
            seed_idx = np.random.randint(N_JOKES)
            seed_topic = topics[seed_idx, :, :]
            
            output, hidden = rnn(input, seed_topic, generate=True)
            # output is 300 dims, from that I get top 10 most likely words, apply softmax to confidence scores and sample
            word_weights = output
            word_idx = torch.Multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx) # next input is the index of the current word



if __name__ == "__main__":
#    seq = Variable(torch.FloatTensor(np.random.randn(10, 5, 300)))
#    hidden = torch.FloatTensor(np.random.randn(1, 1, 300))
#    rnn = RNN()
#    rnn.forward(seq, hidden)
    jokes, topics = data_loader("./joke_data.npy")
    N_JOKES = len(jokes)

    rnn = RNN()
    train()
