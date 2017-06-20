from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate

class ImdbAutoEncoder(object):
    def __init__(self):
        pass

    def autoencoder(self, sentence):
        encoder = torch.load('encoder', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('decoder', map_location={'cuda:0': 'cpu'})

        output_words, attentions = evaluate(
            encoder, decoder, sentence)
        return output_words


if __name__ == '__main__':
    imdb_autoencoder = ImdbAutoEncoder()
    while True:
        text = input()
        print(imdb_autoencoder.autoencoder(text))