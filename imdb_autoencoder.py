from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate
import train

class ImdbAutoEncoder(object):
    def __init__(self):
        pass

    def autoencoder(self, sentence):
        encoder = torch.load('test_encoder_imdb10000_0.0037651115254803288', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('test_decoder_imdb10000_0.0037651115254803288', map_location={'cuda:0': 'cpu'})

        output_words, attentions = evaluate(
            encoder, decoder, sentence)
        return output_words


if __name__ == '__main__':
    imdb_autoencoder = ImdbAutoEncoder()
    criterion = nn.NLLLoss()
    while True:
        text = input()
        text = text[:15]
        decoded_text = imdb_autoencoder.autoencoder(text)
        anomaly_predict_val = 1.0-(len(set(text.split()) - set(decoded_text)))/len(text.split())
        print(decoded_text)
        print('Probability of genuine movie review: {}%'.format(anomaly_predict_val*100))
