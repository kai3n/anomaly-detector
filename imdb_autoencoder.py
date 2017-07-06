from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import numpy as np

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate, normalizeString, prepareData

def sech(x):
    return 1./np.cosh(x)

class ImdbAutoEncoder(object):

    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def autoencoder(self, sentence):
        encoder = torch.load('test_encoder_imdb100000_0.3082789914441499_2.0', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('test_decoder_imdb100000_0.3082789914441499_2.0', map_location={'cuda:0': 'cpu'})

        output_words, _, loss = evaluate(
            encoder, decoder, sentence, self.input_lang, self.output_lang)

        return output_words, loss

def val_test(textfile):
    lines = [line.rstrip('\n') for line in open(textfile)]
    total_loss = 0
    for line in lines[:100]:
        line = normalizeString(line)
        line = ' '.join(line.split()[:14])
        decoded_text, decoded_loss = imdb_autoencoder.autoencoder(line)
        total_loss += decoded_loss
    return total_loss / len(lines)

def show_val_test():
    performance = []
    for i in range(4, 15):
        performance.append(val_test('imdb_len_{}.txt'.format(str(i))))

    import matplotlib.pyplot as plt
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = ('Len 4', 'Len 5', 'Len 6', 'Len 7', 'Len 8', 'Len 9', 'Len 10', 'Len 11', 'Len 12',
               'Len 13', 'Len 14')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('SGD Loss')
    plt.title('Len & Loss Graph')
    plt.show()


if __name__ == '__main__':

    input_lang, output_lang, pairs, _ = prepareData('eng', 'eng', False)
    imdb_autoencoder = ImdbAutoEncoder(input_lang, output_lang)
    # show_val_test()

    criterion = nn.NLLLoss()
    while True:
        text = input()
        text = normalizeString(text)
        text = ' '.join(text.split()[:14])
        print('Trimmed text: ', text)
        decoded_text, decoded_loss = imdb_autoencoder.autoencoder(text)
        anomaly_prob = sech(decoded_loss / len(text.split()))
        print('Original text sequence: ', text.split())
        print('Decoded text sequence: ', decoded_text)
        print('Decoded text total loss: ', decoded_loss)
        print('Decoded text avg loss: ', decoded_loss / len(text.split()))
        print('Probability of genuine movie review: {}%'.format(anomaly_prob*100))
        print('================================================================================')
        print('Judgement of Imdb Anomaly Detector: ', 'Posiive' if anomaly_prob < 0.2 else 'Negative')
