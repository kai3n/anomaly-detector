# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import time
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN


use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)
maximum_norm = 2.0
print_loss_avg = 0
print_val_loss_avg = 0
hidden_size = 200
teacher_forcing_ratio = 0.5
MIN_LENGTH = 2
MAX_LENGTH = 17
SOS_token = 0
EOS_token = 1
log = []


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"it's", " it is", s)
    s = re.sub(r"i'm", " i am", s)
    s = re.sub(r"that's", " that is", s)
    s = re.sub(r"\'s", " 's", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"won't", " will not", s)
    s = re.sub(r"don't", " do not", s)
    s = re.sub(r"doesn't", " does not", s)
    s = re.sub(r"can't", " can not", s)
    s = re.sub(r"cannot", " can not", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/imdb100000_max16-%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[normalizeString(l), normalizeString(l)] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return MIN_LENGTH < len(p[0].split(' ')) < MAX_LENGTH and \
           MIN_LENGTH < len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    split_divider = int(len(pairs) * 0.9)
    val_pairs = pairs[split_divider:]
    pairs = pairs[:split_divider]
    print("%s sentence pairs for training" % len(pairs))
    print("%s sentence pairs for validation" % len(val_pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, val_pairs


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if lang.word2index.get(word) is not None:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(2)  # OOV
    return indexes
    # return [lang.word2index[word] for word in sentence.split(' ') if lang.word2index.get(word) is not None]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, is_validation=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output[0], target_variable[di])
            if ni == EOS_token:
                break

    if not is_validation:
        loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
    clip_grad_norm(encoder.parameters(), maximum_norm)
    clip_grad_norm(decoder.parameters(), maximum_norm)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, val_points, epoch):
    # fig, ax = plt.subplots()
    # # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.1)
    # ax.yaxis.set_major_locator(loc)
    print(points)
    print(val_points)
    print(epoch)
    plt.plot(epoch, points)
    plt.plot(epoch, val_points)
    plt.title('Training Graph for Anomaly Detector')
    plt.xlabel('Sentences')
    plt.ylabel('SGD Loss')
    plt.savefig('log/train_result_{}.png'.format(str(time.time())))
    # plt.show()


def trainIters(encoder, decoder, n_iters, print_every=500, plot_every=500, learning_rate=0.01):
    start = time.time()
    global print_loss_avg
    global print_val_loss_avg

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    plot_val_losses = []
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    epoch = []
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            epoch.append(iter)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            log.append('%s (%d %d%%) %.4f\n' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            validating_pairs = [variablesFromPair(random.choice(val_pairs))
                                for _ in range(print_every // 9)]
            for val_iter in range(1, print_every // 9 + 1):
                validating_pair = validating_pairs[val_iter - 1]
                input_variable = validating_pair[0]
                target_variable = validating_pair[1]

                val_loss = train(input_variable, target_variable, encoder,
                                 decoder, encoder_optimizer, decoder_optimizer, criterion, is_validation=True)
                print_val_loss_total += val_loss
                plot_val_loss_total += val_loss
            print_val_loss_avg = print_val_loss_total / (print_every // 9)
            print_val_loss_total = 0
            print('==========Val_Loss: %.4f==========' % (print_val_loss_avg))
            log.append('==========Val_Loss: %.4f==========\n' % (print_val_loss_avg))
            plot_val_loss_avg = plot_val_loss_total / (print_every // 9)
            plot_val_losses.append(plot_val_loss_avg)
            plot_val_loss_total = 0

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses, plot_val_losses, epoch)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    criterion = nn.NLLLoss()
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(input_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        loss += criterion(decoder_output[0], input_variable[di])
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1], loss.data.numpy()


def evaluateRandomly(encoder, decoder, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions, loss = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':

    input_lang, output_lang, pairs, val_pairs = prepareData('eng', 'eng', False)

    # pre-trained word embedding
    embeddings_index = {}
    max_features = len(input_lang.index2word)
    f = open(os.path.join('glove.6B/', 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((max_features, hidden_size))
    for word, i in input_lang.word2index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embedding_matrix = embedding_matrix

    # load model($python3 train.py encoder decoder)
    if len(sys.argv) == 3:
        if use_cuda:
            imdb_encoder = torch.load(sys.argv[1])
            imdb_decoder = torch.load(sys.argv[2])
        else:
            imdb_encoder = torch.load(sys.argv[1], map_location={'cuda:0': 'cpu'})
            imdb_decoder = torch.load(sys.argv[2], map_location={'cuda:0': 'cpu'})
    else:
        imdb_encoder = EncoderRNN(input_lang.n_words, hidden_size, embedding_matrix)
        imdb_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                      1, dropout_p=0.1)

    if use_cuda:
        imdb_encoder = imdb_encoder.cuda()
        imdb_decoder = imdb_decoder.cuda()


    trainIters(imdb_encoder, imdb_decoder, 1500000, print_every=100, plot_every=100, learning_rate=0.01)

    # save model
    torch.save(imdb_encoder, 'trained_model/encoder_imdb100000_max16_glove_'+str(print_loss_avg) + '_' + str(maximum_norm))
    torch.save(imdb_decoder, 'trained_model/decoder_imdb100000_max16_glove_'+str(print_loss_avg) + '_' + str(maximum_norm))
    with open('log/log_{}.txt'.format(str(time.time())), 'wt') as fo:
        fo.write(' '.join(log))
