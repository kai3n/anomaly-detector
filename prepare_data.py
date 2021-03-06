import pickle
import os
import re

import sys
if sys.version_info[0] == 2:
    # check typo error
    import enchant
    d = enchant.Dict("en_US")

MIN_LENGTH = 3
MAX_LENGTH = 50

class ImdbData(object):

    def __init__(self):
        self.data = ''
        # self.data4 = ''
        # self.data5 = ''
        # self.data6 = ''
        # self.data7 = ''
        # self.data8 = ''
        # self.data9 = ''
        # self.data10 = ''
        # self.data11= ''
        # self.data12 = ''
        # self.data13 = ''
        # self.data14 = ''
        # self.data15 = ''
        # self.data16 = ''
        # self.data17 = ''
        # self.data18 = ''
        # self.data19 = ''
        # self.data20 = ''

        self.trimmed_sentences = []

    @staticmethod
    def _normalize_text(text):
        """Remove impurities from the text"""
        text = re.sub(r"<br />", "", text)
        # text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", text)
        text = re.sub(r"it's", " it is", text)
        text = re.sub(r"that's", " that is", text)
        text = re.sub(r"\'s", " 's", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", " will not", text)
        text = re.sub(r"don't", " do not", text)
        text = re.sub(r"can't", " can not", text)
        text = re.sub(r"cannot", " can not", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def load_data(self):
        data_set = []
        for path_train in ['dataset/train/pos/', 'dataset/train/neg/', 'dataset/test/pos/', 'dataset/test/neg/']:
            data_set.extend([open(path_train + f).read().split('.') for f in os.listdir(path_train) if f.endswith('.txt')])
            for i in data_set:
                for j in i:
                    trimmed_sentence = self._normalize_text(j)
                    if MIN_LENGTH < len(trimmed_sentence.split()) < MAX_LENGTH:
                        flag = 0
                        flag2 = 0
                        for k in ',!?()[]-:<>{}/=+_*^%$#@~"':
                            if trimmed_sentence.find(k) != -1:
                                flag = 1
                        if flag == 0:
                            if sys.version_info[0] == 2:
                                for l in trimmed_sentence.split():
                                        if not d.check(l):
                                            flag2 = 1
                            if flag2 == 0:
                                if trimmed_sentence[0] == ' ':
                                    self.trimmed_sentences.append(trimmed_sentence[1:])
                                else:
                                    self.trimmed_sentences.append(trimmed_sentence)
        return self

    def save_to_txt(self, trimmed_sentences, filename='imdb100000_max16-eng-eng.txt'):
        for sentence in trimmed_sentences[:300000]:
            if len(self.data.split('\n')) == 100001:
                break
            if 2 < len(sentence.split()) < 17:
                self.data += sentence + '.\n'
        with open(filename, 'wt') as fo:
            fo.write(self.data[:-1])

    # def save_to_txt(self, trimmed_sentences, filename='imdb1000000-eng-eng.txt'):
    #     for sentence in trimmed_sentences[100000:300000]:
    #         if len(sentence.split()) == 15:
    #             self.data15 += sentence + '.\n'
    #             if len(self.data15.split('\n')) == 1001:
    #                 with open('imdb_len_15.txt', 'wt') as fo:
    #                     fo.write(self.data15[:-1])
    #         if len(sentence.split()) == 16:
    #             self.data16 += sentence + '.\n'
    #             if len(self.data16.split('\n')) == 1001:
    #                 with open('imdb_len_16.txt', 'wt') as fo:
    #                     fo.write(self.data16[:-1])
    #         if len(sentence.split()) == 17:
    #             self.data17 += sentence + '.\n'
    #             if len(self.data17.split('\n')) == 1001:
    #                 with open('imdb_len_17.txt', 'wt') as fo:
    #                     fo.write(self.data17[:-1])
    #         if len(sentence.split()) == 18:
    #             self.data18 += sentence + '.\n'
    #             if len(self.data18.split('\n')) == 1001:
    #                 with open('imdb_len_18.txt', 'wt') as fo:
    #                     fo.write(self.data18[:-1])
    #         if len(sentence.split()) == 19:
    #             self.data19 += sentence + '.\n'
    #             if len(self.data19.split('\n')) == 1001:
    #                 with open('imdb_len_19.txt', 'wt') as fo:
    #                     fo.write(self.data19[:-1])
    #         if len(sentence.split()) == 20:
    #             self.data20 += sentence + '.\n'
    #             if len(self.data20.split('\n')) == 1001:
    #                 with open('imdb_len_20.txt', 'wt') as fo:
    #                     fo.write(self.data20[:-1])

    def save_to_pkl(self, trimmed_sentences, filename='imdb-eng-eng.pkl'):
        for sentence in trimmed_sentences:
            self.data += sentence + '\n'
        pickle.dump(self.data[:-1], open(filename, 'wb'))

    def load_to_pkl(filename='imdb-eng-eng.pkl'):
        pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    data = ImdbData()
    data.save_to_txt(data.load_data().trimmed_sentences)
