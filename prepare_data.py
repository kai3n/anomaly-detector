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

    def save_to_txt(self, trimmed_sentences, filename='imdb1000000-eng-eng.txt'):
        count = 0
        for sentence in trimmed_sentences:
            if count == 1000000:
                break
            count += 1
            self.data += sentence + '.\n'
        with open(filename, 'wt') as fo:
            fo.write(self.data[:-1])

    def save_to_pkl(self, trimmed_sentences, filename='imdb-eng-eng.pkl'):
        for sentence in trimmed_sentences:
            self.data += sentence + '\n'
        pickle.dump(self.data[:-1], open(filename, 'wb'))

    def load_to_pkl(filename='imdb-eng-eng.pkl'):
        pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    data = ImdbData()
    data.save_to_txt(data.load_data().trimmed_sentences)
