# generate tf.idf vector
import math

class TfidfVectorizor(object):

    def __init__(self, train_data):
        # build vocabulary dict
        self.vocab = dict()
        for sentence in train_data:
            for word in sentence.split(' '):
                if word not in self.vocab:
                    self.vocab[word] = 1
                else:
                    self.vocab[word] += 1
        self.vocab = {key: value for key, value in self.vocab.items() if value > 50}
        self.vocab_size = len(self.vocab)
        print("Vocab_size: " + str(self.vocab_size))

        # word to index
        max_idx = 0
        for word in self.vocab:
            self.vocab[word] = max_idx
            max_idx += 1

        # build inverse table
        self.doc_frequency_num = dict()
        for sentence in train_data:
            for word in set(sentence.split(' ')):
                if word not in self.vocab:
                    continue
                if word not in self.doc_frequency_num:
                    self.doc_frequency_num[word] = 1
                else:
                    self.doc_frequency_num[word] += 1

        # build idf dict
        doc_size = len(train_data)
        self.idf = {key: math.log(doc_size / value) for key, value in self.doc_frequency_num.items()}

    def process(self, sentence):
        ret = [0 for _ in range(self.vocab_size)]
        temp_dict = dict()
        for word in sentence.split(' '):
            if word not in temp_dict:
                temp_dict[word] = 1
            else:
                temp_dict[word] += 1
        for word, count in temp_dict.items():
            if word not in self.vocab:
                continue
            ret[self.vocab[word]] = count * self.idf[word]
        return ret
