# -*- coding: utf-8 -*-

import json
import re
import gensim
import numpy as np
from random import shuffle


def get_data(filename, size):
    comments = []
    ratings = []
    with open(filename, encoding="utf8") as f:
        cnt = 0
        for line in f:
            comment = json.loads(line)
            rating_str = comment["Rate"]
            match = re.search("\d+", rating_str)
            if match is None:
                continue
            comments.append(comment["Text"])
            ratings.append(int(match.group()) / 10)
            cnt += 1
            if cnt == size:
                break

    return comments, ratings


def get_embedding_dict(corpus):
    model = gensim.models.Word2Vec.load("data/model.bin")
    embedding_dict = dict()
    for sent in corpus:
        words = sent.split(" ")
        for word in words:
            if word in model.vocab:
                embedding_dict[word] = model[word]
            else:
                embedding_dict[word] = np.zeros(100)
    return embedding_dict


def split_data(data, labels, dev_rate):
    dev_num = int(dev_rate * len(data))
    return data[dev_num:], data[:dev_num], labels[dev_num:], labels[:dev_num]


def embed(text_data, embedding_dict, sent_length, embedding_size):
    ret = np.zeros([len(text_data), sent_length, embedding_size])
    for i, sent in enumerate(text_data):
        for j, word in enumerate(sent.split(" ")):
            ret[i][j] = embedding_dict[word]
    return ret


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def remove_unknown_word(data):
    ret = []
    model = gensim.models.Word2Vec.load("data/model.bin")
    for sent in data:
        splited = sent.split(" ")
        ret.append(' '.join(word for word in splited if word in model.vocab))
    return ret

if __name__ == '__main__':
    comments, ratings = get_data("D:\AllComments.segmented.txt", 10000)
    train, dev, _, _ = split_data(comments, ratings, 0.1, True)
    sent_length = max(len(c.split(' ')) for c in remove_unknown_word(comments))
    a = train[:10]
    clean = remove_unknown_word(comments)
    b = embed(a, get_embedding_dict(comments), 98, 100)
    print(len(comments))
    print(len(dev))
    print(len(train))
