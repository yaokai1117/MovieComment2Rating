# -*- coding: utf-8 -*-

import json
import re
import gensim
import numpy as np
import pickle
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


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
            ratings.append(int(match.group()) / 10 - 1)
            cnt += 1
            if cnt == size:
                break

    return comments, ratings


def get_embedding_dict(corpus):
    model = gensim.models.Word2Vec.load(config["Paths"]["embedding_path"])
    embedding_dict = dict()
    for sent in corpus:
        words = sent.split(" ")
        for word in words:
            if word in model.vocab:
                embedding_dict[word] = model[word]
            else:
                embedding_dict[word] = np.zeros(int(config["Sizes"]["embedding_size"]))
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


# test english corpus
def get_data_eng():
    x_raw = []
    y_raw = []
    with open(config["Paths"]["eng_data_path_pos"], encoding="utf8") as f:
        x_raw = [line for line in f]
        y_raw = [0 for _ in x_raw]

    with open(config["Paths"]["eng_data_path_neg"], encoding="utf8") as f:
        x_temp = [line for line in f]
        y_temp = [1 for _ in x_temp]
        x_raw.extend(x_temp)
        y_raw.extend(y_temp)
    sent_length = max(len(sent.split(" ")) for sent in x_raw)
    embedding_size = 300
    embedding_dict = pickle.load(open(config["Paths"]["eng_embedding_path"], "rb"))
    x = np.zeros([len(x_raw), sent_length, embedding_size])
    for i, sent in enumerate(x_raw):
        for j, word in enumerate(sent.split(" ")):
            if word in embedding_dict:
                x[i][j] = embedding_dict[word]
            else:
                x[i][j] = np.zeros(embedding_size)
    y = (np.arange(2) == np.array(y_raw)[:, None]).astype(np.float32)
    return x, y


if __name__ == '__main__':
    x1, y1 = get_data_eng()

    print(len(y1))
