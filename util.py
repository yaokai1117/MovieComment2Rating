# -*- coding: utf-8 -*-

import json
import re
import os
import gensim
import numpy as np
import pickle
import configparser
import platform

config = configparser.ConfigParser()
if os.path.isfile("config.ini"):
    config.read("config.ini")
else:
    config.read("../config.ini")
paths = config["Win-Paths" if platform.system() == "Windows" else "Paths"]
sizes = config["Sizes"]


def get_data(filename, size, to_binary=False):
    """  Extract text, label and movie_id from raw json file """
    comments = []
    ratings = []
    movie_ids = []
    with open(filename, encoding="utf8") as f:
        cnt = 0
        for line in f:
            comment = json.loads(line)
            rating_str = comment["Rate"]
            match = re.search("\d+", rating_str)
            if match is None:
                continue
            rating = int(int(match.group()) / 10 - 1)
            if to_binary:
                if rating == 2:
                    continue
                elif rating < 2:
                    ratings.append(0)
                else:
                    ratings.append(1)
            else:
                ratings.append(rating)
            movie_ids.append(comment["MovieId"])
            comments.append(comment["Text"])
            cnt += 1
            if cnt == size:
                break

    return comments, ratings, movie_ids


def get_data_eng():
    """ Get english 2-class data """
    x_raw = []
    y_raw = []
    with open(paths["eng_data_pos"], encoding="utf8") as f:
        x_raw = [line for line in f]
        y_raw = [0 for _ in x_raw]

    with open(paths["eng_data_neg"], encoding="utf8") as f:
        x_temp = [line for line in f]
        y_temp = [1 for _ in x_temp]
        x_raw.extend(x_temp)
        y_raw.extend(y_temp)
    sent_length = max(len(sent.split(" ")) for sent in x_raw)
    embedding_size = 300
    embedding_dict = pickle.load(open(paths["eng_embedding"], "rb"))
    x = np.zeros([len(x_raw), sent_length, embedding_size])
    for i, sent in enumerate(x_raw):
        for j, word in enumerate(sent.split(" ")):
            if word in embedding_dict:
                x[i][j] = embedding_dict[word]
            else:
                x[i][j] = np.zeros(embedding_size)
    y = (np.arange(2) == np.array(y_raw)[:, None]).astype(np.float32)
    return x, y


def get_data_DMSC(size):
    """ Get DMSC data set """
    comments = []
    ratings = []
    movie_ids = []
    with open(paths["dmsc_data"], encoding="UTF-8") as f:
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            splited = line.split(',')
            if len(splited) != 10:
                continue
            if not (splited[0].strip().isdigit() and splited[7].strip().isdigit()):
                continue
            comments.append(splited[8])
            ratings.append(int(splited[7]) - 1)
            movie_ids.append(splited[1])
            if cnt == size:
                break
            cnt += 1

    return comments, ratings, movie_ids


def dump_char2idx_dict(data, dict_name):
    """ Return a map from each word to an index """
    vocab_dict = dict()
    vocab_dict[""] = 0
    max_id = 1
    for sent in data:
        for word in sent.split(" "):
            if word not in vocab_dict:
                vocab_dict[word] = max_id
                max_id += 1
    pickle.dump(vocab_dict, open(dict_name, "wb"))
    return vocab_dict


def get_char2idx_dict(dict_name):
    return pickle.load(open(dict_name, "rb"))


def char2idx(data, vocab_dict, sent_length):
    """ Convert text data to idx representation """
    ret = np.zeros([len(data), sent_length])
    for i, sent in enumerate(data):
        for j, word in enumerate(sent.split(" ")):
            ret[i][j] = vocab_dict[word]
    return ret


def get_embedding_dict(corpus):
    """ Return embedding dict """
    model = gensim.models.Word2Vec.load(paths["embedding"])
    embedding_dict = dict()
    for sent in corpus:
        words = sent.split(" ")
        for word in words:
            if word in model.vocab:
                embedding_dict[word] = model[word]
            else:
                embedding_dict[word] = np.zeros(int(sizes["embedding"]))
    return embedding_dict


def split_data(data, labels, dev_rate):
    """ Divide data into training and dev set """
    dev_num = int(dev_rate * len(data))
    return data[dev_num:], data[:dev_num], labels[dev_num:], labels[:dev_num]


def embed(text_data, embedding_dict, sent_length, embedding_size):
    """ Convert text data to word embedding representation """
    ret = np.zeros([len(text_data), sent_length, embedding_size])
    for i, sent in enumerate(text_data):
        for j, word in enumerate(sent.split(" ")):
            ret[i][j] = embedding_dict[word]
    return ret


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """ Generate batches """
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


if __name__ == '__main__':
    # x_train_raw, y_train_raw, _ = get_data(paths["train_char"], 40000)
    # x_dev_raw, y_dev_raw, _ = get_data(paths["dev_char"], 10000)
    # x_test_raw, y_test_raw, _ = get_data(paths["test_char"], 10000)
    # comments = x_train_raw + x_dev_raw + x_test_raw
    # dump_char2idx_dict(comments, paths["vocab_dict_char"])
    # sent_length = max(len(t.split(' ')) for t in comments)
    # print(sent_length)
    print("not done.")
