# -*- coding: utf-8 -*-

import json
import re
import gensim
import numpy as np


def get_data(filename):
    comments = []
    ratings = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            comment = json.loads(line)
            rating_str = comment["Rate"]
            match = re.search("\d+", rating_str)
            if match is None:
                continue
            comments.append(comment["Text"])
            ratings.append(int(match.group()))
    return comments, ratings


def get_embedding_dict(corpus):
    model = gensim.models.Word2Vec.load("model.bin")
    vocab_dict = dict()
    for sent in corpus:
        words = sent.split(" ")
        for word in words:
            if word in model.vocab:
                vocab_dict[word] = model[word]
            else:
                vocab_dict[word] = np.zeros(100)
    return dict


if __name__ == '__main__':
    comments, ratings = get_data("D:\AllComments.segmented.txt")

    print(len(comments))
