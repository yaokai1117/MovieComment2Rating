import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Sentences(object):
    """
    Generate gensim input from plain text files.
    """

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = Sentences("plain")
model = gensim.models.Word2Vec(sentences)
model.save("model.bin")
