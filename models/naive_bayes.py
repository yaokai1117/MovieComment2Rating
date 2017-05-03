from util import *
import math


class NaiveBayes(object):

    def __init__(self, class_num):
        x_train_raw, y_train_raw, _ = get_data(paths["train"], sizes["train"], class_num == 2)

        self.class_num = class_num
        self.reverse_dict = dict()
        self.label_cnt = [0 for _ in range(class_num + 1)]
        self.label_word_cnt = [0 for _ in range(class_num + 1)]
        self.vocab = dict()

        # get vocabulary
        for sentence in x_train_raw:
            for word in sentence.split(' '):
                if word not in self.vocab:
                    self.vocab[word] = 1
                else:
                    self.vocab[word] += 1
        self.vocab = {key: value for key, value in self.vocab.items() if value > 10}

        for sentence, label in zip(x_train_raw, y_train_raw):
            for word in sentence.split(' '):
                if word not in self.vocab:
                    continue
                if word not in self.reverse_dict:
                    self.reverse_dict[word] = [0 for _ in range(class_num + 1)]
                self.reverse_dict[word][label] += 1
                self.reverse_dict[word][class_num] += 1
                self.label_word_cnt[label] += 1
                self.label_word_cnt[class_num] += 1
            self.label_cnt[label] += 1
            self.label_cnt[class_num] += 1

        self.vocab_size = len(self.vocab)

    def predict(self, sentence):
        score = [1 for _ in range(self.class_num)]

        for i in range(self.class_num):
            score[i] += math.log(float(self.label_cnt[i]) / self.label_cnt[self.class_num])

        for word in sentence.split(' '):
            if word not in self.reverse_dict:
                continue
            for label in range(self.class_num):
                score[label] += math.log(float(self.reverse_dict[word][label] + 1) /
                                         (self.label_word_cnt[label] + self.vocab_size))

        return score.index(max(score))


if __name__ == '__main__':
    class_num = 5
    classifier = NaiveBayes(class_num)
    x_test_raw, y_test_raw, _ = get_data(paths["test"], sizes["test"], class_num == 2)
    y_predict = [classifier.predict(t) for t in x_test_raw]
    correct_predictions = float(sum(p == t for p, t in zip(y_predict, y_test_raw)))
    print(len(y_predict))
    print("Total number of test examples: {}".format(len(y_predict)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_predict))))

