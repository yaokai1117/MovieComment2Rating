from sklearn import svm
from models.tfidf import TfidfVectorizor
from util import *

class SVMClassifier(object):

    def __init__(self, class_num, x_train_raw, y_train_raw):
        self.tfidf = TfidfVectorizor(x_train_raw)
        x_train = [self.tfidf.process(sent) for sent in x_train_raw]
        y_train = y_train_raw
        self.svm = svm.SVC()
        self.svm.fit(x_train, y_train)

    def predict(self, sentence):
        vec = self.tfidf.process(sentence)
        return self.svm.predict([vec])[0]

    def predict_list(self, sentences):
        vecs = [self.tfidf.process(sentence) for sentence in sentences]
        return self.svm.predict(vecs)


if __name__ == '__main__':
    class_num = 5
    x_train_raw, y_train_raw, _ = get_data(paths["train"], sizes["train"], class_num == 2)
    x_test_raw, y_test_raw, _ = get_data(paths["test"], sizes["test"], class_num == 2)

    classifier = SVMClassifier(class_num, x_train_raw, y_train_raw)

    y_predict = classifier.predict_list(x_test_raw)
    correct_predictions = float(sum(p == t for p, t in zip(y_predict, y_test_raw)))
    print(len(y_predict))
    print("Total number of test examples: {}".format(len(y_predict)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_predict))))