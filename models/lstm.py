import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM(object):
    def __init__(self, sent_length, class_num,
                 embedding_size, initial_embedding_dict,
                 l2_lambda, hidden_size):

        self.input_x = tf.placeholder(tf.int32, [None, sent_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        self.dropout_keep_prob_1 = tf.placeholder(tf.float32, name="dropout_keep_prob_1")
        self.dropout_keep_prob_2 = tf.placeholder(tf.float32, name="dropout_keep_prob_2")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.embedding_dict = tf.Variable(initial_embedding_dict, name="Embedding", dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_dict, self.input_x)
            # unstack embedded input
            self.unstacked = tf.unstack(self.embedded_chars, sent_length, 1)

        with tf.name_scope("lstm"):
            # create a LSTM network
            lstm_cell = rnn.BasicLSTMCell(hidden_size)
            self.output, self.states = rnn.static_rnn(lstm_cell, self.unstacked, dtype=tf.float32)
            self.pooling = tf.reduce_mean(self.output, 0)

        with tf.name_scope("linear"):
            weights = tf.get_variable(
                "W",
                shape=[hidden_size, class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")
            l2_loss += tf.nn.l2_loss(weights)
            l2_loss += tf.nn.l2_loss(bias)
            self.linear_result = tf.nn.xw_plus_b(self.pooling, weights, bias, name="linear")
            self.predictions = tf.arg_max(self.linear_result, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.linear_result, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
