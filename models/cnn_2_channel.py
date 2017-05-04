import numpy as np
import tensorflow as tf


class CNNTwoChannel(object):
    def __init__(self, sent_length, class_num,
                 embedding_size, initial_embedding_dict,
                 l2_lambda, filter_sizes, filter_num):

        self.input_x = tf.placeholder(tf.int32, [None, sent_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        self.dropout_keep_prob_1 = tf.placeholder(tf.float32, name="dropout_keep_prob_1")
        self.dropout_keep_prob_2 = tf.placeholder(tf.float32, name="dropout_keep_prob_2")

        l2_loss = tf.constant(0.0)

        # self-adapt embedding
        with tf.name_scope("embedding"):
            self.embedding_dict = tf.Variable(initial_embedding_dict, name="Embedding", dtype=tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_dict, self.input_x)
            self.embedding_dict_static = tf.Variable(initial_embedding_dict, name="Embedding_static", dtype=tf.float32,
                                                     trainable=False)
            self.embedded_chars_static = tf.nn.embedding_lookup(self.embedding_dict_static, self.input_x)
            self.embedded_chars_expanded = tf.stack([self.embedded_chars, self.embedded_chars_static], axis=3)

        # add dropout
        with tf.name_scope("dropout_1"):
            self.embedded_chars_dropped = tf.nn.dropout(self.embedded_chars_expanded, self.dropout_keep_prob_1,
                                                        noise_shape=[tf.shape(self.embedded_chars_expanded)[0],
                                                                     sent_length, 1, 1])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, filter_num]
                weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                bias = tf.Variable(tf.constant(0.01, shape=[filter_num]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_dropped,
                    weights,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sent_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = filter_num * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout_2"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob_2)

        with tf.name_scope("linear"):
            weights = tf.get_variable(
                "W",
                shape=[num_filters_total, class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")
            l2_loss += tf.nn.l2_loss(weights)
            l2_loss += tf.nn.l2_loss(bias)
            self.linear_result = tf.nn.xw_plus_b(self.h_drop, weights, bias, name="linear")
            self.predictions = tf.arg_max(self.linear_result, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.linear_result, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
