import tensorflow as tf
import time
import os
import datetime
import sys
from util import *
from models.cnn import CNN
from models.cnn_dynamic_embedding import CNNDynamic
from models.cnn_2_channel import CNNTwoChannel
from models.cnn_2_layer import CNNTwoLayer


# read hyperparameter from config file
param_config = configparser.ConfigParser()
param_config.read(sys.argv[1])

class_num = int(param_config["Parameter"]["class_num"])
sent_length = int(param_config["Parameter"]["sent_length_char"])
filters = [int(f) for f in param_config["Parameter"]["filters"].split(",")]
filter_num = int(param_config["Parameter"]["filter_num"])
dropout_keep_prob_1 = float(param_config["Parameter"]["dropout_keep_prob_1"])
dropout_keep_prob_2 = float(param_config["Parameter"]["dropout_keep_prob_2"])
l2_lambda = float(param_config["Parameter"]["l2_lambda"])

# prepare raw data and embedding dict

# comments, ratings, movie_ids = get_data(paths["data"], int(sizes["data"]), True)
# x_train_raw, x_dev_raw, y_train_raw, y_dev_raw = split_data(comments, ratings, 0.2)
x_train_raw, y_train_raw, _ = get_data(paths["train_char"], sizes["train"], class_num == 2)
x_dev_raw, y_dev_raw, _ = get_data(paths["dev_char"], sizes["dev"], class_num == 2)
x_test_raw, y_test_raw, _ = get_data(paths["test_char"], sizes["test"], class_num == 2)
comments = x_train_raw + x_dev_raw + x_test_raw
embedding_dict = get_embedding_dict(comments)
embedding_size = int(sizes["embedding"])

# get input data
# x_train = embed(x_train_raw, embedding_dict, sent_length, embedding_size)
# x_dev = embed(x_dev_raw, embedding_dict, sent_length, embedding_size)
vocab_dict = get_char2idx_dict(paths["vocab_dict_char"])
x_train = char2idx(x_train_raw, vocab_dict, sent_length)
x_dev = char2idx(x_dev_raw, vocab_dict, sent_length)

embedding_dict_array = np.zeros([len(vocab_dict), embedding_size])
if param_config["Parameter"]["random_embedding"] == '1':
    embedding_dict_array = np.random.rand(len(vocab_dict), embedding_size) / 10
else:
    for i, word in enumerate(embedding_dict.keys()):
        embedding_dict_array[vocab_dict[word]] = embedding_dict[word]

y_train = (np.arange(class_num) == np.array(y_train_raw)[:, None]).astype(np.float32)
y_dev = (np.arange(class_num) == np.array(y_dev_raw)[:, None]).astype(np.float32)

# for english corpus
# x, y = get_data_eng()
# sent_length = 61
# embedding_size = 300
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
# x_train, x_dev, y_train, y_dev = split_data(x_shuffled, y_shuffled, 0.5)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # model = Softmax(
        #     sent_length=sent_length,
        #     class_num=class_num,
        #     embedding_size=embedding_size,
        #     l2_lambda=0.0
        # )
        # model = CNN(
        #     sent_length=sent_length,
        #     class_num=class_num,
        #     embedding_size=embedding_size,
        #     l2_lambda=l2_lambda,
        #     filter_num=filter_num,
        #     filter_sizes=filters
        # )
        # model = CNNTwoChannel(
        #     sent_length=sent_length,
        #     class_num=class_num,
        #     embedding_size=embedding_size,
        #     initial_embedding_dict=embedding_dict_array,
        #     l2_lambda=l2_lambda,
        #     filter_num=filter_num,
        #     filter_sizes=filters
        # )
        model = CNNTwoLayer(
            sent_length=sent_length,
            class_num=class_num,
            embedding_size=embedding_size,
            initial_embedding_dict=embedding_dict_array,
            l2_lambda=l2_lambda,
            filter_num_1=filter_num,
            filter_sizes_1=filters,
            filter_num_2=64,
            filter_sizes_2=[1, 2]
        )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(paths["output"], "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob_1: dropout_keep_prob_1,
                model.dropout_keep_prob_2: dropout_keep_prob_2
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob_1: 1.0,
                model.dropout_keep_prob_2: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), 128, 40)
        batch_num_per_epoch = int(len(x_train) / 128) + 1
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % batch_num_per_epoch == 0:
                epoch_num = int(current_step / batch_num_per_epoch)
                print("\nEpoch:" + str(epoch_num))
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
