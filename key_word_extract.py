import tensorflow as tf
import sys
import operator
from util import *

class_num = 5
x_test_raw, y_test_raw, _ = get_data(paths["train"], sizes["train"], class_num == 2)
sent_length = int(sizes["sent_length"])
embedding_size = int(sizes["embedding"])

checkpoint_file = sys.argv[1]
task_name = sys.argv[2]
vocab_dict = get_char2idx_dict(paths["vocab_dict"])
x_test = char2idx(x_test_raw, vocab_dict, sent_length)
y_test = y_test_raw

# build reverse vocab dict
inverse_dict = dict()
for key, value in vocab_dict.items():
    inverse_dict[value] = key

word_weight = dict()
two_word_weight = dict()

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob_1 = graph.get_operation_by_name("dropout_keep_prob_1").outputs[0]
        dropout_keep_prob_2 = graph.get_operation_by_name("dropout_keep_prob_2").outputs[0]

        conv_out_op_1 = graph.get_operation_by_name("conv-maxpool-1/conv").outputs[0]
        conv_out_op_2 = graph.get_operation_by_name("conv-maxpool-2/conv").outputs[0]

        batches = batch_iter(list(x_test), 128, 1, shuffle=False)
        all_predictions = []
        for x_test_batch in batches:
            conv_out_1, conv_out_2 = sess.run([conv_out_op_1, conv_out_op_2],
                                         feed_dict={input_x: x_test_batch, dropout_keep_prob_1: 1.0, dropout_keep_prob_2: 1.0})
            for i in range(conv_out_1.shape[0]):
                max_indices = np.argmax(conv_out_1[i], axis=0).reshape([-1])
                for j in max_indices:
                    word = inverse_dict[x_test_batch[i][j]]
                    if word not in word_weight:
                        word_weight[word] = 1
                    else:
                        word_weight[word] += 1

            for i in range(conv_out_2.shape[0]):
                max_indices = np.argmax(conv_out_2[i], axis=0).reshape([-1])
                for j in max_indices:
                    word_1 = inverse_dict[x_test_batch[i][j]]
                    word_2 = inverse_dict[x_test_batch[i][j + 1]]
                    if word_1 + "#" + word_2 not in two_word_weight:
                        two_word_weight[word_1 + "#" + word_2] = 1
                    else:
                        two_word_weight[word_1 + "#" + word_2] += 1

        sorted_word_weight = sorted(word_weight.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(50):
            print(sorted_word_weight[i][0] +
                  str(sorted_word_weight[i][1]))

        print("#################################")

        sorted_two_word_weight = sorted(two_word_weight.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(50):
            print(sorted_two_word_weight[i][0] + "\t" +
                  str(sorted_two_word_weight[i][1]))
