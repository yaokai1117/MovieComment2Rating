import tensorflow as tf
import sys
from util import *

class_num = 5
x_test_raw, y_test_raw, _ = get_data(paths["test"], sizes["test"], class_num == 2)
print(len(x_test_raw))
sent_length = int(sizes["sent_length"])
embedding_size = int(sizes["embedding"])

checkpoint_file = sys.argv[1]
task_name = sys.argv[2]
vocab_dict = get_char2idx_dict(paths["vocab_dict"])
x_test = char2idx(x_test_raw, vocab_dict, sent_length)
y_test = y_test_raw

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob_1 = graph.get_operation_by_name("dropout_keep_prob_1").outputs[0]
        dropout_keep_prob_2 = graph.get_operation_by_name("dropout_keep_prob_2").outputs[0]

        predictions = graph.get_operation_by_name("linear/predictions").outputs[0]

        batches = batch_iter(list(x_test), 128, 1, shuffle=False)
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions,
                                         {input_x: x_test_batch, dropout_keep_prob_1: 1.0, dropout_keep_prob_2: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

correct_predictions = float(sum(all_predictions == y_test))
print(len(all_predictions))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

with open("results\{}.txt".format(task_name), "w") as f:
    f.write("checkpoint file: {}\n".format(checkpoint_file))
    f.write("Total number of test examples: {}\n".format(len(y_test)))
    f.write("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))

log_all_results = True
if log_all_results:
    with open("results\{}.all.txt".format(task_name), "w") as f:
        f.writelines([json.dumps({"Text": text, "GT": str(ground_truth), "PD": str(prediction)}) + '\n'
                     for text, ground_truth, prediction
                     in zip(x_test_raw, y_test_raw, all_predictions)])
    with open("results\{}.wrong.txt".format(task_name), "w") as f:
        f.writelines([json.dumps({"Text": text, "GT": str(ground_truth), "PD": str(prediction)}) + '\n'
                     for text, ground_truth, prediction
                     in zip(x_test_raw, y_test_raw, all_predictions)
                     if ground_truth != prediction])