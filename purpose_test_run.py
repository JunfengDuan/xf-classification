from data_preprocess.build_vocab import read_file
from model.cnn_model import TextCNN
from model.configuration import TCNNConfig
from model_interface.utils import *


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("ckpt_path", "purpose_ckpt", "Path to save model")
flags.DEFINE_string("test_result",     "classification.txt",    "File for log")
flags.DEFINE_string("log_file",     "train.log",    "File for log")


def xf_purpose_test():
    save_result_path = FLAGS.test_result
    config = TCNNConfig()
    logger = get_logger(FLAGS.log_file)
    with tf.Session() as sess:
        model = restore_model(sess, TextCNN, config, FLAGS.ckpt_path, logger)
        print('Model loaded')
        contents, labels = read_file('datafile/sample.txt')
        correct_words_count = 0
        classify_results = []
        for i, line in enumerate(contents):
            purpose, probability = evaluate_text(sess, model, line)
            if purpose == labels[i]:
                correct_words_count += 1
            classify_results.append('\t'.join([purpose, str(probability), labels[i], ''.join(line)]))

        classify_results.append('判断准确度:' + str(correct_words_count / len(labels)))
    write_to_file(save_result_path, classify_results)
    print('Write finished, records lines is:', len(classify_results))

if __name__ == '__main__':
    tf.app.run(xf_purpose_test())

