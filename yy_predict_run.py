from data_preprocess.build_vocab import read_file
from model.cnn_model import TextCNN
from model.configuration import TCNNConfig
from model_interface.utils import *
from flask import Flask, request
from flask_cors import CORS

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("ckpt_path", "yy_ckpt", "Path to load model")
flags.DEFINE_string("classify_result", "yy_classification.txt", "File for store classify result")
flags.DEFINE_boolean("model_test", False, "whether is doing  model test")
flags.DEFINE_string("log_file",     "yy_predict.log",    "File for log")

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/yy_predict', methods=['GET', 'POST'])
def yy_predict():

    text = request.get_json().get('text')
    print('text:', text)

    if text is None or len(text) == 0:
        return ""

    response = xf_yy(text)
    print('\nresponse:', response)

    return response


def xf_yy(text):
    """
    扬言模型测试
    :return:
    """
    save_result_path = FLAGS.classify_result
    config = TCNNConfig()
    logger = get_logger(FLAGS.log_file)
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = restore_model(sess, TextCNN, config, FLAGS.ckpt_path, logger)
        print('Model loaded')

        if FLAGS.model_test:
            contents, labels = read_file('datafile/xf_yy_data/yy_sample.txt')
            correct_words_count = 0
            classify_results = []
            for i, line in enumerate(contents):
                yy, probability = evaluate_text(sess, model, line)
                if yy == labels[i]:
                    correct_words_count += 1
                classify_results.append('\t'.join([yy, str(probability), labels[i], ''.join(line)]))

            classify_results.append('判断准确度:' + str(correct_words_count / len(labels)))
            write_to_file(save_result_path, classify_results)
            print('Write finished, records lines is:', len(classify_results))
        else:
            yy_tag, _ = evaluate_text(sess, model, text)
            return str(dict(result=yy_tag))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8084, debug=True)
    # tf.app.run(xf_yy())

