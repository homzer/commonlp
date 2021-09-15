import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.cnn.conv1d_v2 import Conv1D, MaxPooling1D, Flatten1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.utils.log_util import set_verbosity
from src.utils.visual_util import draw_array_img

set_verbosity()
label2id = {'非金融': 0, '疫情金融': 1, '金融': 2, '疫情非金融': 3}
id2label = {0: '非金融', 1: '疫情金融', 2: '金融', 3: '疫情非金融'}
seq_length = 48
hidden_size = 768


def model_graph(input_ids, label_ids):
    embeddings = Embedding(input_ids)
    with tf.variable_scope("conv"):
        conv_output = Conv1D(embeddings, [2, 1, 6], "layer_0")
        conv_output = MaxPooling1D(conv_output, 3)  # [B, 16, H, 6]
        conv_output = Dropout(conv_output)

        conv_output = Conv1D(conv_output, [2, 6, 12], "layer_1")
        conv_output = MaxPooling1D(conv_output, 4)  # [B, 4, H, O]
        conv_output = Dropout(conv_output)

        conv_output = Conv1D(conv_output, [2, 12, 24], "layer_2")
        conv_output = MaxPooling1D(conv_output, 2)  # [B, 2, H, 24]
        conv_output = Dropout(conv_output)

        conv_output = Flatten1D(conv_output)
    with tf.variable_scope("projection"):
        dense_output = Dense(conv_output, 32, name='dense_32')
        logits = Dense(dense_output, 4, name='dense_1')
        predicts = tf.argmax(logits, axis=-1)
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=label_ids))
    return loss, predicts


def read_file(filename):
    labels = []
    texts = []
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            line = line.strip()
            line = line.split("\t")
            labels.append(line[0])
            texts.append(line[1])
    return texts, labels


def train(checkpoint_file, save_steps=5000):
    train_features, train_labels = read_file("./data/topic_train.txt")
    test_features, test_labels = read_file("./data/topic_test.txt")
    print("*****Examples*****")
    for i in range(5):
        print("Text: ", train_features[i])
        print("Label: ", train_labels[i])
    processor = DataProcessor()
    train_features = processor.texts2ids(train_features, seq_length)
    test_features = processor.texts2ids(test_features, seq_length)
    train_labels = [label2id.get(label, 0) for label in train_labels]
    test_labels = [label2id.get(label, 0) for label in test_labels]

    print("*****Examples*****")
    for i in range(5):
        print("input_ids: ", " ".join([str(x) for x in train_features[i]]))
        print("label_ids: ", train_labels[i])

    train_generator = DataGenerator(
        batch_size=12,
        epochs=32,
        features=train_features,
        labels=train_labels)
    test_generator = DataGenerator(
        batch_size=12,
        epochs=4,
        features=test_features,
        labels=test_labels)
    model = Model(features=tf.placeholder("int32", [None, seq_length]),
                  labels=tf.placeholder("int32", [None, ]),
                  model_graph=model_graph)
    model.freeze(['embeddings'])
    model.compile(learning_rate=0.0001)
    model.restore(checkpoint_file)
    model.auto_train(
        train_generator=train_generator,
        eval_generator=test_generator,
        save_path="result/topic",
        total_steps=20000,
        eval_step=1000,
        save_step=save_steps)


def predict(checkpoint_file):
    texts = ['外需国内地产周期羸弱带动经济下行', '新冠肺炎疫情导致股市下跌', '今天亲师附小即将开学了可能会因为疫情推迟', '今天亲师附小即将开学了可能会推迟']
    labels = [0]
    processor = DataProcessor()
    features = processor.texts2ids(texts, seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    predictions = model.predict(features, labels)
    print("predicts: ", [id2label.get(prediction) for prediction in predictions])


def validate(checkpoint_file):
    texts = ['外需带动经济下行', '新冠肺炎疫情导致股市下跌', '今天亲师附小即将开学了可能会因为疫情推迟', '今天亲师附小即将开学了可能会推迟']
    labels = [0, 0, 0, 0]
    processor = DataProcessor()
    features = processor.texts2ids(texts, seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    var = model.validate(features, labels)
    draw_array_img(var, 'result/topic/image')


if __name__ == '__main__':
    checkpoint = 'result/topic/model.ckpt-20000'
    # train(checkpoint, 10000)
    validate(checkpoint)
