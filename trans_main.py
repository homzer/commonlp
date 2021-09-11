import random

import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.cnn.conv1d_v2 import Conv1D, MaxPooling1D, Flatten1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.utils.log_util import set_verbosity
from src.utils.tensor_util import reshape_to_matrix
from src.utils.visual_util import draw_array_img

set_verbosity()
seq_length = 48
hidden_size = 768
vocab_file = 'config/vocab.txt'


def model_graph(features, labels):
    input_ids = reshape_to_matrix(features)
    embeddings = Embedding(input_ids)  # [b * 2, s, h]
    with tf.variable_scope("conv"):
        conv_output = Conv1D(embeddings, [2, 1, 6], 'layer_0')
        conv_output = MaxPooling1D(conv_output, 3)  # [b*2, 16, h, o]
        conv_output = Dropout(conv_output)

        conv_output = Conv1D(conv_output, [2, 6, 12], 'layer_1')
        conv_output = MaxPooling1D(conv_output, 4)  # [b*2, 4, h, o]
        conv_output = Dropout(conv_output, 0.5)

        conv_output = Conv1D(conv_output, [2, 12, 24], 'layer_2')
        conv_output = MaxPooling1D(conv_output, 2)  # [b*2, 2, h, o]
        conv_output = Dropout(conv_output, 0.5)

        conv_output = tf.transpose(conv_output, [0, 1, 3, 2])
        conv_output = tf.reduce_mean(conv_output, axis=-2)  # [b*2, 2, h]
        conv_output = tf.reduce_mean(conv_output, axis=-2)  # [b*2, h]
        conv_output = tf.reshape(conv_output, [-1, 2 * hidden_size])

    with tf.variable_scope("soft"):
        dense_output = Dense(conv_output, 64)
        dense_output = Dropout(dense_output, 0.5)
        logits = Dense(dense_output, 2)
        predicts = tf.argmax(logits, axis=-1)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
    return predicts, loss


def read_file(filename):
    labels = []
    ens = []
    cns = []
    with open(filename, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        random.shuffle(lines)
        for line in lines:
            line = line.strip()
            line = line.split("\t")
            labels.append(line[0])
            ens.append(line[1])
            cns.append(line[2])
    return labels, ens, cns


def train(checkpoint_file, save_steps=2000):
    train_labels, train_ens, train_cns = read_file('data/trans_train.txt')
    eval_labels, eval_ens, eval_cns = read_file('data/trans_dev.txt')
    print("*****Examples*****")
    for i in range(5):
        print("example-%d" % i)
        print("English: ", train_ens[i])
        print("Chinese: ", train_cns[i])
        print("Label: ", train_labels[i])
    processor = DataProcessor(vocab_file)
    train_ens = processor.texts2ids(train_ens, seq_length)
    train_cns = processor.texts2ids(train_cns, seq_length)
    eval_ens = processor.texts2ids(eval_ens, seq_length)
    eval_cns = processor.texts2ids(eval_cns, seq_length)
    train_features = []
    for en, cn in zip(train_ens, train_cns):
        train_features.append([en, cn])
    eval_features = []
    for en, cn in zip(eval_ens, eval_cns):
        eval_features.append([en, cn])
    print("*****Examples*****")
    for i in range(5):
        print("example-%d" % i)
        print("English ids: ", train_features[i][0])
        print("Chinese ids: ", train_features[i][1])
        print("Label: ", train_labels[i])

    train_generator = DataGenerator(
        batch_size=12,
        epochs=32,
        features=train_features,
        labels=train_labels)
    eval_generator = DataGenerator(
        batch_size=12,
        epochs=32,
        features=eval_features,
        labels=eval_labels)
    model = Model(
        features=tf.placeholder("int32", [None, 2, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    # model.freeze(['embeddings'])
    model.compile()
    model.restore(checkpoint_file)
    model.auto_train(
        train_generator,
        eval_generator,
        save_path="result/trans",
        total_steps=20000,
        eval_step=1000,
        save_step=save_steps)


def predict(checkpoint_file):
    ens = [
        "The library is on the second floor.",
        "Look at me with your books closed.",
        "I still love him.",
        "Hang your coat on the hook.",
        "It was heartless of him to say such a thing to the sick man.",
        "Because of heavy snow, the plane from Beijing arrived 20 minutes late."]
    cns = [
        "日本有很多温泉。",
        "把你的書閤起來看著我。",
        "我依旧爱着他。",
        "內用還是外帶?",
        "他对一个生病的男人说这种事真是没良心。",
        "由于大雪，从北京起飞的飞机晚点了20分钟。"]
    processor = DataProcessor(vocab_file)
    features = []
    labels = [0, 1, 1, 0, 1, 0]
    for en, cn in zip(ens, cns):
        en = processor.texts2ids(en, seq_length)
        cn = processor.texts2ids(cn, seq_length)
        features.append([en, cn])
    model = Model(
        features=tf.placeholder("int32", [None, 2, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    predictions = model.predict(features, labels)
    print("predicts: ", [prediction for prediction in predictions])


def validate(checkpoint_file):
    ens = [
        "The library is on the second floor.",
        "Look at me with your books closed.",
        "I still love him.",
        "Hang your coat on the hook.",
        "It was heartless of him to say such a thing to the sick man.",
        "Because of heavy snow, the plane from Beijing arrived 20 minutes late."]
    cns = [
        "图书馆在二楼",
        "想学外语来找我。",
        "我依旧爱着他。",
        "內用還是外帶?",
        "他对一个生病的男人说这种事真是没良心。",
        "雪下得太大了，导致从北京起飞的飞机晚点20分钟。"]
    processor = DataProcessor(vocab_file)
    features = []
    labels = [1, 0, 1, 0, 1, 1]
    for en, cn in zip(ens, cns):
        en = processor.texts2ids(en, seq_length)
        cn = processor.texts2ids(cn, seq_length)
        features.append([en, cn])
    model = Model(
        features=tf.placeholder("int32", [None, 2, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    var = model.validate(features, labels)
    draw_array_img(var, 'result/trans/img', True)


if __name__ == '__main__':
    checkpoint = 'result/trans/model.ckpt-10000'
    train(checkpoint, 10000)
    # validate(checkpoint)
