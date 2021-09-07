import random

import numpy as np
import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.cnn.Conv2D import Conv2D, Flatten2D, MeanPooling2D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.rnn.Lstm import Lstm, BiLstm
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils.log_util import set_verbosity
from src.utils.tensor_util import reshape_to_matrix, create_tensor_mask, create_attention_mask
from src.utils.visual_util import draw_array_img

set_verbosity()
seq_length = 48
vocab_file = 'config/vocab.txt'


def model_graph(features, labels):
    input_ids = reshape_to_matrix(features)
    embeddings = Embedding(input_ids)  # [b * 2, s, h]
    hidden_size = embeddings.shape[-1]
    with tf.variable_scope("lstm"):
        lstm_output, fw_state, bw_state = BiLstm(embeddings)  # [b*2, h]
    with tf.variable_scope("softmax"):
        dense_input = tf.reshape(lstm_output, [-1, hidden_size * 2])
        dense_output = Dense(dense_input, 64, name='dense64')
        dense_output = Dropout(dense_output)
        logits = Dense(dense_output, 2, name='dense2')
        predicts = tf.argmax(logits, axis=-1)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
    return predicts, loss, None


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


def train(checkpoint_file, save_checkpoint_steps=2000):
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
        batch_size=4,
        epochs=4,
        features=train_features,
        labels=train_labels)
    eval_generator = DataGenerator(
        batch_size=8,
        epochs=4,
        features=eval_features,
        labels=eval_labels)
    model = Model(
        features=tf.placeholder("int32", [None, 2, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.compile()
    model.restore(checkpoint_file)
    # training
    total_steps = train_generator.total_steps
    for step in range(total_steps):
        train_batch_features, train_batch_labels = train_generator.next_batch()
        global_step = model.train(train_batch_features, train_batch_labels)
        if step % 100 == 0:
            print("Global Step %d of %d" % (global_step, total_steps))
        if step % 500 == 0:
            print("Evaluating......")
            all_eval_loss = []
            all_train_loss = []
            all_train_acc = []
            all_eval_acc = []
            for eval_step in range(100):
                if eval_step % 10 == 0:
                    print("Evaluate Step %d of %d......" % (eval_step, 100))
                eval_batch_features, eval_batch_labels = eval_generator.next_batch()
                train_batch_features, train_batch_labels = train_generator.next_batch()
                eval_loss, eval_acc = model.evaluate(eval_batch_features, eval_batch_labels)
                train_loss, train_acc = model.evaluate(train_batch_features, train_batch_labels)
                all_train_loss.append(train_loss)
                all_train_acc.append(train_acc)
                all_eval_loss.append(eval_loss)
                all_eval_acc.append(eval_acc)
            eval_batch_features, eval_batch_labels = eval_generator.next_batch()
            predicts = model.predict(eval_batch_features, eval_batch_labels)
            print("************************************")
            print("Evaluation Loss: ", np.mean(all_eval_loss))
            print("Evaluation Accuracy: ", np.mean(all_eval_acc))
            print("Train Loss: ", np.mean(all_train_loss))
            print("Train Accuracy: ", np.mean(all_train_acc))
            print("Evaluate Labels:\t", " ".join([str(label) for label in eval_batch_labels]))
            print("Evaluate Predicts:\t", " ".join([str(pred) for pred in predicts]))
            print("************************************")

        if step % save_checkpoint_steps == 0 and step != 0:
            model.save(step, "result/trans")
    model.save(total_steps, "result/trans")


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
    draw_array_img(var, 'result/trans/img')


if __name__ == '__main__':
    checkpoint = 'config/model.ckpt'
    train(checkpoint, 3000)
    # predict(checkpoint)
    # validate(checkpoint)
