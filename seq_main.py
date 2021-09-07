import random

import numpy as np
import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils.tensor_util import reshape_to_matrix, create_tensor_mask, create_attention_mask
from src.utils.log_util import set_verbosity
from src.utils.visual_util import draw_array_img

set_verbosity()
seq_length = 48


def graph(features, labels):
    embeddings = Embedding(features)
    with tf.variable_scope("encoder"):
        input_mask = create_tensor_mask(features)
        attention_mask = create_attention_mask(features, input_mask)
        encoder_output = Encoder(embeddings, attention_mask, scope='layer_0')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_1')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_2')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_3')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_4')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_5')
    with tf.variable_scope("softmax"):
        dense_input = reshape_to_matrix(encoder_output)
        dense_output = Dense(dense_input, 128)
        dense_output = Dropout(dense_output)
        dense_output = Dense(dense_output, 64)
        dense_output = Dropout(dense_output)
        logits = Dense(dense_output, 21128)
        predicts = tf.argmax(logits, axis=-1)
        predicts = tf.reshape(predicts, [-1, seq_length])
        labels = tf.reshape(labels, [-1])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
    return predicts, loss, encoder_output


def read_data(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        random.shuffle(lines)
        for line in lines:
            line = line.strip()
            line = line.split("\t")
            texts.append(line[1])
            labels.append(line[1])
    return texts, labels


def train():
    train_texts, train_labels = read_data('data/translate_train.txt')
    eval_texts, eval_labels = read_data('data/translate_dev.txt')
    processor = DataProcessor()
    train_tokens = processor.texts2tokens(train_texts)
    train_tokens = processor.mask(train_tokens)
    train_features = processor.tokens2ids(train_tokens, seq_length)
    eval_tokens = processor.texts2tokens(eval_texts)
    eval_tokens = processor.mask(eval_tokens)
    eval_features = processor.tokens2ids(eval_tokens, seq_length)
    train_labels = processor.texts2ids(train_labels, seq_length)
    eval_labels = processor.texts2ids(eval_labels, seq_length)
    print("*****Examples*****")
    for i in range(5):
        print("example-%d" % i)
        print("Text: ", train_texts[i])
        print("Tokens: ", train_tokens[i])
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
        features=tf.placeholder("int32", [None, seq_length]),
        labels=tf.placeholder("int32", [None, seq_length]),
        model_graph=graph)
    model.compile()
    model.restore('config/model.ckpt')
    # Training
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
            for eval_step in range(50):
                if eval_step % 10 == 0:
                    print("Evaluate Step %d of %d......" % (eval_step, 50))
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
            print("Evaluate Labels:\t", "\n".join(str(line) for line in processor.ids2tokens(eval_batch_labels)))
            print("Evaluate Predicts:\t", "\n".join(str(line) for line in processor.ids2tokens(predicts)))
            print("************************************")
        if step % 2000 == 0 and step != 0:
            model.save(step, "result/seq")
    model.save(total_steps, "result/seq")


def validate():
    processor = DataProcessor()
    features = [
        "图书馆在二楼",
        "想学外语来找我。",
        "我依旧爱着他。",
        "內用還是外帶?",
        "他对一个生病的男人说这种事真是没良心。",
        "雪下得太大了，导致从北京起飞的飞机晚点20分钟。"]
    labels = [
        "图书馆在二楼",
        "想学外语来找我。",
        "我依旧爱着他。",
        "內用還是外帶?",
        "他对一个生病的男人说这种事真是没良心。",
        "雪下得太大了，导致从北京起飞的飞机晚点20分钟。"]
    features = processor.texts2ids(features, seq_length)
    labels = processor.texts2ids(labels, seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, seq_length]),
        labels=tf.placeholder("int32", [None, seq_length]),
        model_graph=graph)
    model.compile()
    model.restore('result/seq/model.ckpt-14000')
    var = model.validate(features, labels)
    draw_array_img(var, 'result/seq/img')


if __name__ == '__main__':
    # train()
    validate()
