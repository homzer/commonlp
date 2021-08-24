import tensorflow as tf
import numpy as np

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.cnn.Conv1D import Conv1D, Pooling1D, Flatten1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils import ConfigUtil
from src.utils.TensorUtil import reshape2Matrix, create_tensor_mask, create_attention_mask
from src.utils.LogUtil import set_verbosity

set_verbosity()


def model_graph(features, labels):
    seq_length = features.shape[-1]
    input_ids = reshape2Matrix(features)
    embedding_output = Embedding(input_ids)  # [b * 2, s, e]
    hidden_size = embedding_output.shape[-1]
    embedding_output = tf.reshape(embedding_output, [-1, 2, seq_length, hidden_size])
    embedding_output = tf.reduce_sum(embedding_output, axis=1)  # [b, s, e]
    with tf.variable_scope("encoder"):
        # attention_mask = create_attention_mask(input_ids, create_tensor_mask(input_ids))
        attention_mask = None
        encoder_output = Encoder(embedding_output, attention_mask, scope='layer_1')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_2')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_3')
    with tf.variable_scope("conv"):
        conv_input = reshape2Matrix(encoder_output)  # [b*s, 768]
        conv_output = Conv1D(conv_input, [3, 1, 16], name="filter_1")  # [b*s, 768, 16]
        conv_output = Pooling1D(conv_output, 4)  # [b*s, 192, 16]
        conv_output = Dropout(conv_output)
        conv_output = Conv1D(conv_output, [3, 16, 32], name="filter_2")  # [b*s, 384, 32]
        conv_output = Pooling1D(conv_output, 4)  # [b*s, 48, 32]
        conv_output = Dropout(conv_output)
        conv_output = Conv1D(conv_output, [3, 32, 64], name="filter_3")  # [b*s, 192, 64]
        conv_output = Pooling1D(conv_output, 4)  # [b*s, 12, 64]
        conv_output = Dropout(conv_output)
        flatten_output = Flatten1D(conv_output, 32)  # [b*s, 32]
    with tf.variable_scope("projection"):
        dense_output = Dense(flatten_output, 16)  # [b*s, 16]
        dense_output = tf.reshape(
            dense_output,
            shape=[-1, ConfigUtil.seq_length * int(dense_output.shape[-1])])
        dense_output = Dropout(dense_output)  # [b, s * 16]
        logits = Dense(dense_output, 2)
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
        for line in reader.readlines():
            line = line.strip()
            line = line.split("\t")
            labels.append(line[0])
            ens.append(line[1])
            cns.append(line[2])
    return labels, ens, cns


def train():
    train_labels, train_ens, train_cns = read_file('data/trans_train.txt')
    eval_labels, eval_ens, eval_cns = read_file('data/trans_dev.txt')
    print("*****Examples*****")
    for i in range(5):
        print("example-%d" % i)
        print("English: ", train_ens[i])
        print("Chinese: ", train_cns[i])
        print("Label: ", train_labels[i])
    processor = DataProcessor(ConfigUtil.vocab_file)
    train_ens = processor.texts2ids(train_ens, ConfigUtil.seq_length)
    train_cns = processor.texts2ids(train_cns, ConfigUtil.seq_length)
    eval_ens = processor.texts2ids(eval_ens, ConfigUtil.seq_length)
    eval_cns = processor.texts2ids(eval_cns, ConfigUtil.seq_length)
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
        features=tf.placeholder(
            "int32", [None, 2, ConfigUtil.seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore("config")

    # training
    total_steps = train_generator.total_steps
    for step in range(total_steps):
        train_batch_features, train_batch_labels = train_generator.next_batch()
        global_step = model.train(train_batch_features, train_batch_labels)
        if step % 100 == 0:
            print("Global Step %d of %d" % (global_step, total_steps))
        if step % 500 == 0:
            print("Evaluating......")
            all_loss = []
            all_acc = []
            for eval_step in range(100):
                if eval_step % 10 == 0:
                    print("Evaluate Step %d of %d......" % (eval_step, 100))
                eval_batch_features, eval_batch_labels = eval_generator.next_batch()
                loss, acc = model.evaluate(eval_batch_features, eval_batch_labels)
                all_loss.append(loss)
                all_acc.append(acc)
            eval_batch_features, eval_batch_labels = eval_generator.next_batch()
            predicts = model.predict(eval_batch_features, eval_batch_labels)
            print("************************************")
            print("Evaluation Loss: ", np.mean(all_loss))
            print("Evaluation Accuracy: ", np.mean(all_acc))
            print("Evaluate Labels:\t", " ".join([str(label) for label in eval_batch_labels]))
            print("Evaluate Predicts:\t", " ".join([str(pred) for pred in predicts]))
            print("************************************")
    model.save(total_steps, ConfigUtil.output_dir)


if __name__ == '__main__':
    train()
