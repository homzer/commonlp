import tensorflow as tf
import numpy as np

from src.input.data_generator import DataGenerator
from src.model.cnn.Conv1D import Conv1D, Pooling1D, Flatten1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.dnn.LayerNorm import LayerNorm
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils import ConfigUtil
from src.utils import LogUtil
from src.utils.TensorUtil import create_tensor_mask, create_attention_mask, reshape2Matrix
from src.input.data_processor import DataProcessor
from src.utils.VisualUtil import draw_array_img

LogUtil.set_verbosity(LogUtil.ERROR)
label2id = {'非金融': 0, '疫情金融': 1, '金融': 2, '疫情非金融': 3}
id2label = {0: '非金融', 1: '疫情金融', 2: '金融', 3: '疫情非金融'}


def model_graph(input_ids, label_ids):
    embeddings = Embedding(input_ids)
    with tf.variable_scope("encoder"):
        attention_mask = create_attention_mask(input_ids, create_tensor_mask(input_ids))
        encoder_output, _ = Encoder(embeddings, attention_mask, scope='layer_0')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_1')
    # embeddings = reshape2Matrix(embeddings)
    # dense_output = Dense(embeddings, 3072)
    # dense_output = Dense(dense_output, 768)
    # dense_output = Dropout(dense_output)
    # norm_output = LayerNorm(dense_output + embeddings)
    # dense_output = Dense(norm_output, 3072)
    # dense_output = Dense(dense_output, 768)
    # dense_output = Dropout(dense_output)
    # norm_output = LayerNorm(dense_output + norm_output)
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
            dense_output, shape=[-1, ConfigUtil.seq_length * int(dense_output.shape[-1])])
        dense_output = Dropout(dense_output)  # [b, s * 16]
        logits = Dense(dense_output, 4)
    predicts = tf.argmax(logits, axis=-1)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_ids))
    return predicts, loss, None


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
    train_features, train_labels = read_file("./data/new_topic_train.txt")
    test_features, test_labels = read_file("./data/new_topic_test.txt")
    print("*****Examples*****")
    for i in range(5):
        print("Text: ", train_features[i])
        print("Label: ", train_labels[i])
    processor = DataProcessor(ConfigUtil.vocab_file)
    train_features = processor.texts2ids(train_features, ConfigUtil.seq_length)
    test_features = processor.texts2ids(test_features, ConfigUtil.seq_length)
    train_labels = [label2id.get(label, 0) for label in train_labels]
    test_labels = [label2id.get(label, 0) for label in test_labels]

    print("*****Examples*****")
    for i in range(5):
        print("input_ids: ", " ".join([str(x) for x in train_features[i]]))
        print("label_ids: ", train_labels[i])

    train_generator = DataGenerator(
        batch_size=8,
        epochs=64,
        features=train_features,
        labels=train_labels)
    test_generator = DataGenerator(
        batch_size=8,
        epochs=4,
        features=test_features,
        labels=test_labels)
    model = Model(features=tf.placeholder("int32", [None, ConfigUtil.seq_length]),
                  labels=tf.placeholder("int32", [None, ]),
                  model_graph=model_graph)

    model.restore(checkpoint_file)
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
                eval_batch_features, eval_batch_labels = test_generator.next_batch()
                loss, acc = model.evaluate(eval_batch_features, eval_batch_labels)
                all_loss.append(loss)
                all_acc.append(acc)
            eval_batch_features, eval_batch_labels = test_generator.next_batch()
            predicts = model.predict(eval_batch_features, eval_batch_labels)
            print("************************************")
            print("Evaluation Loss: ", np.mean(all_loss))
            print("Evaluation Accuracy: ", np.mean(all_acc))
            print("Evaluate Labels:\t", " ".join([str(label) for label in eval_batch_labels]))
            print("Evaluate Predicts:\t", " ".join([str(pred) for pred in predicts]))
            print("************************************")
        if step % save_steps == 0 and step != 0:
            model.save(step, "result/topic")
    model.save(total_steps, "result/topic")


def predict(checkpoint_file):
    texts = ['外需国内地产周期羸弱带动经济下行', '新冠肺炎疫情导致股市下跌', '今天亲师附小即将开学了可能会因为疫情推迟', '今天亲师附小即将开学了可能会推迟']
    labels = [0]
    processor = DataProcessor(ConfigUtil.vocab_file)
    features = processor.texts2ids(texts, ConfigUtil.seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, ConfigUtil.seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    predictions = model.predict(features, labels)
    print("predicts: ", [id2label.get(prediction) for prediction in predictions])


def validate(checkpoint_file):
    texts = ['外需国内地产周期羸弱带动经济下行', '新冠肺炎疫情导致股市下跌', '今天亲师附小即将开学了可能会因为疫情推迟', '今天亲师附小即将开学了可能会推迟']
    labels = [0, 1, 1, 2]
    processor = DataProcessor(ConfigUtil.vocab_file)
    features = processor.texts2ids(texts, ConfigUtil.seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, ConfigUtil.seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    variables = model.validate(features, labels)
    draw_array_img(variables, 'result/topic/image')


if __name__ == '__main__':
    checkpoint = 'config/model.ckpt'
    train(checkpoint, 3000)
    # validate()
