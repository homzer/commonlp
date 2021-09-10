import numpy as np
import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.cnn.conv1d_v1 import Conv1D, Flatten1D, Pooling1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils.log_util import set_verbosity
from src.utils.tensor_util import create_tensor_mask, create_attention_mask, reshape_to_matrix
from src.utils.visual_util import draw_array_img

set_verbosity()
label2id = {'非金融': 0, '疫情金融': 1, '金融': 2, '疫情非金融': 3}
id2label = {0: '非金融', 1: '疫情金融', 2: '金融', 3: '疫情非金融'}
seq_length = 48


def model_graph(input_ids, label_ids):
    embeddings = Embedding(input_ids)
    with tf.variable_scope("encoder"):
        input_mask = create_tensor_mask(input_ids)
        attention_mask = create_attention_mask(input_ids, input_mask)
        encoder_output = Encoder(embeddings, attention_mask, scope='layer_0')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_1')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_2')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_3')
        encoder_output = Encoder(encoder_output, attention_mask, scope='layer_4')
    with tf.variable_scope("conv"):
        conv_input = reshape_to_matrix(encoder_output)  # [b*s, h]
        conv_output = Conv1D(conv_input, [3, 1, 16], "layer_0")  # [b*s, h, 16]
        conv_output = Pooling1D(conv_output, 2)  # [b*s, h/2, 16]
        conv_output = Conv1D(conv_output, [3, 16, 32], "layer_1")  # [b*s, h/2, 32]
        conv_output = Pooling1D(conv_output, 2)  # [b*s, h/4, 32]
        conv_output = Flatten1D(conv_output, 32)  # [b*s, 32]
    with tf.variable_scope("project"):
        project_input = tf.reshape(conv_output, [-1, seq_length * 32])
        project_output = Dense(project_input, 128)
        project_output = Dropout(project_output)
        logits = Dense(project_output, 4)
        predicts = tf.argmax(logits, axis=-1)
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=label_ids))
    return predicts, loss, encoder_output


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
        batch_size=4,
        epochs=64,
        features=train_features,
        labels=train_labels)
    test_generator = DataGenerator(
        batch_size=8,
        epochs=4,
        features=test_features,
        labels=test_labels)
    model = Model(features=tf.placeholder("int32", [None, seq_length]),
                  labels=tf.placeholder("int32", [None, ]),
                  model_graph=model_graph)
    model.compile()
    model.restore(checkpoint_file)
    total_steps = train_generator.total_steps
    for step in range(total_steps):
        train_batch_features, train_batch_labels = train_generator.next_batch()
        model.train(train_batch_features, train_batch_labels)
        if step % 100 == 0:
            print("Global Step %d of %d" % (step, total_steps))
        if step % 500 == 0:
            print("Evaluating......")
            all_train_loss = []
            all_train_acc = []
            all_loss = []
            all_acc = []
            for eval_step in range(100):
                if eval_step % 10 == 0:
                    print("Evaluate Step %d of %d......" % (eval_step, 100))
                eval_batch_features, eval_batch_labels = test_generator.next_batch()
                train_batch_features, train_batch_labels = train_generator.next_batch()
                loss, acc = model.evaluate(eval_batch_features, eval_batch_labels)
                train_loss, train_acc = model.evaluate(train_batch_features, train_batch_labels)
                all_loss.append(loss)
                all_acc.append(acc)
                all_train_loss.append(train_loss)
                all_train_acc.append(train_acc)
            eval_batch_features, eval_batch_labels = test_generator.next_batch()
            predicts = model.predict(eval_batch_features, eval_batch_labels)
            print("************************************")
            print("Evaluation Loss: ", np.mean(all_loss))
            print("Evaluation Accuracy: ", np.mean(all_acc))
            print("Train Loss: ", np.mean(all_train_loss))
            print("Train Accuracy: ", np.mean(all_train_acc))
            print("Evaluate Labels:\t", " ".join([str(label) for label in eval_batch_labels]))
            print("Evaluate Predicts:\t", " ".join([str(pred) for pred in predicts]))
            print("************************************")
        if step % save_steps == 0 and step != 0:
            model.save(step, "result/topic")
    model.save(total_steps, "result/topic")


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
    texts = ['外需国内地产周期羸弱带动经济下行', '新冠肺炎疫情导致股市下跌', '今天亲师附小即将开学了可能会因为疫情推迟', '今天亲师附小即将开学了可能会推迟']
    labels = [0, 0, 0, 0]
    processor = DataProcessor()
    features = processor.texts2ids(texts, seq_length)
    model = Model(
        features=tf.placeholder("int32", [None, seq_length]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(checkpoint_file)
    var = model.validate(features, labels)
    draw_array_img(var, 'result/topic/img')


if __name__ == '__main__':
    checkpoint = 'result/topic/model.ckpt-bert'
    # train(checkpoint, 3000)
    # validate(checkpoint)
    predict(checkpoint)
