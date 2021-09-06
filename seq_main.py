import numpy as np
import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.input.data_processor import DataProcessor
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils import ConfigUtil
from src.utils import LogUtil
from src.utils.TensorUtil import create_tensor_mask, create_attention_mask, reshape_to_matrix

LogUtil.set_verbosity()


def model_graph(input_ids, label_ids):
    embeddings = Embedding(input_ids)
    with tf.variable_scope("encoder"):
        attention_mask = create_attention_mask(input_ids, create_tensor_mask(input_ids))
        encoder_output, _ = Encoder(embeddings, attention_mask, scope='layer_1')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_2')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_3')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_4')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_5')
        encoder_output, probs = Encoder(encoder_output, attention_mask, scope='layer_6')
    with tf.variable_scope("projection"):
        project_input = reshape_to_matrix(encoder_output)
        project_input = Dense(project_input, 128)
        project_input = Dropout(project_input)
        project_input = Dense(project_input, 64)
        project_input = Dropout(project_input)
        logits = Dense(project_input, ConfigUtil.vocab_size)
        predicts = tf.argmax(logits, axis=-1)
        label_ids = tf.reshape(label_ids, [-1, ])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_ids))
    return predicts, loss, tf.reshape(probs, [-1, ConfigUtil.seq_length, ConfigUtil.seq_length])


def read_file(filename):
    labels = []
    texts = []
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            line = line.strip()
            line = line.split("\t")
            labels.append(line[1])
            texts.append(line[1])
    return texts, labels


def train():
    train_features, train_labels = read_file("./data/topic_train.txt")
    test_features, test_labels = read_file("./data/topic_test.txt")
    print("*****Examples*****")
    for i in range(5):
        print("Text: ", train_features[i])
        print("Label: ", train_labels[i])

    processor = DataProcessor(ConfigUtil.vocab_file)
    train_features = processor.texts2ids(train_features, ConfigUtil.seq_length)
    train_labels = processor.texts2ids(train_labels, ConfigUtil.seq_length)
    test_features = processor.texts2ids(test_features, ConfigUtil.seq_length)
    test_labels = processor.texts2ids(test_labels, ConfigUtil.seq_length)
    print("*****Examples*****")
    for i in range(5):
        print("input_ids: ", " ".join([str(x) for x in train_features[i]]))
        print("label_ids: ", " ".join([str(x) for x in train_labels[i]]))

    train_generator = DataGenerator(
        batch_size=4,
        epochs=32,
        features=train_features,
        labels=train_labels)
    test_generator = DataGenerator(
        batch_size=8,
        epochs=4,
        features=test_features,
        labels=test_labels)
    model = Model(features=tf.placeholder("int32", [None, ConfigUtil.seq_length]),
                  labels=tf.placeholder("int32", [None, ConfigUtil.seq_length]),
                  model_graph=model_graph)

    model.restore("config")
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
            predicts = np.reshape(predicts, newshape=[-1, ConfigUtil.seq_length])
            print("************************************")
            print("Evaluation Loss: ", np.mean(all_loss))
            print("Evaluation Accuracy: ", np.mean(all_acc))
            print("Evaluate Labels:\t", "\n".join([str(label) for label in eval_batch_labels]))
            print("Evaluate Predicts:\t", "\n".join([str(pred) for pred in predicts]))
            print("************************************")
        if step % 5000 == 0 and step != 0:
            model.save(step, ConfigUtil.output_dir)
    model.save(total_steps, ConfigUtil.output_dir)


train()
