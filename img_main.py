import numpy as np
import tensorflow as tf

from src.input.data_generator import DataGenerator
from src.model.cnn.Conv2D import Conv2D, Pooling2D, Flatten2D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.modeling import Model
from src.utils import ConfigUtil
from src.utils import LogUtil

LogUtil.set_verbosity(LogUtil.ERROR)


def model_graph(input_imgs, label_ids):
    conv_output = Conv2D(input_imgs, [3, 3, 1, 32], "conv1")  # [batch_size, 28, 28, 32]
    conv_output = Pooling2D(conv_output, [2, 2])  # [batch_size, 14, 14, 32]
    conv_output = Dropout(conv_output)
    conv_output = Conv2D(conv_output, [3, 3, 32, 64], "conv2")  # [batch_size, 14, 14, 64]
    conv_output = Pooling2D(conv_output, [2, 2])  # [batch_size, 7, 7, 64]
    conv_output = Dropout(conv_output)
    conv_output = Conv2D(conv_output, [3, 3, 64, 128], "conv3")  # [batch_size, 7, 7, 128]
    flatten_output = Flatten2D(conv_output, 32)
    dense_output = Dense(flatten_output, 64)
    dense_output = Dense(dense_output, 32)
    logits = Dense(dense_output, 10)
    predicts = tf.argmax(logits, axis=-1)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label_ids))
    return predicts, loss, None


def train():
    file = np.load('./data/mnist.npz')
    train_features = file['x_train']
    train_labels = file['y_train']
    test_features = file['x_test']
    test_labels = file['y_test']

    train_generator = DataGenerator(
        batch_size=32,
        epochs=1,
        features=train_features,
        labels=train_labels)

    test_generator = DataGenerator(
        batch_size=32,
        epochs=1,
        features=test_features,
        labels=test_labels)

    model = Model(
        features=tf.placeholder("float32", [None, 28, 28]),
        labels=tf.placeholder("int32", [None, ]),
        model_graph=model_graph)
    model.restore(ConfigUtil.output_dir)
    total_steps = train_generator.total_steps
    for step in range(total_steps):
        batch_features, batch_labels = train_generator.next_batch()
        global_step = model.train(batch_features, batch_labels)
        if step % 100 == 0:
            print("Global steps %d of %d" % (global_step, total_steps))
        if step % 1000 == 0:
            print("Evaluating......")
            all_loss = []
            all_acc = []
            for _ in range(100):
                batch_test_features, batch_test_labels = test_generator.next_batch()
                loss, acc = model.evaluate(batch_test_features, batch_test_labels)
                all_loss.append(loss)
                all_acc.append(acc)
            batch_test_features, batch_test_labels = test_generator.next_batch()
            predicts, _ = model.predict(batch_test_features, batch_test_labels)
            print("************************************")
            print("Evaluate Loss: %.6f" % np.mean(all_loss))
            print("Evaluate Accuracy: %.6f" % np.mean(all_acc))
            print("Evaluate Labels:\t", " ".join([str(label) for label in batch_test_labels]))
            print("Evaluate Predicts:\t", " ".join([str(pred) for pred in predicts]))
            print("************************************")
    model.save(total_steps, ConfigUtil.output_dir)


if __name__ == '__main__':
    train()
