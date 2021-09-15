# 自然语言处理通用架构

## 1.架构说明

- 支持的版本为3.6和3.5

- config包下存放模型的配置文件和字嵌入映射文件

- data包下存放训练和测试集

- src包存放模型源码，包含：
    
    - input 输入处理包，负责对数据进行预处理，创建数据生成器等
    
    - model 神经网络包，包含卷积神经网络、循环神经网络以及Transformer等通用网络结构
    
    - optimizer 模型优化器，包含adam优化器
    
    - utils 工具包，包含对checkpoint和Tensor的操作方法

- result包用来存放模型训练结果和参数文件等
    
    
## 2.重要使用步骤说明

1）数据预处理
```
from src.input.data_processor import DataProcessor

seq_length = 48

processor = DataProcessor()
train_features = processor.texts2ids(train_features, seq_length)
test_features = processor.texts2ids(test_features, seq_length)
```

2）创建数据生成器
```
from src.input.data_generator import DataGenerator

train_generator = DataGenerator(
        batch_size=4,
        epochs=64,
        features=train_features,
        labels=train_labels)
test_generator = DataGenerator(
        batch_size=4,
        epochs=64,
        features=test_features,
        labels=test_labels)
```

3）定义模型结构图
```
import tensorflow as tf

from src.model.cnn.conv1d_v1 import Conv1D, Flatten1D, Pooling1D
from src.model.dnn.Dense import Dense
from src.model.dnn.Dropout import Dropout
from src.model.transformer.Embedding import Embedding
from src.model.transformer.Encoder import Encoder
from src.utils.tensor_util import create_tensor_mask, create_attention_mask, reshape_to_matrix

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
    return loss, predicts, encoder_output
```

4）编译模型、冻结参数
```
from src.model.modeling import Model

model = Model(features=tf.placeholder("int32", [None, seq_length]),
                  labels=tf.placeholder("int32", [None, ]),
                  model_graph=model_graph)
model.freeze(['embeddings'])
model.compile(learning_rate=0.0001)
```

5）恢复参数、训练模型
```
model.restore(checkpoint_file)
model.auto_train(
        train_generator=train_generator,
        eval_generator=test_generator,
        save_path="result/topic",
        total_steps=20000,
        eval_step=1000,
        save_step=save_steps)
```

## 3.项目意义

进一步封装了神经网络训练方法，使得训练过程简洁直观，经过自己动手实践，加深了对网络训练的理解。
在自己的项目下开发，可以尝试搭建创新的网络结构，能够更好的对模型进行改进并验证。
