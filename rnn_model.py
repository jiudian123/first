# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:39:46 2018

@author: zhu
"""

import tensorflow as tf
    
class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim =64     # 词向量维度
    seq_length = 500        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表大小
    hidden_dim = 128        # 隐藏层神经元 
    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率
    batch_size = 64         # 每批训练大小
    num_epochs = 10          # 总迭代轮次
    pre_trianing=None       #预训练词向量
    num_layers=1             # 隐藏层层数  
    rnn = 'lstm'             # lstm
    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
         #第一个参数表示成员类型，[None,784]是tensor的shape， None表示第一维是任意数量，784表示第二维是784维
        self.rnn()

    def rnn(self):
        """rnn模型"""
        
        def lstm_cell():   # lstm核
            return tf.contrib.rnn.LSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0') ,tf.name_scope('embedding'):
            embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],
                                         initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
                #查找张量中的序号为x的
       
        with tf.name_scope("cell"):#tf.name_scope当传入字符串时，用以给变量名添加前缀，类似于目录
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            #用于构建多层循环神经网络，接受和返回的states是n-tuples n元序列，其中n=len(cells)
            
        with tf.name_scope("rnn"):
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')#添加全链接层
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)#dropout
            fc = tf.nn.relu(fc)#relu将大于0的数保持不变，小于0的数置为0

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            #Softmax把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
            #axis=1将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组
        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #for v in tf.trainable_variables():
             #   print (v)
            '''
            w1=tf.get_default_graph().get_tensor_by_name('rnn/multi_rnn_cell/cell_0/lstm_cell/weights:0')
            tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.01)(w1))
            error= tf.reduce_mean(cross_entropy)#reduce_mean()就是按照某个维度求平均值
            tf.add_to_collection('losses', error) 
            self.loss = tf.add_n(tf.get_collection('losses'))
            '''
            self.loss =tf.reduce_mean(cross_entropy)
              # 优化器，建Adam优化器
        with tf.name_scope("optimition"):     
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)#判相等
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
