 # -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:14:02 2018

@author: zhu
"""

import os         #os模块调用系统命令
import time
from datetime import timedelta#计时用。timedelta代表两个datetime之间的时间差

import numpy as np           #由多维数组对象和用于处理数组的例程集合
import tensorflow as tf 
from sklearn import metrics   #数据挖掘与分析，评估用

from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab,export_word2vec_vectors,get_training_word2vec_vectors

base_dir = r'F:\lstm\data\cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
vector_word_dir= os.path.join(base_dir, 'vector_word.txt')#vector_word trained by word2vec
vector_word_npz=os.path.join(base_dir, 'vector_word.npz')# save vector_word to numpy file
#最佳验证结果保存路径
save_dir = r'F:\lstm\checkpoints\textrnn'
save_path = os.path.join(save_dir, 'best_validation') 
#获取词典
config=TRNNConfig()
build_vocab(train_dir,vocab_dir)
words,word_to_id=read_vocab(vocab_dir)
categories,cat_to_id=read_category()
config.vocab_size = len(words)
if not os.path.exists(vector_word_npz):
   export_word2vec_vectors(word_to_id, vector_word_dir, vector_word_npz)
config.pre_trianing = get_training_word2vec_vectors(vector_word_npz)
model=TextRNN(config)

init=tf.global_variables_initializer()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)#当axis=1时，是在y_test列方向上找最大值
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 返回来一个给定形状和类型的用0填充的数组，用于保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = { 
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print ('精度:{0:.3f}'.format(metrics.precision_score(y_test_cls, y_pred_cls,average='micro')))
    print ('召回:{0:0.3f}'.format(metrics.recall_score(y_test_cls, y_pred_cls,average='micro')))
    print ('f1-score:{0:.3f}'.format(metrics.f1_score(y_test_cls, y_pred_cls,average='micro')))
    print("..................................")
    print(metrics.classification_report(y_test_cls, y_pred_cls,target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time) 
    print("Time usage:", time_dif)

test()

     