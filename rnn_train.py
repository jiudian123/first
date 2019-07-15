# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:34:01 2019

@author: zhu
"""

import os         #os模块调用系统命令
import time
from datetime import timedelta#计时用。timedelta代表两个datetime之间的时间差
import tensorflow as tf 

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


def train():
    tf.get_variable_scope().reuse_variables()
    #当前变量作用域可以用tf.get_variable_scope()进行检索并且reuse设置为True .
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textrnn'
    tf.summary.scalar("loss", model.loss)         #用来显示标量信息，一般在画loss,accuary时会用到这个函数。
    tf.summary.scalar("accuracy", model.acc)        #生成准确率标量图  
    merged_summary = tf.summary.merge_all()      #merge_all可以将所有summary全部保存到磁盘，以便tensorboard显示。
    writer = tf.summary.FileWriter(tensorboard_dir) #指定一个文件用来保存图。定义一个写入summary的目标文件，dir为写入文件地址
    # 配置 Saver
    saver = tf.train.Saver()#创建一个Saver对象用来保存模型
    
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # 创建session
    session = tf.Session()
    session.run(init)
    writer.add_graph(session.graph)#写入数据流图
    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):#轮次
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)#每批训练大小
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
train()