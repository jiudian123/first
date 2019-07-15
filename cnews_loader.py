# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:36:05 2018

@author: zhu
"""

from collections import Counter #从Collections 模块导入Counter类
import numpy as np #导入numpy模块
import tensorflow.contrib.keras as kr #导入keras模块
import jieba
import re

def open_file(filename,mode='r'):
    """
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')
 
def read_file(filename):
    """读取文件数据"""
    re_han = re.compile(u"([\u4E00-\u9FD5]+)")
    contents, labels = [], []        #创建两个列表
    with open_file(filename) as f:
        for line in f:
            try:
                line=line.rstrip()
                assert len(line.split('\t'))==2
                label,content=line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        seglist=jieba.lcut(blk)
                        word.extend([w for w in seglist ])#if w not in stopwords])
                contents.append(word)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    #data_train存储的是一个一个的文本

    all_data = []   #创建列表
    for content in data_train:
        all_data.extend(content)  #content添加到列表all_date中

    counter = Counter(all_data) #计数器，每个字符出现次数  
    count_pairs = counter.most_common(vocab_size - 1)
    #返回列表前4999个(元素,个数)
    words, _ = list(zip(*count_pairs))
    #zip()将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    #输出词汇表


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]#按行读入列表
    word_to_id = dict(zip(words, range(len(words))))
    #len（）计算列表的元素个数，range()产生从0开始计数的整数列表
    #dict（） 映射函数方式来构造字典
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '彩票', '房产', '家居', '教育', '科技', '时尚', '时政', '财经', '娱乐']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=500):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)#读取文件内容

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度，返回2维张量
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)#data_id是一个两层嵌套列表，max_length序列的最大长度
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
   #label_id为int数组，num_classes为标签类别总数
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    #np.arange均匀等分区间，它是一个序列，可被当做向量使用
    #permutation返回一个新的打乱顺序的数组，并不改变原来的数组
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
#yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器， 
#当你调用这个函数的时候，函数内部的代码并不立马执行 ，这个函数只是返回一个生成器对象
#当你使用for进行迭代的时候，函数中的代码才会执行

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
   
    file_r = open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
   
    with np.load(filename) as data:
        return data["embeddings"]
