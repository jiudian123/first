# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:17:21 2019

@author: zhu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:14:50 2019

@author: zhu
"""

           #encoding:utf-8
import logging
import time
import re
import jieba
from gensim.models import word2vec

re_han= re.compile(u"([\u4E00-\u9FD5]+)") # 只保留文字

class Get_Sentences(object):
    def __init__(self,filenames):
        self.filenames= filenames

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('\t')
                        assert len(line)==2
                        blocks=re_han.split(line[1])
                        word=[]
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend( jieba.lcut(blk))
                        yield word
                    except:
                        pass

def train_word2vec(filenames):
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, sg=0,size=64,window=5, min_count=10, workers=6,iter=5)
    model.wv.save_word2vec_format('F:/lstm/data/cnews/vector_word.txt', binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

if __name__ == '__main__':
   filenames=[r'F:\lstm\data\cnews\cnews.train.txt',r'F:\lstm\data\cnews\cnews.val.txt',r'F:\lstm\data\cnews\cnews.test.txt']

train_word2vec(filenames)