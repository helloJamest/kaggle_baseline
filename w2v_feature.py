# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 4:08 PM
# @Author  : Inf.Turing
# @Site    : 
# @File    : w2v_feature.py
# @Software: PyCharm

# 预处理复赛数据
import os
import pandas as pd

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import multiprocessing
import numpy as np

L = 10

path = './'
save_path = path + '/w2v'
if not os.path.exists(save_path):
    print(save_path)
    os.makedirs(save_path)

train1 = pd.read_csv(path + '/train_all.csv')
train = pd.read_csv(path + '/train_2.csv')
test = pd.read_csv(path + '/test_2.csv')

data = pd.concat([train, test, train1]).reset_index(drop=True).sample(frac=1, random_state=2018).fillna(0)
data = data.replace('\\N', 999)
sentence = []
for line in list(data[['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']].values):
    sentence.append([str(float(l)) for idx, l in enumerate(line)])

print('training...')
model = Word2Vec(sentence, size=L, window=2, min_count=1, workers=multiprocessing.cpu_count(),
                 iter=10)
print('outputing...')

for fea in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
    values = []
    for line in list(data[fea].values):
        values.append(line)
    values = set(values)
    print(len(values))
    w2v = []
    for i in values:
        a = [i]
        a.extend(model[str(float(i))])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)

    name = [fea]
    for i in range(L):
        name.append(name[0] + 'W' + str(i))
    out_df.columns = name
    out_df.to_csv(save_path + '/' + fea + '.csv', index=False)
















import tensorflow as tf

def f(row):
    return str(row['id'])+','+'|'.join([str(x) for x in row['cat2']])
csv = (data.groupby('id')['cat2'].apply(list).reset_index()[['id','cat2']].apply(lambda row:f(row),axis=1)).values
def sparse_from_csv(csv):
  ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
  table = tf.contrib.lookup.index_table_from_tensor(
      mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
  split_tags = tf.string_split(post_tags_str, "|")
  return tf.SparseTensor(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)
TAG_EMBEDDING_DIM = 10
TAG_SET = data['cat2'].unique()
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))
tags = sparse_from_csv(csv)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)
with tf.Session() as s:
    s.run([tf.global_variables_initializer(), tf.tables_initializer()])
    cat2_fea = s.run([embedded_tags])
    # print(cat2_fea[0].shape)
    # print(cat2_fea)

cat2_fea_df = pd.DataFrame(cat2_fea[0],columns=['cat2_vec_'+str(i) for i in range(TAG_EMBEDDING_DIM)])
cat2_fea_df['id'] = data['id'].unique()
print('shape',cat2_fea_df.shape)
cat2_fea_df.head()
cat2_fea_df.to_csv('./cat2_fea_df.csv',index=False)

# http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/
