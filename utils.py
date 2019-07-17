#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 23:08:26 2018

@author: bmj
"""

import gc
import time
from time import strftime,gmtime
import numpy as np
import pandas as pd
import os
import itertools
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
import warnings
from sklearn.preprocessing import LabelEncoder
from utils import *
from datetime import datetime
from datetime import timedelta
#test

warnings.simplefilter(action='ignore', category=FutureWarning)
USE_KFOLD = True

data_path = './input/'



load = False
cache_path = './cache3/'

from time import strftime,gmtime


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger()



def _reduce_mem_usage_(df, verbose=True):
    # logger.info('Reduce mem usage')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#将groupby的内容格式化输出dataframe
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result


def get_feat_size(train,size_feat):
    """计算A组的数量大小（忽略NaN等价于count）"""
    result_path = cache_path +  ('_').join(size_feat)+'_feat_count'+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        result = train[size_feat].groupby(by=size_feat).size().reset_index().rename(columns={0: ('_').join(size_feat)+'_count'})
        result = left_merge(train,result,on=size_feat)
    return result


def get_feat_size_feat(train,base_feat,other_feat):
    """计算唯一计数（等价于unique count）"""
    result_path = cache_path + ('_').join(base_feat)+'_count_'+('_').join(other_feat)+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        result = train[base_feat].groupby(base_feat).size().reset_index()\
                      .groupby(other_feat).size().reset_index().rename(columns={0: ('_').join(base_feat)+'_count_'+('_').join(other_feat)})
        result = left_merge(train,result,on=other_feat)
    return result

#分组计算统计量
def get_feat_stat_feat(train,base_feat,other_feat,stat_list=['min','max','var','size','mean','skew']):
    name = ('_').join(base_feat) + '_' + ('_').join(other_feat) + '_' + ('_').join(stat_list)
    result_path = cache_path + name +'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        agg_dict = {}
        for stat in stat_list:
            agg_dict[name+stat] = stat
        result = train[base_feat + other_feat].groupby(base_feat)[",".join(other_feat)]\
        .agg(agg_dict)
        result = left_merge(train,result,on=base_feat)
    return result

def get_feat_ngroup(train,base_feat):
    name = ('_').join(base_feat)+'_ngroup'
    result_path = cache_path + ('_').join(base_feat)+'_ngroup'+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        train[name] = train.groupby(base_feat).ngroup()
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)        
    return result

#----------------------------------------pipeline--------------------------------------#



#构建模型



#对count在n以上的进行onehot编码
def one_hot_encoder(train,column,n=100,nan_as_category=False):
    tmp = train[column].value_counts().to_frame()
    values = list(tmp[tmp[column]>n].index)
    train.loc[train[column].isin(values),column+'N'] = train.loc[train[column].isin(values),column]
    train =  pd.get_dummies(train, columns=[column+'N'], dummy_na=False)
    return train

#RMSLE
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'macro_f1_score', score_vali, True


def evaluate_macroF1_lgb(data_vali, preds):
    labels = data_vali.astype(int)
    preds = np.array(preds)
    preds = np.argmax(preds, axis=1)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return score_vali


def kfold_lightgbm(params, df, predictors, target, num_folds, stratified=True,
                   objective='', metrics='', debug=False,
                   feval=f1_score_vali, early_stopping_rounds=100, num_boost_round=100, verbose_eval=50,
                   categorical_features=None, sklearn_mertric=evaluate_macroF1_lgb):
    lgb_params = params

    train_df = df[df[target].notnull()]
    test_df = df[df[target].isnull()]

    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df[predictors].shape,
                                                                      test_df[predictors].shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1234)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1234)
    # folds = GroupKFold(n_splits=5)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros((train_df.shape[0], 11))
    sub_preds = np.zeros((test_df.shape[0], 11))
    feature_importance_df = pd.DataFrame()
    feats = predictors
    cv_resul = []
    '''
    perm = [i for i in range(len(train_df))]
    perm = pd.DataFrame(perm)
    perm.columns = ['index_']

    for n_fold in range(5):
        train_idx = np.array(perm[train_df['cv'] != n_fold]['index_'])
        valid_idx = np.array(perm[train_df['cv'] == n_fold]['index_'])
    '''
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target])):
        if (USE_KFOLD == False) and (n_fold == 1):
            break
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]

        train_x = pd.concat([train_x, train_old[feats]])
        train_y = pd.concat([train_y, train_old[target]])

        train_y_t = train_y.values
        valid_y_t = valid_y.values
        print(train_y_t)
        xgtrain = lgb.Dataset(train_x.values, label=train_y_t,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )
        xgvalid = lgb.Dataset(valid_x.values, label=valid_y_t,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )

        clf = lgb.train(lgb_params,
                        xgtrain,
                        valid_sets=[xgvalid],  # , xgtrain],
                        valid_names=['valid'],  # ,'train'],
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=verbose_eval,
                        #                         feval=feval
                        )

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        gain = clf.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature': clf.feature_name(),
                                           'split': clf.feature_importance('split'),
                                           'gain': 100 * gain / gain.sum(),
                                           'fold': n_fold,
                                           }).sort_values('gain', ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        result = evaluate_macroF1_lgb(valid_y, oof_preds[valid_idx])
        #        result = clf.best_score['valid']['macro_f1_score']
        print('Fold %2d macro-f1 : %.6f' % (n_fold + 1, result))
        cv_resul.append(round(result, 5))
        gc.collect()

    score = 'model_1'
    # score = np.array(cv_resul).mean()
    if USE_KFOLD:
        # print('Full f1 score %.6f' % score)
        for i in range(11):
            train_df["class_" + str(i)] = oof_preds[:, i]
            test_df["class_" + str(i)] = sub_preds[:, i]
        train_df[['user_id'] + ["class_" + str(i) for i in range(11)]].to_csv('./cv/val_prob_{}.csv'.format(score),
                                                                              index=False, float_format='%.4f')
        test_df[['user_id'] + ["class_" + str(i) for i in range(11)]].to_csv('./cv/sub_prob_{}.csv'.format(score),
                                                                             index=False, float_format='%.4f')
        oof_preds = [np.argmax(x) for x in oof_preds]
        sub_preds = [np.argmax(x) for x in sub_preds]
        train_df[target] = oof_preds
        test_df[target] = sub_preds
        print(test_df[target].mean())
        train_df[target] = oof_preds
        train_df[target] = train_df[target].map(label2current_service)
        test_df[target] = sub_preds
        test_df[target] = test_df[target].map(label2current_service)
        print('all_cv', cv_resul)

        train_df[['user_id', target]].to_csv('./sub/val_{}.csv'.format(score), index=False)
        test_df[['user_id', target]].to_csv('./sub/sub_{}.csv'.format(score), index=False)
        print("test_df mean:")

    display_importances(feature_importance_df, score)


def display_importances(feature_importance_df_, score):
    ft = feature_importance_df_[["feature", "split", "gain"]].groupby("feature").mean().sort_values(by="gain",
                                                                                                    ascending=False)
    print(ft.head(60))
    ft.to_csv('importance_lightgbm_{}.csv'.format(score), index=True)
    cols = ft[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]


####################################计算#################################################################

'''
params = {
    'metric': 'multi_logloss',
    'num_class': 11,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'feature_fraction': 0.7,
    'learning_rate': 0.02,
    'bagging_fraction': 0.7,
    # 'bagging_freq': 2,
    'num_leaves': 64,
    'max_depth': -1,
    'num_threads': 16,
    'seed': 2018,
    'verbose': -1,
    # 'is_unbalance':True,
}

categorical_columns = [
    'contract_type',
    'net_service',
    'gender']
for feature in categorical_columns:
    print('Transforming {feature}...')
    encoder = LabelEncoder()
    train[feature] = encoder.fit_transform(train[feature].astype(str))

x = []
no_use = ['current_service', 'user_id', 'group'] + x

categorical_columns = []
all_data_frame = []
all_data_frame.append(train)

for aresult in result:
    all_data_frame.append(aresult)

train = concat(all_data_frame)
feats = [f for f in train.columns if f not in no_use]
categorical_columns = [f for f in categorical_columns if f not in no_use]

train_old = train.iloc[shape1:shape2]
train = train.iloc[:shape1]
# train = train[train.service_type!=1]
# train_old = train_old[train_old.service_type!=1]
clf = kfold_lightgbm(params, train, feats, 'current_service', 5, num_boost_round=4000,
                     categorical_features=categorical_columns)

tmp_res[tmp_res['count']>10].to_csv(output_path + outname, columns = ['first','second'],index = False)
'''






