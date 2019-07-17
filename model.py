from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
import os
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import  KFold, StratifiedKFold,GroupKFold, RepeatedKFold
import logging
# from utils import *

logging.basicConfig(level=logging.DEBUG, filename="baseline_logfile_1_15",
                    filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")


label2current_service =dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label =dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))
train['current_service'] = train['current_service'].map(current_service2label)
res[target] = res[target].map(label2current_service)

train_shape = train_data.shape[0]


use_fea = [clo for clo in data.columns if clo!='id' and data[clo].dtype!=object]
features = data[use_fea].columns
X_train = data[:train_shape][use_fea].values
y_train = target
X_test = data[train_shape:][use_fea].values

#-----------------------------------------------------------------------------模型训练-------------------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ligtgbm<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#use csr matrix
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0

base_train_csr = sparse.csr_matrix((len(train_x), 0))
base_predict_csr = sparse.csr_matrix((len(predict_x), 0))
enc = OneHotEncoder()
for feature in cate_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr','bool')
    base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),'csr','bool')

train_csr = sparse.hstack((sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype('float32')
predict_csr = sparse.hstack((sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')



lgb_model = lgb.LGBMClassifier(
boosting_type='gbdt', num_leaves=61, reg_alpha=3, reg_lambda=1,
max_depth=-1, n_estimators=5000, objective='binary',
subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
learning_rate=0.035, random_state=2018, n_jobs=10
)
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)

best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
     eval_set=[(train_csr[train_index], train_y[train_index]),
     (train_csr[test_index],train_y[test_index])],early_stopping_rounds=200, verbose=10)

    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv("../res/lgb_baseline_%s.csv" % now, index=False)



#classify
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
# 'objective': 'multiclass',
# 'num_class':3,
    'verbosity': 1
}

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train_x))
predictions = np.zeros(len(train_x))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx], label=train_y[trn_idx])
    val_data = lgb.Dataset(train_x[val_idx], label=train_y[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=3000)
    oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(predict_x, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))



#regression
param = {'num_leaves': 120,
'min_data_in_leaf': 30,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.025,
"min_child_samples": 30,
"boosting": "gbdt",
"feature_fraction": 0.9,
"bagging_freq": 1,
"bagging_fraction": 0.9 ,
"bagging_seed": 2029,
"metric": 'mse',
"lambda_l1": 0.1,
"verbosity": -1}
folds = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))




#---------------特征重要性
df = pd.DataFrame(data[use_fea].columns.tolist(), columns=['feature'])
df['importance']=list(clf.feature_importance())
df = df.sort_values(by='importance',ascending=False)
df

plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="feature", data=df.head(50))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()

# without cv
def lgb_train(train_, valid_, pred, label, fraction):
    print(f'data shape:\ntrain--{train_.shape}\nvalid--{valid_.shape}')
    print(f'days:\ntrain--{train_.day.sort_values().unique()}\nvalid--{valid_.day.unique()}')
    print(train_['dayofweek'].unique())
    print(f'Use {len(pred)} features ...')
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'feature_fraction': fraction,
        'bagging_fraction': fraction,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'reg_sqrt': True,
        'nthread': -1,
        'verbose': -1,
    }

    dtrain = lgb.Dataset(train_[pred], label=train_[label], categorical_feature=['stationID'])
    dvalid = lgb.Dataset(valid_[pred], label=valid_[label], categorical_feature=['stationID'])

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dvalid],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    return clf


def lgb_sub_all(train_, test_, pred, label, num_rounds, fraction, add_rounds, model_file):
    print(f'data shape:\ntrain--{train_.shape}\ntest--{test_.shape}')
    print(f'days:\ntrain--{train_.day.sort_values().unique()}\ntest--{test_.day.unique()}')
    print(f'Use {len(pred)} features and {num_rounds} rounds...')
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'feature_fraction': fraction,
        'bagging_fraction': fraction,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'reg_sqrt': True,
        'nthread': -1,
        'verbose': -1,
    }

    dtrain = lgb.Dataset(train_[pred], label=train_[label], categorical_feature=['stationID'])

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain],
        num_boost_round=num_rounds+add_rounds,
        verbose_eval=500
    )
    test_[label] = clf.predict(test_[pred])
    clf.save_model(f'{model_file}_{label}.txt')



#---------------特征重要性
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
df = pd.DataFrame(data[use_fea].columns.tolist(), columns=['feature'])
df['importance']=list(clf.feature_importance())
df = df.sort_values(by='importance',ascending=False)
df#.head()



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>xgboost<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#classify
from xgboost import XGBClassifier

#自定义评价函数
def f1_error(preds, dtrain):
    label = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    pred = [int(i >= 0.5) for i in preds]
    tp = sum([int(i == 1 and j == 1) for i, j in zip(pred, label)])
    precision = float(tp) / sum(pred)
    recall = float(tp) / sum(label)
    return 'f1-score', 2 * (precision * recall / (precision + recall))

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8,
colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)



# regression
xgb_params = {'eta': 0.01, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'gamma': 0,
              'lambda': 1, 'alpha': 0,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': 2019}
# 自定义评价函数
def MSE(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score
folds = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(X_train))
predictions_xgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=MSE)
    #     clf = xgb.train(dtrain=trn_data, num_boost_round=200, evals=watchlist, verbose_eval=100, params=xgb_params,feval = myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
mean_squared_error(target.values, oof_stack)

#-----------------------------------------------------------------------------模型调参-------------------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ligtgbm<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
grid_param = [
[{
'n_estimators': grid_n_estimator, #default=50
'learning_rate': grid_learn, #default=1
'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
'random_state': grid_seed
}]
]
best_search = model_selection.GridSearchCV(AdaBoostClassifier(), param_grid = param, cv = cv_split, scoring = 'roc_auc')
best_search.fit(data1[data1_x_bin], data1[Target])
run = time.perf_counter() - start
best_param = best_search.best_params_
print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>xgboost<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# xgb调参   XGBRegressor
# 第一步：确定学习速率和tree_based 参数调优的估计器数目n_estimators
from sklearn.model_selection import GridSearchCV
cv_params = {'n_estimators': [100,200, 500, 700,1000,1300],
             'learning_rate': [0.002,0.005]}
other_params = { 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,'min_child_weight' :1,'gamma':0,'lambda':1, 'alpha':0,
'objective': 'reg:linear', 'eval_metric': 'rmse','seed':2019}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(X_train, y_train)
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#使用热图可视化
results = pd.DataFrame(optimized_GBM.cv_results_)
results.head()
import mglearn
scores = np.array(results.mean_test_score).reshape(3,3)
mglearn.tools.heatmap(scores,xlabel='n_estimators',xticklabels=cv_params['n_estimators'],ylabel='learning_rate',yticklabels=cv_params['learning_rate'],cmap='viridis')

# 进一步探索参数max_depth，min_child_weight
# max_depth subsample min_child_weight colsample_bytree
param_test1 = {
    'max_depth':list(range(3,10,2)),
    'min_child_weight':list(range(1,6,2))
}
#继续搜索参数min_child_weight
param_test2b = {
    'min_child_weight':[2,4,6,8]
}
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}

#调整参数subsample，colsample_bytree
#Grid seach on subsample and max_features
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>KNeighborsClassifier<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)






















#-----------------------------------------------------------------模型评估----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"学习曲线", X, y)


#Accuracy Summary Report
print(metrics.classification_report(data1['Survived'], Tree_Predict))


import itertools
#自定义显示混淆矩阵预测图，正则化
'''
cm:数据
normalize：是否正则化
title：名称
cmap：颜色
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    函数功能展示混淆矩阵
    支持正则化
    """
    if normalize:#是否在正则化，除以行数据之和
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict)
np.set_printoptions(precision=4)#这是显示的精确度

class_names = ['Dead', 'Survived']
# Plot non-normalized confusion matrix
#非正则化混淆矩阵
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#正则化混淆矩阵
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')




#随机划分15%作为测试/取最后2个小时作为测试
def LGB_test(train_x,train_y,test_x,test_y,cate_col=None):
    if cate_col:
        data = pd.concat([train_x, test_x])
        for fea in cate_col:
            data[fea]=data[fea].fillna('-1')
            data[fea] = LabelEncoder().fit_transform(data[fea].apply(str))
        train_x=data[:len(train_x)]
        test_x=data[len(train_x):]
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25,random_state=2018,n_jobs=50
    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y),(test_x,test_y)],early_stopping_rounds=100)
    feature_importances=sorted(zip(train_x.columns,clf.feature_importances_),key=lambda x:x[1])
    return clf.best_score_[ 'valid_1']['binary_logloss'],feature_importances





































