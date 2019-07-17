from sklearn import preprocessing
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm      #for i in tqdm(range(1, len))
import gc,os

#-------------------------------------------------变量转换  LabelEncoder-------------------------------------------------
data[feature_col] = pd.factorize(data[feature_col])[0]
data[feature_col+'code'] = LabelEncoder().fit_transform(dataset[feature_col])
# 编码，加速 LabelEncoder
for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
#one_hot feature
embark_dummies = pd.get_dummies(train_data['Embarked'],prefix='Embarked')
train_data = train_data.join(embark_dummies) #train_data = pd.concat([train_data, embark_dummies], axis=1)
train_data.drop(['Embarked'], axis=1,inplace=True)


#定量归一化(缩放)
scaler = preprocessing.StandardScaler() #RobustScaler,MinMaxScaler,Normalizer：
data['age_scaled'] = scaler.fit_transform(data['Age'].values.reshape(-1,1))


#利用qq图观察数据是否拟合正态分布,使用np.log1p对非正态特征值进行处理,对于线性模型至关重要
#train[feature_col] = np.log1p(train[feature_col])#做log处理
sns.distplot(data[feature_col] , fit=norm)
(mu, sigma) = norm.fit(data[feature_col]) #计算其平均和标准差
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
##如果数据拟合模型效果好，残差应该遵从正态分布(0,d*d:这里d表示残差)
#画出QQ图
fig = plt.figure()
res = stats.probplot(data[feature_col], plot=plt)
plt.show()


#计算数据集的偏度。
skewed_feats = data[feature_col].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

#boxcox1p:   y = ((1+x)**lmbda - 1) / lmbda if lmbda != 0;    log(1+x) if lmbda == 0
skewness = skewness[abs(skewness) > 0.75]  #对偏斜特征做boxcox1p变换
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    data[feat] = boxcox1p(data[feat], lam)




#Binning  （分箱）
data[feature_col] = pd.qcut(data[feature_col],5)
bins = [0, 12, 18, 65, 100];data[feature_col+'_bin'] = pd.cut(data[feature_col], bins)

#交互特征与多项式特征
poly = PolynomialFeatures(degree=10,include_bias=False)#多项式次数为10
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X)


#-------------------------------------------------处理特征-------------------------------------------------
#df相关操作
data[feature_col]= data[feature_col].astype('int')
data[feature_col] = data[feature_col].round(6)

data.loc[data['age']==0,'age'] = np.nan
data[feature_col].replace('\\N',np.nan,inplace=True)
data.drop_duplicates(subset = ['1_total_fee','2_total_fee','3_total_fee'],inplace=True,keep='first')

data.sort_index(by = ['user_id','register_day'],inplace = True)
data.sample(frac=1, random_state=2018)#sample

# 衍生特征
data['feat_kur'] = data[['feat1','feat2','feat3']].kurt() #skew
train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(4)]].min(axis = 1)
tmpall = tmpall.sort_values(['first','rate'],ascending=False)


def getres1(row):
    return len([x for x in row.values if type(x)==int and x<0])
data['neg1'] = data[data.columns].apply(lambda row:getres1(row),axis=1)  #某一行数据特征
data.loc[data['neg1']>20,'neg1'] = 20  #平滑处理


train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
mean_features = []
for f1 in cate_columns:
    for f2 in li:
        col_name = f1+"_"+f2+'_mean'
        mean_features.append(col_name)
        order_label = train.groupby([f1])[f2].mean()
        for df in [train, test]:
            df[col_name] = df[f1].map(order_label)
train.drop(li, axis=1, inplace=True)

#频率特征
unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in range(df_test.shape[1]):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


#group by 分组统计特征
feature = df_part[(df_part['date_index']>=5)&(df_part['behavior_type']==4)].groupby(['user_id'],as_index=False)['user_id'].agg({'u_b4_count_in_1':'count'})
data = pd.merge(data, feature, on=['user_id'], how='left').fillna(0)

data['use_reg_people'] = data.groupby(['register_type'])['user_id'].transform('count').values#直接生成统计特征
data = data.merge(data.groupby('use_feature',as_index=False)['count'].agg({'sum':'sum'}),on='use_feature',how='left')

def get_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['create_continue_day'] = continue_day(np.unique(row['day']))#最大连续天数
    return feature
def continue_day(day_list):
    day_list.sort()
    max_count = count = 1
    for day in day_list:
        if day+1 in day_list:
            count+=1
        else:
            max_count = max(max_count,count)
            count = 1
    return max_count
feature = df.groupby('user_id',sort = True).apply(get_feature)
data = pd.merge(data,pd.DataFrame(feature),on = 'user_id',how='left')

#w2v feature
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import multiprocessing
sentence = []
w2v_features = data.columns
for line in list(data[w2v_features].values):
    sentence.append([str(float(l)) for idx, l in enumerate(line)])
print('training...')
model = Word2Vec(sentence, size=100, window=5, min_count=1, sg=0, workers=multiprocessing.cpu_count(),iter=10)#size: 词向量的维度，默认值是100;window：即词向量上下文最大距离，默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
print('outputing...')
for fea in w2v_features:
    values = []
    for line in list(data[fea].values):
        values.append(line)
    values = set(values)
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
    data = pd.merge(data, df, on=fea, how='left')
    # out_df.to_csv(save_path + '/' + fea + '.csv', index=False)



#gmm feature
from sklearn.mixture import GaussianMixture
val = data.values#.reshape(-1,1)
gmm = GaussianMixture(n_components=32)#max_iter=3000,means_init=[[-1],[8],[12]],weights_init=[0.01,0.495,0.495])#, reg_covar=0.03
gmm.fit(val)
labels = gmm.predict(val)

#-------------------------------------------------时间序列特征--------------------------------------------------
#提取时间段的统计信息     10T 10minutes
tmp_df[f'minute_unique_for_'] = tmp.resample('10T', closed='left')['minutes'].nunique().values
tmp_df[f'minute_min_for_'] = tmp.resample('10T', closed='left')['minutes'].min().values
tmp_df[f'minute_max_for_'] = tmp.resample('10T', closed='left')['minutes'].max().values
tmp_df[f'minute_mean_for_'] = tmp.resample('10T', closed='left')['minutes'].mean().values
tmp_df[f'minute_skew_for_'] = tmp.resample('10T', closed='left')['minutes'].skew().values

#滑动数据
tmp['shift_b_for_in'] = data['inNums'].shift(1).values
tmp['shift_f_for_in'] = data['inNums'].shift(-1).values

#历史窗口
t = 6
tmp[f'roll_{t}_for_in_mean'] = data['inNums'].rolling( window=t).mean().values   #移动滑窗  前t个的平均
tmp[f'ewm_{t}_for_in_mean'] = data['inNums'].ewm(span=t).mean().values   #指数加权的移动窗口函数
tmp[f'roll_{t}_for_in_mean_center'] = data['inNums'].rolling(window=t, center=True).mean().values  #窗口的标签设置为居中  相当于移动滑窗前移


#时间序列加权
#将用户活跃天数视为二进制数
def get_binary_seq(now,start_date,end_date):
    day = list(range(1,end_date-start_date+2))
    day.reverse()
    ans1 = 0
    binary_day = []
    now_uni = now.unique()
    for i in day:
        if i in now_uni:
            binary_day.append(1)
        else:
            binary_day.append(0)
    return binary_day

#a.按二进制方式对活跃天数进行加权，越接近预测日期权重越高
def get_binary1(now,start_date,end_date): # Boss Feature
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(2**i)
    return ans

#b.直接按离预测日期距离进行加权
def get_binary2(now,start_date,end_date): # Boss Feature
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(1/(end_date-i))
    return ans

#对时间序列进行衰减系数编码(reference:https://github.com/luoda888/2018-KUAISHOU-TSINGHUA-Top13-Solutions)
def get_time_log_weight_sigma(now,start_date,end_date):
    window_len = end_date+1-start_date
    ans = np.zeros(window_len)
    sigma_ans = 0
    for i in now:
        ans[(i-1)%window_len] += 1
     for i in range(window_len):
        if ans[i]!=0:
            sigma_ans += np.log(ans[i]/(window_len-i))
    return sigma_ans


#------------------------------------------------------------------------特征选择--------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Filter<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#方差选择法
from sklearn.feature_selection import VarianceThreshold #方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
sele = VarianceThreshold(threshold=3)
sele.fit(data)#方差大于3 的特征
data.columns.values[sele.get_support()]


#pearson相关系数
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
#输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
#在此定义为计算相关系数 #参数k为选择的特征个数
# sele = SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T, k=5)
# sele.fit(X_train, y_train)
# data.columns.values[sele.get_support()]

# 卡方检验
from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# #选择K个最好的特征，返回选择特征后的数据
# sele = SelectKBest(chi2, k=2)
# sele.fit(X_train, y_train)
# data.columns.values[sele.get_support()]
SKB = SelectPercentile(chi2, percentile=95)
SKB.fit(X_train, y_train)
data.columns.values[SKB.get_support()]


# 互信息法
from sklearn.feature_selection import SelectKBest
from minepy import MINE
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
#选择K个最好的特征，返回特征选择后的数据
X_new = SelectKBest(lambda X, Y: np.array(list(map(lambda x:mic(x, Y), X.T))).T, k=2).fit_transform(X_train, y_train)


#SelectPercentile，selectKBest
from sklearn.feature_selection import SelectPercentile,selectKBest
select = SelectPercentile(percentile=50)
select.fit(X_train,y_train)
X_train_selected = select.transform(X_train)
select.get_support()    #查看哪些特征被选中


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Wrapper<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#递归特征消除法
from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import Ridge
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
rfe = RFE(estimator=Ridge(), n_features_to_select=35)
X_new = rfe.fit_transform(X_train, y_train)#Ridge
# X_new.shape
data.columns.values[rfe.get_support()]

# 递归特征消除法调参
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
for i in range(10, 65):
    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=i)
    rfe.fit(X_train, y_train)  # Ridge
    use_col = data.columns.values[rfe.get_support()]
    X_train_use = data[use_col][:train_shape].values
    X_test_use = data[use_col][train_shape:].values

    folds = KFold(n_splits=10, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(train_shape)
    predictions_rf = np.zeros(len(X_test_use))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_use, y_train)):
        print("fold n°{}".format(fold_ + 1))
        rf = RandomForestRegressor()
        trn_data = lgb.Dataset(X_train_use[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train_use[val_idx], y_train[val_idx])
        num_round = 10000
        clf = rf.fit(X_train_use[trn_idx], y_train[trn_idx])
        oof_lgb[val_idx] = clf.predict(X_train_use[val_idx])
        predictions_rf += clf.predict(X_test_use) / folds.n_splits
    print('use columns:', i)
    print("valid CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target) / 2))
    print("test my bast_score CV score: {:<8.8f}".format(mean_squared_error(predictions_rf, sample_res)))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>other<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# https://github.com/abhishekkrthakur/greedyFeatureSelection
#greedy feature selection based on ROC AUC    #classify

# PyFeast    contains JMI, BetaGamma, CMIM, CondMI, DISR, ICAP, and mRMR    need Linux or OS X
# https://github.com/mutantturkey/PyFeast
# from feast import *

# #https://github.com/EpistasisLab/scikit-rebate
# #ReliefF       classify

# https://github.com/scikit-learn-contrib/boruta_py

# https://github.com/luoda888/tianchi-diabetes-top12

#基于模型的特征选择
from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=42),threshold="median") #使用中位数作为阈值
select.fit(X_train,y_train)
X_train_l1 = select.transform(X_train)
X_test_l1 = select.transform(X_test)

#利用不同的模型来对特征进行筛选，选出较为重要的特征
















#--------------------------------------------------------特殊处理--------------------------------------------------------
#多进程处理
from multiprocessing import Pool
def parallelize_df_func(df, func, start, end, num_partitions=21, n_jobs=7):
    df_split = np.array_split(df, num_partitions)
    start_date = [start] * num_partitions
    end_date = [end] * num_partitions
    param_info = zip(df_split, start_date, end_date)
    pool = Pool(n_jobs)
    gc.collect()
    df = pd.concat(pool.map(func, param_info))
    pool.close()
    pool.join()
    gc.collect()
    return df





#样本不均衡处理
#下采样
def subsample(df, sub_size):
    if sub_size >= len(df):
        return df
    else:
        return df.sample(n=sub_size)
#直接读取数据，不分块
def k_means(index):
    '''using k_means to make clustering on negative samples (clusters_number ~ 1k)'''
    path_part_uic_label = "data/temp/uic/part_"+str(index)+"_uic.csv"

    path_feature_C = 'data/feature/part_' + str(index) + '_C.csv'

    path_part_uic_label_0 = "data/temp/k_means/df_part_"+str(index)+"_uic_label_0.csv"
    path_part_uic_label_1 = "data/temp/k_means/df_part_"+str(index)+"_uic_label_1.csv"

    path_part_uic_label_cluster = "data/temp/k_means/df_part_"+str(index)+"_uic_label_cluster.csv"
    path_part_scaler = "data/temp/k_means/df_part_"+str(index)+"_scaler"

    #读取正负样本集
    df_part_uic_label = pd.read_csv(path_part_uic_label, index_col=False)
    df_part_uic_label_0 = df_part_uic_label[df_part_uic_label['label'] == 0]
    df_part_uic_label_1 = df_part_uic_label[df_part_uic_label['label'] == 1]
    df_part_uic_label_0.to_csv(path_part_uic_label_0, index=False)
    df_part_uic_label_1.to_csv(path_part_uic_label_1, index=False)
    df_part_U = pd.read_csv(path_feature_U, index_col=False)


    scaler = preprocessing.StandardScaler()

    df_part_1_uic_label_0 = pd.read_csv(path_part_uic_label_0)
    # construct of part_1's sub-training set
    train_data_df_part_1 = pd.merge(df_part_1_uic_label_0, df_part_U, how='left', on=['user_id'])
    train_X_1 = train_data_df_part_1.as_matrix(
['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
'u_b1_count_in_3','uc_b_count_rank_in_u'])
# feature standardization
    standardized_train_X_1 = scaler.fit_transform(train_X_1)

    #将MiniBatchKMeans初始化
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=1000, batch_size=500, reassignment_ratio=10 ** -4,random_state=2018)
    classes_1 = []
    mbk.fit(standardized_train_X_1)
    classes_1 = np.append(classes_1, mbk.labels_)

    pickle.dump(scaler, open(path_part_scaler, 'wb'))

    df_part_uic_label_0 = pd.read_csv(path_part_uic_label_0, index_col=False)
    df_part_uic_label_1 = pd.read_csv(path_part_uic_label_1, index_col=False)

    #将分好的类别属性加入样本集中
    df_part_uic_label_0['class'] = classes_1.astype('int') + 1
    df_part_uic_label_1['class'] = 0
    df_part_uic_label_class = pd.concat([df_part_uic_label_0, df_part_uic_label_1])
    df_part_uic_label_class.to_csv(path_part_uic_label_cluster, index=False)


def sample(index):
    input_file_path = "data/temp/k_means/df_part_" + str(index) + "_uic_label_cluster.csv"
    part_uic_label_cluster = pd.read_csv(input_file_path)

    part_train_uic_label = part_uic_label_cluster[part_uic_label_cluster['class']==0].sample(frac=1,random_state=24)
    frac_ratio = 60 / 1200 #N/P balanced
    for i in range(1,1001,1):
        part_1_train_uic_label_1 = part_uic_label_cluster[part_uic_label_cluster['class']==i]
        part_1_train_uic_label_1 = part_1_train_uic_label_1.sample(frac=frac_ratio,random_state=24)
        part_train_uic_label = pd.concat([part_train_uic_label, part_1_train_uic_label_1])
    return part_train_uic_label




