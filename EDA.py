import sys
# from utils import *
# from modin import pandas as pd
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', 500)
#显示所有行
pd.set_option('display.max_rows', 500)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
import gc,os
import matplotlib.pyplot as plt
fig,axes = plt.subplots(figsize = (14, 10))
import matplotlib
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import norm,skew,kurtosis #for some statistics
import IPython
from IPython import display
import sklearn
import random
import time
import pickle
import datetime
from tqdm import tqdm
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
%matplotlib inline
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


%%time
input_path = '../input/'
train_data = pd.read_csv(input_path+'train_dataset.csv',sep=',')#,names= ['user_id','register_day','register_type','device_type'],header = None,index_col=False, parse_dates=True,encoding='utf8'
test_data = pd.read_csv(input_path+'test_dataset.csv',sep=',')
submit_example = pd.read_csv(input_path+'submit_example.csv',sep=',')


# train_data = _reduce_mem_usage_(train_data)
# test_data = _reduce_mem_usage_(test_data)
print('train shape:',train_data.shape)
print('test shape:',test_data.shape)
print('sample shape:',submit_example.shape)

# train_data = train_data[train_data["happiness"]!=-8].reset_index(drop=True)  #去除某一类
train_data_copy = train_data.copy()
target_col = "信用分"
target = train_data_copy[target_col]#.apply(lambda x:np.log1p(x))
del train_data_copy[target_col]

train_shape = train_data.shape[0]
data = pd.concat([train_data_copy,test_data],axis=0,ignore_index=True)
data.head()
# data = data.fillna(-1)



train_data.info()

train_data.describe()


#----------------------------------------------------连续值target---------------------------------------------------
#target_col ana
target_col = "信用分"
plt.figure(figsize=(8,6))
plt.scatter(range(len(np.sort(train_data[target_col].values))), np.sort(train_data[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('yield', fontsize=12)
plt.show()

#We use the numpy fuction log1p which applies log(1+x) to all elements of the column
train_data[target_col] =(train_data[target_col])
#Check the new distribution
sns.distplot(train_data[target_col] , fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_data[target_col])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_data[target_col], plot=plt)
plt.show()


#analysis
def stat_df(df):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=False,inplace=True)
    return stats_df

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

missing_data(train_data)
stat_df(train_data)

stat_df(test_data)


print('data.shape:',data.shape)


#analysis columns
for col in data.columns:
    print('feature:',col)
    print('\t Percentage of missing values:\n\t\t train:',train_data[col].isnull().sum() * 100 / train_data.shape[0])
    print('\t\t test:',test_data[col].isnull().sum() * 100 / test_data.shape[0])

    print('\t feature unique:\n\t\t train:',train_data[col].nunique())
    print('\t\t test:',test_data[col].nunique())
    print('\t feature type:',data[col].dtype)

for col in data.columns:
    print('feature:',col)
    print('feature value_counts:\ntrain:\n',train_data[col].value_counts().head(10))
    print('test:\n',test_data[col].value_counts().head(10))
    print('-------------------------')

# 计算相关系数
cols = [col for col in train_data.columns if col not in [target_col] if train_data[col].dtype!='object']

labels = []
values = []
for col in cols:
    labels.append(col)
    values.append(np.corrcoef(train_data[col].values,train_data[target_col].values)[0,1])
# corr_df = pd.DataFrame('col_labels':labels,'corr_values':values)
corr_df = pd.DataFrame({'col_labels':labels,'corr_values':values})
corr_df = corr_df.sort_values(by = 'corr_values')

ind = np.arange(len(labels))
width=0.5
fig,ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind,np.array(corr_df['corr_values'].values),color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values,rotation='horizontal')
ax.set_xlabel('Correlation coefficient')
ax.set_title('Correlation cofficient of the variables')
# corr_df.columns
# np.array(corr_df['corr_values'].values)


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(30, 26))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    df = df.fillna(0)
    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .15},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 18}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
correlation_heatmap(train_data)


#-------------------------------------------------日期处理-------------------------------------------------
train_data ['time'] = pd.to_datetime(train_data ['time'])
train_data ['time'] = train_data ['time'].astype(datetime.datetime)
#根据时间戳获取天数、小时
train_df['day'] = train_df['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
train_df['hour'] = train_df['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
#根据时间戳转化为格式化时间
train["time_sting"]=train["context_timestamp"].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
train["time_sting"]=pd.to_datetime(train["time_sting"])
train['dayofweek'] = train['startTime'].dt.dayofweek.values + 1
train["hour"]=train["time_sting"].dt.hour
train["day"]=train["time_sting"].dt.day
train["date"]=train["time_sting"].dt.date
train['minutes'] = train['time_sting'].dt.minute
train["day"]=train["day"].apply(lambda x:0 if x==31 else x)
grouped_df = train.groupby(["day", "hour"])["is_trade"].aggregate("mean").reset_index()
grouped_df = grouped_df.pivot('day', 'hour', 'is_trade')

train['endTime'] = train['startTime'] + datetime.timedelta(minutes=10)

#生成时间段DataFrame
add_tmp = pd.DataFrame()
add_tmp['startTime'] = pd.date_range(start=last_value + timedelta(minutes=10),end=last_time, freq='10T')

#-------------------------------------------------缺失值处理-------------------------------------------------
all_data_na = (data.isnull().sum() / len(data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
# 1、填充众数、中位数
missing_column = 'missing'
train_data[missing_column][train_data[missing_column].isnull()] = train_data[missing_column].dropna().mode().values #众数
train_data[missing_column].fillna(train_data[missing_column].median(), inplace = True)#中位数
train_data[missing_column].fillna(train_data[missing_column].mean(), inplace = True)#均值
data[missing_column] = data.groupby("Neighborhood")[missing_column].transform(lambda x: x.fillna(x.median()))#分组后填充中位数
# 2、赋一个代表缺失的值
train_data.Cabin[train_data.Cabin.isnull()] = 'U0'
data[data.select_dtypes('object').columns.tolist()] = data.select_dtypes('object').fillna("-999")
data.fillna(-999, inplace=True)
# 3、使用模型预测



#-------------------------------------------------异常值处理-------------------------------------------------
#explor outliers
fig, ax = plt.subplots()
ax = plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()  #画出面积与价格的关系，删除异常值
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)#Deleting outliers

plt.plot(train_data['X'],train_data['Y'],'.')
plt.show()



#-------------------------------------------------可视化-------------------------------------------------
#不同label（离散）中feature（连续）分布
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10, 10, figsize=(18, 22))

    for feature in features:
        i += 1
        plt.subplot(10, 10, i)
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();
t0 = train_data.loc[train_data['target'] == 0]
t1 = train_data.loc[train_data['target'] == 1]
features = train_data.columns.values[:]
plot_feature_distribution(t0, t1, '0', '1', features)


#单变量的密度曲线
plt.figure(figsize=(28,26))
# 设置刻度字体大小
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
data[feature_col].plot(kind='kde')
data[feature_col].plot(kind = 'hist',bins = 70)
data[feature_col].hist(bins=20)
data[feature_col].boxplot(sym='r*',vert=False,patch_artist=True,meanline=False,showmeans=True) #boxplot 箱型图
data[feature_col].value_counts()[:15].plot(kind='barh')

sns.distplot(data[feature_col], bins=20,hist=False,kde=False, rug=True)
sns.kdeplot(data[feature_col],shade=True)
#观察多变量的密度曲线
plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],label = ['Survived','Dead'])
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
#violinplot 琴形图、boxplot 箱型图、barplot 条形图、countplot 计数图、distplot 条形图、scatter 散点图
sns.violinplot(x='type_day',y='online_hour', hue='school',data=data,grid = False)
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,palette={"male": "blue", "female": "pink"},markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)#x离散y连续

pp = sns.pairplot(data, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])

#降维可视化
import pandas as pd
import multiprocessing
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
df=pd.read_csv('1_total_fee.csv')
l=list(df['1_total_fee'].astype('str'))
name=list(df)

def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize= (10, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x, y), textcoords = 'offset points', ha = 'right', va = 'bottom')
    plt.savefig(filename)

tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000)

plot_only = 300
low_dim_embs = tsne.fit_transform(df.iloc[:plot_only][name[1:]])#前300行数据w2v特征可视化
labels = [l[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)