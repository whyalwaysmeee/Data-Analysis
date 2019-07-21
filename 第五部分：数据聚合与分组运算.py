import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import statsmodels.api as sm

df = DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':np.random.randn(5),'data2':np.random.randn(5)})
people = DataFrame(np.random.randn(5,5),columns = ['a','b','c','d','e'],index = ['Joe','Steve','Wes','Jim','Travis'])
tips = pd.read_csv('tips.csv')

#groupby技术
#要根据key1进行分组，计算data1列的平均值
#通过groupby得到一个groupb对象，它不能直接返回结果，但含有所需的所有信息
grouped = df['data1'].groupby(df['key1'])
#调用groupby的mean方法计算平均值
grouped.mean()
#一次性传入多个参数以获得更详细的分类
means = df['data1'].groupby([df['key1'],df['key2']]).mean()
#获得层次化索引
means.unstack()
#获得一个含有分组大小的Series
df.groupby(['key1']).size()

#对分组迭代
#产生一个二元元组，按key1进行分组
for name,group in df.groupby('key1'):
    print(name)
    print(group)
#使用多重键以达到更详细的分组
for (a1,a2),group in df.groupby(['key1','key2']):
    print (a1,a2)
    print(group)
#将数据片段做成字典
dict(list(df.groupby('key1')))

#选取一个或一组列
#计算data2的平均值并返回一个DataFrame
df.groupby(['key1','key2'])[['data2']].mean()
#对代码稍作修改，会发现结果也略有不同
df.groupby(['key1','key2'])['data2'].mean()

#通过字典或Series进行分组
people.ix[2:3,['b','c']] = np.nan
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
#把这个字典传给groupby
by_column = people.groupby(mapping,axis = 1)
#得到所需统计结果
by_column.sum()
by_column.mean()
#Series效果一样
map_series = Series(mapping)
people.groupby(map_series,axis = 1).count()

#通过函数进行分组和根据索引级别分组见书P270-271

#数据聚合
#查看按key1分类，data1数据里的样本分位数0.9
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)
#使用自己的聚合函数
def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)
#describe在这里也可用
grouped.describe()

#面向列的多函数应用
tips['tip_pct'] = tips['tip'] / tips['total_bill']
#按性别和是否是抽烟者，查看小费比例的平均值
grouped = tips.groupby(['sex','smoker'])
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
#传入一组函数或函数名，自动命名相应的列
grouped_pct.agg(['mean','std',peak_to_peak])
#自定义列名
grouped_pct.agg([('foo','mean'),('bar','std')])
#定义一组应用于全部列的函数
functions = ['count','mean','max']
result = grouped['tip_pct','total_bill'].agg(functions)
#只查看小费比例
result['tip_pct']
#自定义名称
functions = [('aaa','mean'),('bbb','var')]

#无索引
#禁用分组键作为索引
tips.groupby(['sex','smoker'],as_index = False).mean()

#分组及运算和转换
#添加一个按key1存放分组平均值的列
k1_means = df.groupby('key1').mean().add_prefix('mean_')
pd.merge(df,k1_means,left_on = 'key1',right_index  =True)
#用key去对应纵轴上的索引，然后进行汇总
key = ['one','two','one','two','one']
people.groupby(key).mean()
#使用transform将一个函数应用到各个分组，并将结果放在合适的位置
people.groupby(key).transform(np.mean)

#分位数和桶分析
frame = DataFrame({'data1':np.random.randn(1000),'data2':np.random.randn(1000)})
#将数据放入等长的4个桶中
factor = pd.cut(frame.data1,4)
#由cut返回的Factor对象可直接用于groupby
grouped = frame.data2.groupby(factor)
def get_stats(group):
    return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean()}
grouped.apply(get_stats).unstack()
#要想得到样本分位数大小相等(每组数据点数量相等)的桶
grouping = pd.qcut(frame.data1,10,labels = False)

#示例1：随机采样和排列
#构造一副英语型扑克牌
#构造花色
suits = ['H','S','C','D']
#构造大小
card_val = [1,2,3,4,5,6,7,8,9,10,11,12,13] * 4
#构造牌名
base_names = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
cards = []
#将花色和牌名结合
for suit in ['H','S','C','D']:
    cards.extend(str(num)+ suit for num in base_names)
deck = Series(card_val,index = cards)
#随机抽出5张
def draw(deck,n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
draw(deck)

#示例2：分组加权平均数和相关系数
close_px = pd.read_csv('stock_px.csv',parse_dates = True,index_col = 0)
#首先去除空值并计算日收益率，即（下一行的数值-上一行的数值）/上一行的数值
rets = close_px.pct_change().dropna()
#计算日收益率与SPX（标准普尔500指数）之间的年度相关系数
spx_corr = lambda x:x.corrwith(x['SPX'])
by_year = rets.groupby(lambda  x:x.year)
by_year.apply(spx_corr)
#苹果和微软的年度相关系数
by_year.apply(lambda g:g['AAPL'].corr(g['MSFT']))
#面向分组的线性回归
#这个函数对各数据块执行普通最小二乘法
def regress(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y,X).fit()
    return result.params
#按年计算AAPL对SPX收益率的线性回归
by_year.apply(regress,'AAPL',['SPX'])

#透视表
tips.pivot_table(index = ['sex','smoker'])
#根据sex和day分组，聚合tip_pct和size，margins添加分项小计
tips.pivot_table(['tip_pct','size'],index = ['sex','day'],columns = 'smoker',margins = True)


#示例3：2012联邦选举委员会数据库
fec = pd.read_csv('P00000001-ALL.csv')
#查看所有候选人姓名
unique_cands = fec.cand_nm.unique()
#添加党派
parties = {'Bachmann, Michelle':'Republican',
           'Romney, Mitt':'Republican',
           'Obama, Barack':'Democrat',
           "Roemer, Charles E. 'Buddy' III":'Republican',
           'Pawlenty, Timothy':'Republican',
           'Johnson, Gary Earl':'Republican',
           'Paul, Ron' :'Republican',
           'Santorum, Rick' :'Republican',
           'Cain, Herman':'Republican',
           'Gingrich, Newt':'Republican',
           'McCotter, Thaddeus G':'Republican',
           'Huntsman, Jon' :'Republican',
           'Perry, Rick':'Republican'}
#根据候选人姓名得到党派信息
fec.cand_nm[123456:123461].map(parties)
#将其添加为一个新列
fec['party'] = fec.cand_nm.map(parties)
#查看出资情况，包括退款，结果有正有负
(fec.contb_receipt_amt>0).value_counts()
#规定只能有正值
fec = fec[fec.contb_receipt_amt > 0]
#只查看奥巴马和罗姆尼的赞助信息
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]

#根据职业和雇主统计赞助信息（对两党派）
#清理职业信息
occ_mapping = {'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
               'INFORMATION REQUESTED':'NOT PROVIDED',
               'INFORMATION REQUESTED(BEST EFFORTS)':'NOT PROVIDED',
               'C.E.O.':'CEO'}
f = lambda x:occ_mapping.get(x,x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
               'INFORMATION REQUESTED':'NOT PROVIDED',
               'SELF':'SELF-EMPLOYED',
               'SELF EMPLOYED':'SELF-EMPLOYED',}
f = lambda x:emp_mapping.get(x,x)
fec.contbr_employer = fec.contbr_employer.map(f)
#通过透视表根据党派和职业对数据进行聚合
by_occu = fec.pivot_table('contb_receipt_amt',index = 'contbr_occupation',columns = 'party',aggfunc = 'sum')
#滤掉不足200万美元的数据
over_2mm = by_occu[by_occu.sum(1) > 2000000]
#用柱状图来表示
over_2mm.plot(kind='barh')

#对出资额分组
#划分组
bins = np.array([0,1,10,100,1000,10000,100000,1000000,10000000])
#利用cut根据出资额的大小将数据分到每个面元中
labels = pd.cut(fec_mrbo.contb_receipt_amt,bins)
#根据候选人姓名及面元标签对数据进行分组（赞助人数）
grouped = fec_mrbo.groupby(['cand_nm',labels])
grouped.size()
#unstack用于构建层次化索引
grouped.size().unstack(0)
#对每个面元内的出资额求和
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
#进一步规格化，显示比例
normed_sums = bucket_sums.div(bucket_sums.sum(axis = 1),axis = 0)
#排除两个最大的面元（非个人捐赠）并作图
normed_sums[:-2].plot(kind = 'barh',stacked = True)

#根据州统计赞助信息
grouped = fec_mrbo.groupby(['cand_nm','contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals[totals.sum(1)>100000]
#各候选人在各州的总赞助比例
totals.div(totals.sum(1),axis = 0)
















