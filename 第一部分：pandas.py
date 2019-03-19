import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pandas.io.data as web

#创建Series数据结构,默认索引为0,1,2......
obj = Series([1,2,3,4])

#创建字典
data = {'state':['ohio','ohio','ohio','nevada','nevada'],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5,1.7,3.6,2.4,2.9]}

#创建Dataframe数据结构，可将实现创建的字典的内容直接导入，默认索引为0,1,2......
frame = DataFrame(data,columns=['pop', 'state', 'year'])

#自定义添加索引
frame2 = DataFrame(data,columns=['pop', 'state', 'year', 'debt'],
                           index=['a', 'b', 'c', 'd', 'f'])

#将Dataframe的列获取为一个Series
a = frame2.debt

#改变列的值
frame2['debt'] = 16.5

#将Series赋值给某个列,会精确匹配索引，空位会填上NaN
val = Series([-1.2,-1.5,-1.7],index = ['two','four','five'])
frame2['debt'] = val

#将嵌套的字典传给Dataframe，外层键将作为列，内层键将作为行
pop = {'nevada':{2001:2.4,2002:2.9},'ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3 = DataFrame(pop)

#转置
frame3.T

#指定索引
frame3 = DataFrame(pop,index = [2005,2006,2007])

#以二维ndarray的形式返回Dataframe中的数据
frame3.values

#reindex方法,改变索引值（对Dataframe同样适用）
obj = Series([4,7,3,1],index = ['d','b','c','a'])
obj.reindex(['a','b','c','d','e'],fill_value = 0)
#method实现填充值
obj3 = Series(['blue','pruple','yellow'],index = [0,2,4])
obj3.reindex(range(6),method = 'ffill')

#drop方法，丢弃指定项（对Dataframe同样适用）
obj = Series(range(3),index = ['a','b','c'])
newobj = obj.drop('b')
print(newobj)

#索引、选取和过滤
data = DataFrame(np.arange(16).reshape((4,4)),index = ['ohio','colorado','utah','new york'],columns = ['one','two','three','four'])
d1 = DataFrame(np.arange(12.).reshape((4,3)),columns = list('bde'),index = ['utah','ohio','texas','oregon'])
series2 = Series(range(3),index = ['b','e','f'])
#提取'two'和'three'列
data['two','three']
#提取前两行
data[:2]
#从'one'列中第一个大于2的行开始提取
data[data['one']>2]
#把所有小于5的值变成0
data[data<5] = 0
#ix标签索引
data.ix['Colorado',['two','three']]
data.ix[['Colorado','Utah'],[3,0,1]]
#提取第三行,并以Series的形式返回
data.ix[2]
#提取前三行
data.ix[:3]
#提取前四行的第三列
data.ix[:4,2]
#提取前四行的前三列
data.ix[:4,:3]
#修改前四行的前三列的值
data.ix[:4,:3] = None
#提取第二行至最后一行
data.ix[2:]
#提取第三行至最后一行
data[2:]
#广播（对行）:类似于excel中的下拉功能
data - data.ix[0]
series = d1.ix[0]
#减一个series的情况下只有第一行减
d1-series
#减多个series的情况所有行都要减，有几个减几个
d1-series*3
#相加：如果某个索引值在DataFrame或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集
frame + series2
#广播（对列），必须使用算数方法
frame.sub(series,axis = 0)

#函数应用和映射
d2 = DataFrame(np.random.randn(4,3),columns = list('bde'),index = ['utah','ohio','texas','oregon'])
#取绝对值
d3 = np.abs(d2)
f = lambda x:x.max() - x.min()
#对列应用函数
d2.apply(f)
#对行应用函数
d2.apply(f,axis = 2)
#返回多个值组成的Series
def g(x):
    return Series([x.min(),x.max()], index = ['min','max'])
d3.apply(g)

#排序和排名
#排序
obj = Series(range(4),index = ['d','a','b','c'])
d1 = DataFrame(np.arange(12.).reshape((4,3)),columns = list('bde'),index = ['utah','ohio','texas','oregon'])
#默认按索引的字典顺序升序排序
obj.sort_index()
#对DataFrame可根据任意一个轴上的索引排序（同样遵循字典顺序）
d1.sort_index()
#降序
d1.sort_index(axis = 1 , ascending = False)
#若要对值进行排序（缺失值会被放到末尾）
obj.order()
d1.sort_index(by = 'b')
#排名
obj = Series([7,-5,7,4,2,0,4])
#通过“为各组分配一个平均排名”的方式破坏平级关系，此结果是小数
obj.rank()
#若按值在原数据中出现的顺序排名
obj.rank(method = 'first')
#对行排名
d1.rank(axis = 1)

#带有重复的轴索引
obj = Series(range(5),index = list('aabbc'))
#检查索引是否唯一
obj.index.is_unique
#选取数据：若索引对应多个值，则返回一个Series，否则返回一个标量值
obj['a']


#汇总和计算描述统计
df = DataFrame([[1,2],[3,4],[5,None],[7,8]], index = list('abcd'),columns= ['one','two'])
#求列的和，返回一个含有小计的Series（NA值会被自动排除）
df.sum()
#求行的和
df.sum(axis = 1)
#一次性产生多个汇总统计（DataFrame），包括：
#count 非NA值的数量
#min，max 最大值和最小值
#quantile 样本的分位数（自行设定一个百分比，25%，50%，75%......）
#std  样本的标准差
#mean 样本的平均值
df.describe()
#其他操作：
#argmin，argmax 最大值和最小值的索引位置（整数）
#idxmin，idxmax 最大值和最小值的索引值
#sum 求和
#median 中位数
#mad 根据平均值计算平均绝对离差
#var 方差
#skew 偏度（三阶矩）
#kurt 峰度（四阶矩）
#cumsum 从上往下的累计和
#cummin，cummax 累计最大值和累计最小值
#cumprod 累计积
#diff 一阶差分（对时间序列有用）
#pct_change 百分数变化

#相关系数与协方差
all_data = {}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000','1/1/2010')
price = DataFrame({tic:data['Adj Close'] for tic, data in all_data.iteritems()})
volumn = DataFrame({tic:data['Adj Close'] for tic, data in all_data.iteritems()})
#价格的百分数变化
returns = price.pct_change()
returns.tail()
#计算两个Series中重叠的、非NA的、按索引对齐的值得相关系数
returns.MSFT.corr(returns.IBM)
#类似的，计算协方差
returns.MSFT.cov(returns.ISM)
#计算完整的相关系数矩阵
returns.corr()
#计算完整的协方差矩阵
returns.cov()
#计算列或行跟另一个Series之间的相关系数
returns.corrwith(returns.IBM)
#类似的，计算跟另一个DataFrame之间的相关系数
returns.corrwith(volumn)

obj = Series(['c','a','c','d','a','b','b','c'])
df = DataFrame([[1,2],[3,4],[5,6],[7,8]], index = list('abcd'),columns= ['one','two'])
#一个Series中的唯一值数组
obj.unique()
#计算各值出现的频率，按降序排列
obj.value_counts()
#应用于DataFrame，须要加“pd.”
df.apply(pd.value_counts).fillna(0)
#判断矢量化集合的成员资格，也就是Series各值是否包含传入的参数
mask = obj.isin(['b','c'])
obj(mask)


#处理缺失数据（对Series）,None值同样会被当做NaN处理
#滤除缺失数据
s1 = Series(['a','b','v',np.nan])
s = DataFrame([[1,2,None],[3,None,None],[5,6,None],[7,8,None],[9,10,11]],columns = ['a','b','c'])
#返回一个仅含非空数据和索引值的Series
s1.dropna()
#或者
s1.notnull()
#处理缺失数据（对DataFrame）
#默认丢弃任何含有缺失值的行
s.dropna()
#只丢弃全是NA的行
s.dropna(how = 'all')
#以相同方式丢弃列
s.dropna(how = 'all', axis = 1)
#保留至少含有3个非NA值的行
s.dropna(thresh = 3)
#保留至少含有3个非NA值的列
s.dropna(thresh = 3, axis = 1)

#填充缺失数据
s = DataFrame([[1,2,None],[3,None,None],[5,6,None],[7,8,9],[9,10,11]],columns = ['a','b','c'],index = [1,2,3,4,5])
#全部填充为5
s.fillna(5)
#‘b’列填充为4，‘c’列填充为6
s.fillna({'b':4,'c':6})
#将该列的缺失值全部填充为该列第一行的值，若第一行为空，则还是空
s.fillna(method = 'ffill')
#对填充的值的个数进行限制
s.fillna(method = 'ffill',limit = 2)


#层次化索引
s = Series(np.random.randn(10), index = [list('aaabbbccdd'),[1,2,3,1,2,3,1,2,2,3]])
y = DataFrame(np.arange(12).reshape(4,3),index = [list('aabb'),list('1212')],columns = [['ohio','ohio','colorado'],['green','red','green']])
#查看索引
s.index
#查看标签c的内容
s['c']
#查看标签b和d的内容
s[['b','d']]
#查看标签b到d的内容
s['b':'d']
#把s转化为DataFrame
s.unstack()
#再转回来
s.unstack.stack()
#查看索引
y.index
#给各层起名字
y.index.names = ['key1','key2']
y.columns.names = ['state','color']
#选取分组
y['ohio']
#互换索引级别
y.swaplevel('key1','key2')
#或者
y.swaplevel(0,1)
#根据单个级别中的值进行排序,可能会产生future warning
y.sortlevel(1)
#改进
y.sort_index(level = 1)


