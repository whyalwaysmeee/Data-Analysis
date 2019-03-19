import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#合并数据集
df1 = DataFrame({'key':['b','b','a','c','a','a','b'], 'data1':range(7)})
df2 = DataFrame({'key':['a','b','d'], 'data2':range(3)})
df3 = DataFrame({'lkey':['b','b','a','c','a','a','b'], 'data1':range(7)})
df4 = DataFrame({'rkey':['a','b','d'], 'data2':range(3)})
df5 = DataFrame({'key':['a','b','a','b','d'],'data2':range(5)})
df6 = DataFrame({'key':['b','b','a','c','a','b'], 'data1':range(6)})
#数据库风格的DataFrame合并
#多对一的合并。df1中有多个被标记为a和b的行，df2中每个key仅对应一行
#直接合并，没有指定哪个列连接的情况下，将会用重叠的列名当做键（默认交集）
pd.merge(df1,df2)
#制定列名
pd.merge(df1,df2, on = 'key')
#若两对象的列名不同，可分别指定
pd.merge(df3,df4, left_on = 'lkey',right_on = 'rkey')
#若要将列名的并集合并
pd.merge(df1,df2,how = 'outer')


#多对多的合并
left = DataFrame({'key1':['foo','foo','bar'],
                  'key2':['one','two','one'],
                  'lval':[1,2,3]})
right = DataFrame({'key1':['foo','foo','bar','bar'],
                  'key2':['one','one','one','two'],
                  'rval':[4,5,6,7]})
#outer表示将搜有的列名key都要包含进来，在这里最后生成的结果应该是一个12行的DataFrame
#理由：df5中2个b，df6中3个b，产生6行数据，df5、df6中各2个a，产生4行数据，df5中d、df6中c各产生1行数据
pd.merge(df5,df6, on = 'key', how = 'outer')
#根据多个键进行合并，传入一个由列名组成的列表
pd.merge(left, right, on = ['key1','key2'], how = 'outer')
#根据单个键合并，会生成key2_x和key2_y两个新列名，分别代表两个对象中的key2，因为两个对象中都有key2
pd.merge(left,right, on = 'key1')
#对新生成的列名进行命名
pd.merge(left,right, on = 'key1', suffixes = ('_left','_right'))


#索引的合并
left1 = DataFrame({'key':['a','b','a','a','b','c'],
                   'value' : range(6)})
right1 = DataFrame({'group_val':[3.5,7]},index = ['a','b'])
#right_index表示右侧（第二个）对象的索引被用作连接键，默认取交集
pd.merge(left1, right1, left_on = 'key',right_index = True)
#若要取并集
pd.merge(left1, right1, left_on = 'key',right_index = True, how = 'outer')
#join方法
left1.join(right1, how = 'outer')


#连接
#该种操作对DataFrame对象同样适用
s1 = Series([0,1],index = ['a','b'])
s2 = Series([2,3,4],index = ['c','d','e'])
s3 = Series([5,6],index = ['f','g'])
#使用concat，结果仍是一个Series（默认并集）
pd.concat([s1,s2])
#两个以上对象时，传入参数axis = 1会生成DataFrame（若不传入sort参数会出现警告）,结果为每个Series的数据占据一列，其余为空值
pd.concat([s1,s2,s3],axis = 1,sort = True)
#若要得到交集
pd.concat([s1,s3],axis = 1,join = 'inner')
#指定索引
pd.concat([s1,s3],axis = 1,join_axes = [['a','f']])
#层次化索引
pd.concat([s1,s2,s3],keys = ['one','two','three'])
#让keys成为列头
pd.concat([s1,s2,s3],keys = ['one','two','three'],axis = 1)

#合并重叠数据
a = Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index = ['f','e','d','c','b','a'])
b = Series(np.arange(len(a)),dtype=np.float64,index = ['f','e','d','c','b','a'])
#b会将a中的空值填充成自己的值，对DataFrame也同样适用
b.combine_first(a)

#重塑和轴向旋转
data = DataFrame(np.arange(6).reshape((2,3)),
                 index = pd.Index(['Ohio','Colorado'],name = 'state'),
                 columns = pd.Index(['one','two','three'],name = 'number'))
#重塑层次化索引
#将列旋转为行，结果是一个Series
data.stack()
#将行旋转为列，结果是一个DataFrame。默认情况是旋转最内层，传入分层级别的编号或名称可对其他级别进行操作
result = data.stack()
result.unstack()
result.unstack(0)
result.unstack('state')
#若不是有些值在索引中无法找到，那么会引入缺失值
data2 = pd.concat([s1,s2],keys = ['one','two'])
data2.unstack()
#stack会默认滤除缺失值
data2.unstack().stack()


#将长格式旋转为宽格式
data0 = pd.read_excel("D:/cs/P4/time.xlsx")
#不同item值分别形成一列，并用作列名，时间值作为索引
data0.pivot('data','item','value')







