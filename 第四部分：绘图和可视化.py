import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
import pandas as pd
from datetime import datetime
from pandas import Series, DataFrame
from mpl_toolkits.basemap import Basemap

#绘制一张简单的点图
plt.plot(np.arange(10))
#显示
plt.show()
#绘制一张空白图
fig = plt.figure()
#在fig上绘制子图
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
#默认在最后一个使用过的subplot上绘制一个点图，'k--'代表黑色虚线图
plt.plot(randn(50).cumsum(),'k--')
#ax1，ax2是实例，直接调用他们的方法即可绘图
ax1.hist(randn(100),bins=20,color='k',alpha = 0.3)
ax2.scatter(np.arange(30),np.arange(30) + 3 * randn(30))
#一个更简便的创建subplot的方法
axes = plt.subplots(2,2)
#调整各种间距
for i in range(2):
    for j in range(2):
        axes[i,j].hist(randn(500),bins=50,color='k',alpha=0.5)
plt.subplots_adjust( left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

#设置标题、轴标签、刻度以及刻度标签,x轴默认刻度为0,200,400...1000
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum())
#重设x轴刻度
ticks = ax.set_xticks([0,250,500,750,1000])
#重命名x轴刻度
lables = ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small')
#设置标题
ax.set_title('ctgod')
#设置x轴标签
ax.set_xlabel('Stages')

#图例
ax.plot(randn(1000).cumsum(),'k',label='one')
ax.plot(randn(1000).cumsum(),'g--',label='two')
ax.plot(randn(1000).cumsum(),'r.',label='three')
ax.legend(loc='best')

#注解
data = pd.read_csv('spx.csv',index_col=0,parse_dates=True)
#在指定位置绘制文本nihao!
ax.text(1,1,'nihao!',family='monospace',fontsize=10)
#绘图
spx = data['SPX']
spx.plot(ax=ax,style='k-')
#要加入的注解，参数为注解位置的横坐标和内容
crisis_data = [(datetime(2007,10,11),'Peak of bull market'),(datetime(2008,3,12),'Bear Sterns Fails'),(datetime(2008,9,15),'Lehman Bankruptcy')]
#添加注解
for date,label in crisis_data:
    ax.annotate(label,xy=(date,spx.asof(date)+50),xytext = (date,spx.asof(date)+200),arrowprops = dict(facecolor = 'black'),
                horizontalalignment = 'left',verticalalignment = 'top')
#在x轴上放大到某一时间段
ax.set_xlim(['1/1/2007','1/1/2011'])
#在y轴上放大
ax.set_ylim([600,1800])
#添加标题
ax.set_title('Important dates in 2008-2009 financial crisis')
#图片保存
plt.savefig('fig.png',dpi = 400,bbox_inches = 'tight')
plt.savefig('fig.svg',dpi = 400,bbox_inches = 'tight')
plt.savefig('fig.pdf',dpi = 400,bbox_inches = 'tight')

#pandas中的绘图函数
#线图
#Series数据绘制成单条线图
s = Series(np.random.randn(10).cumsum(),index = np.arange(0,100,10))
s.plot()
#DataFrame数据绘制成多条线图
df = DataFrame(np.random.randn(10,4).cumsum(0),columns = ['A','B','C','D'],index = np.arange(0,100,10))
df.plot()

#柱状图
data = Series(np.random.rand(16),index = list('abcdefghijklmnop'))
df = DataFrame(np.random.rand(5,4),index = ['one','two','three','four','five'],columns = pd.Index(['A','B','C','D'],name = 'Genus'))
tips = pd.read_csv('tips.csv')
#在线型图的代码上加上kind参数即可,bar为垂直柱状图，barh为水平柱状图
fig, axes = plt.subplots(2,1)
data.plot(kind = 'bar',ax = axes[0],color = 'k',alpha = 0.7)
data.plot(kind = 'barh',ax = axes[1],color = 'k',alpha = 0.7)
#对DataFrame数据，会将每一行分为一组，stacked=True可以生成堆积柱状图
df.plot(kind = 'bar',stacked = True,alpha = 0.5)
#创建交叉表
party_counts = pd.crosstab(tips.day, tips.size)
#截取size为2-5的部分
party_counts = party_counts.ix[:,2:5]
#规格化，使各行和为1
party_pcts = party_counts.div(party_counts.sum(1).astype(float),axis = 0)
#绘图
party_pcts.plot(kind = 'bar',stacked = True)

#直方图和密度图
tips = pd.read_csv('tips.csv')
comp1 = np.random.normal(0,1,size=200)
comp2 = np.random.normal(10,2,size=200)
values = Series(np.concatenate([comp1,comp2]))
#直方图
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips['tip_pct'].hist(bins= 50)
#密度图
tips['tip_pct'].plot(kind = 'kde')
#由两个不同的标准正态分布组成的双峰分布
values.hist(bins=100,alpha = 0.3,color = 'k',normed = True)
values.plot(kind = 'kde',style='k--')

#散点图
macro = pd.read_csv('macrodata.csv')
data = macro[['cpi','m1','tbilrate','unemp']]
#计算对数差
trans_data = np.log(data).diff().dropna()
#绘图
plt.scatter(trans_data['m1'],trans_data['unemp'])
plt.title('Changes in log %s vs. log %s'%('mi','unemp'))
#从DataFrame创建散点图矩阵
pd.scatter_matrix(trans_data,diagonal='kde',color='k',alpha=0.3)


#示例项目：2010海地地震危机数据
#直接引入csv数据
data = pd.read_csv('Haiti.csv')
#清除错误位置信息并移除缺失分类信息
data = data[(data.LATITUDE > 18) & (data.LATITUDE < 20) & (data.LONGITUDE > -75) & (data.LONGITUDE < -70) & data.CATEGORY.notnull()]
#下面两个函数用于获取所有分类
def to_cat_list(catstr):
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x ]
def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))
all_cats = get_all_categories(data.CATEGORY)
#将各个分类信息拆分为编码和英语名称
def get_english(cat):
    codes,names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return codes,names.strip()
#得到一个编码和英语名称相互对应的字典
english_mapping = dict(get_english(x) for x in all_cats)
#下面要根据分类选取记录
#添加指标列
#先抽取出唯一的分类编码
def get_code(seq):
    return [x.split('.')[0] for x in seq if x]
all_codes = get_code(all_cats)
#索引化
code_index = pd.Index(np.unique(all_codes))
#构造一个新的DataFrame
dummy_frame = DataFrame(np.zeros((len(data),len(code_index))),index= data.index,columns = code_index)

#绘制海地地图(Basemap库无法导入)
for row,cat in zip(data.index,data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.ix[row,codes] = 1
data = data.join(dummy_frame.add_prefix('category_'))
def basic_haiti_map(ax=None,lllat = 17.25, urlat = 20.25, lllon = -75, urlon = -71):
    m = Basemap(ax = ax, projection = 'stere',lon_0 = (urlon + lllon) / 2,lat_0 = (urlat + lllat) /2 , llcrnrlat = lllat, urcrnrlat = urlat, llcrnrlon = lllon,
                urcrnrlon = urlon,resolution = 'f')
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m
fig,axes = plt.subplots(nrows = 2,ncols = 2,figsize = (12,10))
fig.subplots_adjust(hspace = 0.05,wspace = 0.05)
to_plot = ['2a','1','3c','7a']
lllat = 17.25;urlat = 20.25;lllon = -75;urlon = -71
for code,ax in zip(to_plot,axes.flat):
    m = basic_haiti_map(ax,lllat=lllat,urlat = urlat, lllon = lllon,urlon = urlon)
    cat_data = data[data['category_%s' % code] == 1]
    x,y = m(cat_data.LONGITUDE,cat_data.LATITUDE)
    m.plot(x,y,'k.',alpha = 0.5)
    ax.set_title('%s:%s'%(code,english_mapping[code]))









