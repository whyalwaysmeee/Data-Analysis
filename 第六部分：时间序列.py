from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from dateutil.parser import parse
from pandas import Series, DataFrame
from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
import pytz
import matplotlib.pyplot as plt

#获取当前时间及年月日
now = datetime.now()
now.year
now.month
now.day
#计算时间差及日差、秒差
delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)
delta.days
delta.seconds
#将时间往后移12天
start = datetime(2011,1,7)
start + timedelta(12)

#字符串和datetime的相互转换
stamp = datetime(2011,1,7)
datestrs = ['7/6/2011','8/6/2011']
#datetime转字符串
str(stamp)
#指定格式
stamp.strftime('%Y-%m-%d')
#字符串转datetime
#方法1：
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
#方法2：
parse('2011-01-03')
parse('Jan 31,1997 10:45 PM')
#国际通用的格式中会将日放在前面，而该转换的结果为2011-06-12，这显然不对，加入dayfirst参数即可解决该问题
parse('6/12/2011',dayfirst=True)
#方法3：
pd.to_datetime(datestrs)



#时间序列基础
dates = [datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7),datetime(2011,1,8),datetime(2011,1,10),datetime(2011,1,12)]
ts = Series(np.random.randn(6),index = dates)
#不同索引的时间序列之间的算术运算会自动按日期对齐，于是结果中2、4、6行为空值
ts + ts[::2]


#索引、选区、子集构造
#pd.date_range用于生成指定长度的DatetimeIndex，后面会做详细介绍
longer_ts = Series(np.random.randn(1000),index = pd.date_range('1/1/2000',periods=1000))
#选取一个日期对应的数据，stamp是一个Timestamp对象，可随时自动转换为datetime对象
stamp = ts.index[2]
ts[stamp]
#更为简便的方法：直接传入一个可被解释为日期的字符串
ts['1/10/2011']
ts['20110110']
#对于较长时间序列的切片，传入“年”或“年月”
longer_ts['2001']
longer_ts['2001-05']
#通过日期切片只对规则Series有效，这里可以传入字符串日期、datetime或Timestamp
ts['1/6/2011':'1/11/2011']
ts[datetime(2011,1,5):]
ts[ts.index[0]]
#通过truncate函数，参数before表示返回该日期之后的所有日期及数据，after表示之前
ts.truncate(before='1/4/2011')
#对DataFrame进行上述操作
dates1 = pd.date_range('1/1/2000',periods = 100, freq = 'W-WED')
long_df = DataFrame(np.random.randn(100,4),index = dates1,columns = ['Colorado','Texas','New York','Ohio'])
long_df.ix['5/2001']


#日期的范围、频率及移动
#生成日期范围
index = pd.date_range('4/1/2012','6/1/2012')
#默认情况下date_range按天计算时间点（参数periods表示天数），如果只传入起始或结束日期，则还需传入一个表示时长的数字
pd.date_range(start = '4/1/2012', periods = 20)
pd.date_range(end = '6/1/2012', periods = 20)
#生成一个由每个月最后一个工作日组成的日期索引
index_bm = pd.date_range('1/1/2000','12/1/2000',freq = 'BM')

#频率和偏移量
#频率为1小时、4小时和40分钟
hour = Hour()
four_hour = Hour(4)
forty_minute = Minute(40)
#可进行加法运算
Hour(2) + Minute(30)
#一般来说无需创建这样的对象，用参数freq导入就好
pd.date_range('1/1/2000','1/3/2000 23:59',freq = '4h')
pd.date_range('1/1/2000','1/3/2000 23:59',freq = '4h30min')
#从起始时间起经过10个freq
pd.date_range('1/1/2000',periods = 10,freq = '4h30min')
#WOM日期
#获得期间内每个月的第三个周五
pd.date_range('1/1/2001','1/1/2002',freq = 'WOM-3FRI')
#移动数据
ts1 = Series(np.random.randn(4),index = pd.date_range('1/1/2000',periods = 4, freq = 'M'))
#沿时间轴将数据后移2位
ts1.shift(2)
#前移2位
ts1.shift(-2)
#将时间轴后移2月
ts1.shift(2,freq = 'M')
#通过偏移量对日期进行移动
now = datetime(2011,11,17)
now + 3 * Day()
#该操作会将原日期先向前滚动到符合频率规则的一个日期2011-10-31，再滚动到2011-10-31的下一个日期2011-11-30
now + MonthEnd()
#传入参数2则滚动到下两个日期2011-12-31
now + MonthEnd(2)
#第二种移动日期的方法
offset = MonthEnd()
#将日期移动到该月的最后一日
offset.rollforward(now)
#将该日期移动到上个月的最后一日
offset.rollback(now)
#结合groupby使用滚动
ts2 = Series(np.random.randn(20),index = pd.date_range('1/15/2000',periods = 20, freq = '4d'))
#计算当月的平均值并赋给当月最后一个日期
ts2.groupby(offset.rollforward).mean()


#时区处理
#查看所有时区
pytz.common_timezones
#本地化和转换
#生成时间序列
rng = pd.date_range('3/9/2012 9:30',periods = 6,freq = 'D')
#以时间序列rng作为index构建Series
ts3 = Series(np.random.randn(len(rng)),index = rng)
#时区化、本地化
#将'3/9/2012 9:30'及后面的6个时间赋给UTC时区，UTC默认永远是0时区
ts3_utc = ts3.tz_localize('UTC')
#转化到别的时区
ts3_utc.tz_convert('US/Eastern')
#将'3/9/2012 9:30'及后面的6个时间赋给美国东部时区，但UTC仍为0时区
ts3_eastern = ts3.tz_localize('US/Eastern')
#验证UTC仍为0时区
ts3_eastern.tz_convert('UTC')

#操作时区意识型Timestamp对象
stamp = pd.Timestamp('2011-03-12 04:00')
#Timestamp对象也可以时区化
stamp_utc = stamp.tz_localize('utc')
#查看美国东部时间
stamp_utc.tz_convert('US/Eastern')
#创建Timestamp时还可以传入一个时区信息，此时仍是以UTC作为0时区
stamp_moscow = pd.Timestamp('2011-03-12 04:00',tz = 'Europe/Moscow')

#不同时区时间的运算
ts11 = ts3[:7].tz_localize('US/Eastern')
ts12 = ts11[2:].tz_convert('Europe/Moscow')
#由于两个时间序列时区不同，结果会是UTC
res = ts11 + ts12


#时期及其算术运算
#时期表示的是时间区间，表示2007一整年的时间（'A-DEC'表示以12月作为末尾）
p = pd.Period(2007, freq='A-DEC')
#位移
p + 5
p - 2
#差
pd.Period('2014',freq = 'A-DEC') - p
#创建规则的时期范围
pr = pd.period_range('1/1/2000','6/30/2000',freq = 'M')
#PeriodIndex类的构造函数可直接使用字符串
values = ['200103','200203','200303']
pd.PeriodIndex(values,freq = 'Q-DEC')
#可在Series中被用作索引
Series(np.random.randn(6),index = pr)
#时期的频率转换，将2007年转换成2007年的1月和12月（低频转高频）
p.asfreq('M',how = 'start')
p.asfreq('M',how = 'end')
#如果不以12月作为年尾
p1 = pd.Period('2007',freq = 'A-JUN')
#转换结果会是2006年的7月和2007年的6月
p1.asfreq('M',how = 'start')
p1.asfreq('M',how = 'end')
#高频转低频，此例中2007年8月属于2008年
p2 = pd.Period('2007-08','M')
p2.asfreq('A-JUN')
#PeriodIndex和TimeSeries的频率转换方式也是如此

#按季度计算的时期频率
#表示该财年以2012年的1月作为结尾，所以第四季度2012Q4是从2011年11月到2012年1月
p3 = pd.Period('2012Q4',freq = 'Q-JAN')
#获取该季度倒是第二个工作日下午4点的时间戳
p4pm = (p.asfreq('B','e')-1).asfreq('T','s') + 16*60
#转为Timestamp
p4pm.to_timestamp()
#生成季度性范围
pr1 = pd.period_range('2011Q3','2012Q4',freq = 'Q-JAN')
ts4 = Series(np.arange(len(pr1)),index = pr1)

#将Timestamp转为Period
pr2 = pd.date_range('1/1/2000',periods = 3, freq = 'M')
ts5 = Series(np.random.randn(3),index = pr2)
#将由时间戳作为索引的Series或DataFrame转换为以时期索引
ts5.to_period()
#转为时间戳
ts5.to_timestamp()
#频率由日变为月
pr3 = pd.date_range('1/29/2000',periods = 6,freq = 'D')
ts6 = Series(np.random.randn(6),index = pr3)
#新的频率默认由时间戳推断而来，所以会出现重复的时期
ts6.to_period('M')

#通过数组创建PeriodIndex
#这个数据集中年度和季度被存放在不同列中，现在要把这两列合并
data = pd.read_csv('macrodata.csv')
#将这两个数组以及一个频率传入PeriodIndex，就可以合成DataFrame的一个索引
index9 = pd.PeriodIndex(year = data.year,quarter = data.quarter,freq = 'Q-DEC')
data.index = index9


#重采样及频率转换
#重采样：将时间序列从一个频率转换到另一个频率
#降采样：将高频率数据聚合到低频率，升采样则反之
pr4 = pd.date_range('1/1/2000',periods = 100,freq = 'D')
ts7= Series(np.random.randn(len(pr4)),index = pr4)
ts7.resample('M',how = 'mean')
ts7.resample('M',how = 'mean', kind = 'period')
#降采样
pr5 = pd.date_range('1/1/2000',periods = 12, freq = 'T')
ts8 = Series(np.arange(12),index = pr5)
#通过求和的方式将这些数据聚合到“5分钟”块中，默认以右边界作为标记，即结果显示的是每个时间段的右边界
ts8.resample('5min',how = 'sum')
#将区间设置为左闭合
ts8.resample('5min',how = 'sum',closed = 'left')
#将左边界设置为标记
ts8.resample('5min',how = 'sum',closed = 'left',label = 'left')
#将索引向前位移1秒
ts8.resample('5min',how = 'sum',loffset = '-1s')
#OHLC重采样，用于金融领域，计算四个值（open，开盘；close，收盘；high，最大值；low，最小值）
ts8.resample('5min',how = 'ohlc')

#升采样和插值
f = DataFrame(np.random.randn(2,4),index = pd.date_range('1/1/2000',periods = 2,freq = 'W-WED'),columns = ['Colorado','Texas','New York','Ohio'])
#填充NA值
f_daily = f.resample('D',fill_method='ffill')
#只填充2行NA值
f_daily = f.resample('D',fill_method='ffill',limit = 2)

#通过时期进行重采样
f1 = DataFrame(np.random.randn(24,4),index = pd.date_range('1-2000','12-2001',freq = 'M'),columns = ['Colorado','Texas','New York','Ohio'])
#降采样频率为年，算平均值
annual_f1 = f1.resample('A-DEC',how = 'mean')
#升采样要决定新频率中各区间的哪端用于放置原来的值，默认参数是“end”
annual_f1.resample('Q-DEC',fill_method = 'ffill',convention = 'start')

#时间序列绘图
#先简单处理，提取需要的列、填充NA值
close_px_all = pd.read_csv('stock_px.csv',parse_dates=True,index_col=0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px = close_px.resample('B',fill_method='ffill')
#选取其中一列绘图
close_px['AAPL'].plot()
#对DataFrame调用plot时，所有时间序列会被绘制在一个图里，并自动生成一个图例，月份和年度都会被格式化在X轴上
close_px.ix['2009'].plot()
#苹果公司在2011年1月到3月的每日股价
close_px['AAPL'].ix['01-2011':'03-2011'].plot()
#季度型频率
close_px['AAPL'].resample('Q-DEC',fill_method= 'ffill')

#移动窗口函数


















