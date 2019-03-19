import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pandas.io.data as web
import sys
import csv
import json
import urllib
from lxml.html import parse
from pandas.io.parsers import TextParser

#读取excel文件
#表格用excel
#记事本用table

data1 = pd.read_excel("D:/cs/P4/a.csv")
data2 = pd.read_table("D:/cs/P4/aaa.txt")

#读取没有标题行的文件，自行加标题
#默认
pd.read_excel("D:/cs/P4/1.csv", header = None)
#自定义
pd.read_excel("D:/cs/P4/test.xlsx", names = ['a','b','c','d','e'])

#由于该文件各字段由数量不等的空白格分隔，所以使用正则表达式将其调整为更加整洁的形式
pd.read_table("D:/cs/P4/aaa.txt",sep = '\s+')
#跳过奇怪的或是非数据行
pd.read_table("D:/cs/P4/aaa.txt",sep = '\s+',skiprows = [1,2])
#只需要读取其中的3行
pd.read.table("D:/cs/P4/aaa.txt",nrows = 3)

#以下操作已经将结果写出，无需print
#添加分隔符“|”
data1.to_csv(sys.stdout,sep = '|')
#把缺失值表示为别的标记值"NULL"
data1.to_csv(sys.stdout,na_rep = 'NULL')
#禁用行、列标签
data1.to_csv(sys.stdout,index = False, header = False)
#只读出指定序列
data1.to_csv(sys.stdout,cols = ['a','b'])

#JSON数据
obj = """
{"name":"wes",
"place_lived":["US","Spain","Germany"],
"pet":null,
"siblings":[{"name":"Scott","age":25,"pet":"Zuko"},
{"name":"Katie","age":33,"pet":"Cisco"}]
}
"""
#将JSON数据转化为PYTHON数据
result = json.loads(obj)
#将PYTHON数据转化为JSON数据
ajson = json.loads(result)
#将JSON数据转化为DataFrame格式
siblings = DataFrame(result['siblings'],columns = ['name','age','pet'])

#XML和HTML：web信息收集
#对yahoo!finance当天的股票信息进行爬取
#复制粘贴即可获得所需网址
url = "https://finance.yahoo.com/quote/AAPL/options?ltr=1"
#使用parse解析得到的数据流
parsed = parse(urllib.request.urlopen(url))
#获取网页源代码
doc = parsed.getroot()
#获取网页中所有的表格形式文件（发现一共只有两个，calls和puts），此时是lxml文件
tables = doc.findall('.//table')
calls = tables[0]
puts = tables[1]
#获取标题行th和数据行td每个单元格内的文本
def unpack(row,kind = 'td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]
#将网页中的表格转化为一个DataFrame
def parse_options_data(table):
    #获取所有行
    rows = calls.findall('.//tr')
    #获取标题行
    header = unpack(rows[0], kind = 'th')
    #获取数据行
    data = [unpack(r) for r in rows[1:]]
    return TextParser(data, names = header).get_chunk()
#最后用上述函数对两个lxml表格文件进行解析
call_data = parse_options_data(calls)
put_data = parse_options_data(puts)

#读取Excel文件,xls是ExcelFile格式，table是DataFrame格式
xls = pd.ExcelFile('test.xlsx')
table = xls.parse('Sheet1')

#处理XML数据、二进制数据和HDF5格式数据，参照课本P177——P180
#从数据库中载入数据，参照课本P182——P185

