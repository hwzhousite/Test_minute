import os
import pandas as pd
import json

FACTOR_ALL = ['size收益', 'non_linear_size收益', 'momentum收益',
       'liquidity收益', 'book_to_price收益', 'leverage收益', 'growth收益',
       'earnings_yield收益', 'beta收益', 'residual_volatility收益', 'comovement收益',
       '银行收益', '计算机收益', '环保收益', '商贸零售收益', '电力设备收益', '建筑装饰收益', '建筑材料收益',
       '农林牧渔收益', '电子收益', '交通运输收益', '汽车收益', '纺织服饰收益', '医药生物收益', '房地产收益', '通信收益',
       '公用事业收益', '综合收益', '机械设备收益', '石油石化收益', '有色金属收益', '传媒收益', '家用电器收益',
       '基础化工收益', '非银金融收益', '社会服务收益', '轻工制造收益', '国防军工收益', '美容护理收益', '煤炭收益',
       '食品饮料收益', '钢铁收益']
FN = len(FACTOR_ALL)   # 共42个因子(10风格+1国家+31行业)
FACTO_DATA_PATH = "/data/share_from_zhou/barra_tools/barra_file_rice"

"""
功能:  从周华生成的每日barra数据中提取因子收益数据,单独输出一个文件
FACTO_DATA_PATH 路径下,有以日期开头的parquet文件, 名称格式如: 20260430.parquet
这个文件中有全部股票的因子数据, 每只股票下又有84列数据, 其中前42列是当前股票的因子暴露.这里暂时不需要使用
需要提取的是后42列数据,(所有股票的这后42列数据都是相同的,冗余数据), 这42列是当天各因子收益值, 我们统一提取第一只股票的就可以了.
提取完成后,生成以日期为主键的字典 {key=date, value=[42]} 并输出文件: /data/indexData/barra_factor_value.json

"""


def get_fv(x):
    x = list(x)
    x = x[42:42+FN]
    return x

def make_barra_factor_value(path=FACTO_DATA_PATH):
    """
    指定所有barra原始数据路径, 从目录中遍历所有yyyymmdd.parquet文件,读取该日期下的所有因子收益数据
    生成一个dict保存到文件
    """

    if not os.path.exists(path):
        raise ValueError(f"数据路径不存在{path}")

    dic={}
    # 遍历指定路径下的所有parquet文件
    FCN = 0
    for file in os.listdir(path):
        if not file.endswith(".parquet"):
            continue
        # 获取文件名,不含扩展名
        fbasename = os.path.splitext(file)[0]
        if len(fbasename)!=8:
            continue
        # 如果长度是8位,那一定是日期了
        fv = read_fv(os.path.join(path,file))
        dic[fbasename]=fv
        FCN +=1
        if FCN % 100 ==0:
            print(f"#{FCN}-{fbasename}---{len(fv)}")
    print(f"读取完成,文件总数:{FCN}")

    if len(dic)==0:
        raise ValueError(f"读取数据失败@{path}")

    # 将dic排序
    dic = dict(sorted(dic.items(),key=lambda x:x[0]))

    # 保存到文件
    fn = "/data/indexData/barra_factor_value.json"
    with open(fn,"w") as f:
        json.dump(dic,f,indent=4)
    dayList = list(dic.keys())
    print(f"生成每日因子收益数据完成输出到文件:{fn}")
    print(f"因子数据时段{dayList[0]}~{dayList[-1]},长度{len(dayList)}")


def read_fv(file):
    """
    读取一个parquet文件, 返回一个list
    """
    df = pd.read_parquet(file)
    return get_fv(df.values[0])


if __name__ == '__main__':
    make_barra_factor_value()