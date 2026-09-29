import math
import sys

import pandas

sys.path.append('/data/share_from_zhou/barra_tools')
from barra_rice import get_exposure_one_day

"""
def get_exposure_one_day(date, pm_weight, return_dataframe=False,contains=True,save_path=None):
	获取投资组合在[43个因子上的暴露值,43个因子上的收益，1个特质收益]
	指数：{'all': '中证全A'}
	因子：
		10个风格因子
		1个国家因子
		31个申万一级行业分类因子
		1个特质收益因子：['Alpha']
	参考资料：https://www.ricequant.com/doc/rqdata/python/risk-factors-mod
	:param date:日期
	:param pm_weight:{str:float},pm投资组合
	:param return_dataframe:是否以DataFrame返回
	:param contains:是否包含现金
	:param save_path:保存路径
	:return: 1.中证全A每只成分股：10风格因子暴露+1国家因子暴露+31行业因子暴露+10风格因子收益+1国家因子收益+31行业因子收益+alpha收益
			 2.中证全A指数(all)：10风格因子暴露+1国家因子暴露+31行业因子暴露+10风格因子收益+1国家因子收益+31行业因子收益+alpha收益
			 3.your_portfolio: 10风格因子暴露+1国家因子暴露+31行业因子暴露+10风格因子收益+1国家因子收益+31行业因子收益+alpha收益


    [  'size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price',
       'leverage', 'growth', 'earnings_yield', 'beta', 'residual_volatility',
       'comovement', 
       '银行', '计算机', '环保', '商贸零售', '电力设备', '建筑装饰', '建筑材料', '农林牧渔','电子', '交通运输', 
       '汽车', '纺织服饰', '医药生物', '房地产', '通信', '公用事业', '综合', '机械设备',  '石油石化', '有色金属', 
       '传媒', '家用电器', '基础化工', '非银金融', '社会服务', '轻工制造', '国防军工', '美容护理', '煤炭', '食品饮料', 
       '钢铁', 
       'size收益', 'non_linear_size收益', 'momentum收益', 'liquidity收益', 'book_to_price收益', 
       'leverage收益', 'growth收益', 'earnings_yield收益', 'beta收益', 'residual_volatility收益', 
       'comovement收益',
       '银行收益', '计算机收益', '环保收益', '商贸零售收益', '电力设备收益', '建筑装饰收益', '建筑材料收益',
       '农林牧渔收益', '电子收益', '交通运输收益', '汽车收益', '纺织服饰收益', '医药生物收益', '房地产收益', '通信收益',
       '公用事业收益', '综合收益', '机械设备收益', '石油石化收益', '有色金属收益', '传媒收益', '家用电器收益',
       '基础化工收益', '非银金融收益', '社会服务收益', '轻工制造收益', '国防军工收益', '美容护理收益', '煤炭收益',
       '食品饮料收益', '钢铁收益', 
       'alpha']
"""

# 11个风格因子
style_factor1 = ['size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price', 'leverage', 'growth',
                 'earnings_yield', 'beta', 'residual_volatility', 'comovement']
# 31个行业因子
industry_factor = ['银行', '计算机', '环保', '商贸零售', '电力设备', '建筑装饰', '建筑材料', '农林牧渔', '电子',
                   '交通运输', '汽车', '纺织服饰', '医药生物', '房地产', '通信', '公用事业', '综合', '机械设备',
                   '石油石化', '有色金属', '传媒', '家用电器', '基础化工', '非银金融', '社会服务', '轻工制造',
                   '国防军工', '美容护理', '煤炭', '食品饮料', '钢铁']

ALL_FACTOR_NAME = ['size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price', 'leverage', 'growth',
                   'earnings_yield', 'beta', 'residual_volatility', 'comovement', '银行', '计算机', '环保', '商贸零售',
                   '电力设备', '建筑装饰', '建筑材料', '农林牧渔', '电子',
                   '交通运输', '汽车', '纺织服饰', '医药生物', '房地产', '通信', '公用事业', '综合', '机械设备',
                   '石油石化', '有色金属', '传媒', '家用电器', '基础化工', '非银金融', '社会服务', '轻工制造',
                   '国防军工', '美容护理', '煤炭', '食品饮料', '钢铁']


def get_factor_index(fn, t='exp'):
    """
    在通用barra数据中, 一般是一个长度为85的列表,表示某一日的42个因子暴露+42个因子收益+alpha收益
    当前这个方法用于查询某个因子所对应的下标索引
    """
    global ALL_FACTOR_NAME
    fi = ALL_FACTOR_NAME.index(fn)
    if fi < 0:
        print(f"------尝试查找不存在的因子{fn}--------,是否传递参数错误?请检查")
        return -1
    if t == 'exp':
        return fi
    else:
        return fi + 42


def get_exp_and_value(balist):
    factor_exp = {}
    # 11个风格因子暴露值
    tv = balist[0:len(style_factor1)]
    for i in range(len(style_factor1)):
        factor_exp[style_factor1[i]] = tv[i]

    # 31个行业因子暴露值
    tv = balist[11:11 + len(industry_factor)]
    for i in range(len(industry_factor)):
        factor_exp[industry_factor[i]] = tv[i]
    # 10个风格因子收益
    factor_value = {}
    tv = balist[42:42 + len(style_factor1)]
    for i in range(len(style_factor1)):
        factor_value[style_factor1[i]] = tv[i]
    # 31个风格因子收益
    tv = balist[53:53 + len(industry_factor)]
    for i in range(len(industry_factor)):
        factor_value[industry_factor[i]] = tv[i]
    alpha = balist[-1]
    return factor_exp, factor_value, alpha


def re_weight_cash(weight):
    """
    将一个权重去掉现金并重新归一化
    """

    if (not weight.__contains__('CASH')) and (not weight.__contains__('CASH')):
        return weight
    stock_w = 0.
    nw = {}
    SC = 0
    for k, v in weight.items():
        if k == 'cash' or k == 'CASH':
            continue
        nw[k] = v
        stock_w += v
        SC += 1
    if SC == 0:
        return {'000001.SZ': 0}
    for k, v in nw.items():
        nw[k] = nw[k] / stock_w
    return nw


def get_barra(day, weight, keep_cash=True):
    """
    调用barra_rice工具对当天的持仓组合进行分析,返回当日中证全A指数以及本投资组合的barra数据
    :param day: 日期字符串(或int)-YYYYMMDD格式
    :param weight: 权重字典{'CASH':0.02,'000001.SZ':0.98}
    :param keep_cash 是否保留现金,默认为TRUE
    :return 全A指数的barra暴露值,全A指数的barra收益,全A-alpha,本PM投资组合的barra暴露值,本PM投资组合的barra收益,本alpha
    """

    if keep_cash:
        # 如果保留现金项, 则使用下面这几行
        barra = get_exposure_one_day(int(day), weight, return_dataframe=True, contains=True)
    else:
        # 如果不保留现金, 就要去掉现金项并重新归一化
        weight = re_weight_cash(weight)
        barra = get_exposure_one_day(int(day), weight, return_dataframe=True, contains=False)

    barra = pandas.DataFrame(barra)
    # pandas.core.frame.DataFrame
    # print(barra.columns)

    # 输出字段共85个, 分别是这42个因子的暴露值+42个因子收益+alpha收益
    qa_factor = barra.loc['all'].tolist()
    qa_exp, qa_value, qa_alpha = get_exp_and_value(qa_factor)

    pm_factor = barra.loc['your_portfolio'].tolist()
    pm_exp, pm_value, pm_alpha = get_exp_and_value(pm_factor)

    if math.isnan(qa_alpha):
        print("**有NAN***qa", day)
        qa_alpha = 0.

    if math.isnan(pm_alpha):
        print("**有NAN---pm**", day)
        pm_alpha = 0.

    return qa_exp, qa_value, qa_alpha, pm_exp, pm_value, pm_alpha
