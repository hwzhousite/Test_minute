"""
对两个不同的模型的输出进行对比
"""
import argparse
import json
import os
from time import time
import numpy as np
import utils as UTILS
import common_utils as CM
import product_eval as EVTOOL
import utils_date as DT

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    parser.add_argument('--out_path', type=str, default='', help='如果有输出文件时,保存到这个路径下')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--patha', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--pathb', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--matcha', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--matchb', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='eval', help='标识,输出某些文档时以此命名')
    parser.add_argument('--indexName', type=str, default='全A', help='业绩比较基准,请确保/data/indexData/下有同名的json文件')

    args = parser.parse_args()

    assert args.patha != '' and os.path.exists(args.patha), f"必须指定合法存在的权重文件路径:{args.path}"
    assert args.pathb != '' and os.path.exists(args.pathb), f"必须指定合法存在的权重文件路径:{args.path}"
    if args.out_path != '' and (not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    if args.DS == 'day1062':
        args.g_csv_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/106f_pad_csv_calendar"  # 每日增量更新
        args.scaler_file = f"{CM.G_ROOT_PATH}/yy_data/tushare_data/ts_260201_106f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 如果使用1062数据集回测,该数据集中的天数据是严格按照交易日历来生成的,没有padd节假日数据了
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
        args.keep_fields = [i for i in range(105)]  # 全部字段序列,一共105个
        # 检查虚拟内存数据文件是否存在, 如果存在,则指向这个内存数据文件目录,加快读取速度
        # 需要运行当前目录下的memdata.sh来加载和初始化
        # 服务器未重启的话,生命期内只需要运行一次, 当然,如果真实的数据更新, 这个内存文件并不会自动同步, 需要手动更新
        # virtual_mem_data_path = '/memdata_virtual/106f_pad_data/'
        # # 这里的106f_pad_data是在memdata.sh中定义的
        # if os.path.exists(virtual_mem_data_path):
        #     args.g_csv_path = virtual_mem_data_path
    elif args.DS == 'day134':
        args.g_csv_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/ts_260413_134f/tgt_pad_data"  # 每日增量更新
        # args.g_csv_path = f"{CM.G_ROOT_PATH}/yy_data/tushare_data/ts_260413_134f/pad_data"  # 数据集对应的字段字典表文件路径
        args.scaler_file = f"{CM.G_ROOT_PATH}/yy_data/tushare_data/ts_260413_134f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 检查虚拟内存数据文件是否存在, 如果存在,则指向这个内存数据文件目录,加快读取速度
        # 需要运行当前目录下的memdata.sh来加载和初始化
        # 服务器未重启的话,生命期内只需要运行一次, 当然,如果真实的数据更新, 这个内存文件并不会自动同步, 需要手动更新
        # virtual_mem_data_path = '/memdata_virtual/134f_pad_data/'
        # # 这里的134f_pad_data是在memdata.sh中定义的
        # if os.path.exists(virtual_mem_data_path):
        #     args.g_csv_path = virtual_mem_data_path

        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
        args.keep_fields = [i for i in range(133)]  # 全部字段序列,一共133个
    elif args.DS == 'wd311':
        args.g_csv_path = f"{CM.G_ROOT_PATH}/yy_data/qd/train/wd_311f/0603/pads"  # 每日增量更新
        args.scaler_file = f"{CM.G_ROOT_PATH}/yy_data/qd/train/wd_311f/0603/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8-sig') as f:
            DT.g_calendar = json.load(f)
        args.keep_fields = [i for i in range(310)]  # 全部字段序列,一共310个
    else:
        raise ValueError(f"数据集参数错误{args.DS}")
    UTILS.G_ALL_STOCK_DATA.reset(args.DS, args.g_csv_path, args.indexName)

    args.indexFile = os.path.join("/data/indexData/", f"{args.indexName}.json")
    print(f"业绩比较基准为<{args.indexName}>,数据文件<{args.indexFile}>")
    assert os.path.exists(args.indexFile),"指数数据文件不存在"

    return args


def make_merge_day_weight(w1, w2):
    w = {}
    for k, v in w1.items():
        w[k] = v
    for k, v in w2.items():
        if w.__contains__(k):
            continue
        w[k] = v
    return w


def make_merge_weight(w1, w2):
    """
    将两个权重图按天合并,只是为了构造一个类似的结构,但包含同一天在不同投资组合中的所有股票
    构造的这个临时权重用于读取数据依据
    外部已经确保了两个权重图的日期列表完全一致
    """
    mw = {}
    for k1d, k1w in w1.items():
        mw[k1d] = make_merge_day_weight(k1w, w2[k1d])
    return mw


def cal_same_day_weight(wa,wb):
    """
    统计两个单权重之间的重合度(需要将现金项去除后归一化)
    """
    if wa['CASH']>=0.999999 or wb['CASH']>=0.999999:
        return 0
    stockA = list(wa.keys())
    totalA = sum(wa.values()) - wa['CASH']
    stockB = list(wb.keys())
    totalB = sum(wb.values()) - wb['CASH']
    # 求出股票交集
    sameStock = list(set(stockA) & set(stockB))
    sameW = 0
    for s in sameStock:
        if s == 'CASH':
            continue
        # 将股票权重都分别反归一化(即假定全仓股票,现金为0)
        va = wa[s] / totalA
        vb = wb[s] / totalB
        sameW +=min(va,vb) # 取二者较小值为重合度
    return sameW


def cal_same_weight(weightsA,weightsB):
    """
    逐日计算两个权重图之间的股票重合度(去除现金后的重新归一值)
    """
    dayList = list(weightsA.keys())
    DL = len(dayList)
    sameW = {}
    for i in range(DL):
        day = dayList[i]
        vw = cal_same_day_weight(weightsA[day],weightsB[day])
        sameW[day]=round(vw*100.,4)
    return sameW


# def comp_two_models(weightsA,weightsB,targetPrice):
#     """
#     给定两个调仓序列(已确保时间同步),计算这两个调仓序列的相关性
#     """
#
#     relation={} # 返回一个相关性字典
#
#     # step1: 准备数据
#     weights_S = EVTOOL.make_combin_weight(weightsA, weightsB, 0.5)
#     # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
#     all_stock_info = UTILS.read_all_stock_info(weights_S)
#     # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
#     priceInfo = EVTOOL.make_price_info_from_totalinfo(targetPrice, all_stock_info)



if __name__ == '__main__':
    args = get_args()

    # 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weightsA = UTILS.read_weigh_from_path(args.patha, args.matcha)
    weightsB = UTILS.read_weigh_from_path(args.pathb, args.matchb)
    DayCount = len(weightsA)
    assert DayCount == len(weightsB) and DayCount > 5, f"读取权重数据天数异常{DayCount}-{len(weightsB)}"
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    dayList = list(weightsA.keys())
    dayListB = list(weightsB.keys())
    # 两个日期列表必须完全一致
    checkDS = set(dayList) & set(dayListB)
    if len(checkDS) != DayCount:
        raise ValueError(f"两个投资组合的交易日期列表不一致,总长{DayCount}-重合数{len(checkDS)}")
    print(f"交易序列首日{dayList[0]},末日{dayList[-1]}")

    # 读取指数数据
    indexBase = CM.load_index(args.indexFile)

    # 读取行业/板块字典
    industory_info = CM.load_index("industry_info.json")
    industory_info = EVTOOL.reverse_dict_valueIsInt(industory_info)
    market_info = CM.load_index("market_info.json")
    market_info = EVTOOL.reverse_dict_valueIsInt(market_info)

    # 读取所有股票信息
    print(f"{CM.timestr()}读取所有交易日股票信息...")
    ts = time()
    tmpW2D = make_merge_weight(weightsA, weightsB)
    all_stock_info = UTILS.read_all_stock_info(tmpW2D)
    priceInfo = EVTOOL.make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # 计算每日收益,这里会计算每日绝对收益(扣除成本)!!!注意:去掉整个日期的两端的2天,因为这两天没有前导/后续,因而无法计算相对价值
    # 下面的方法返回:
    # 1.每天模型的相对前一天的价值(税后)--税后是指扣除了交易成本
    # 2.每天盈利股票
    # 3.每天带来亏损的股票
    model_value_A, win_dict_A, lost_dict_A = EVTOOL.cal_trade_series_value(args.tradeCost, weightsA, priceInfo)
    model_value_B, win_dict_B, lost_dict_B = EVTOOL.cal_trade_series_value(args.tradeCost, weightsB, priceInfo)
    base_value = EVTOOL.cal_seris_value(indexBase, dayList)
    # 以上这些数据字典的key都是dayList(去除两端日期)
    dayList.pop(-1)  # 日期列表去除两端的日期
    dayList.pop(0)

    ##################################################
    # 分析两个投资策略的异同
    ##################################################

    # 指标1: 绝对收益曲线的相似度(相关性)
    # 计算皮尔逊相关系数
    # 生成每日/周/月涨跌幅序列
    period = 'week'
    ms_A,_ = EVTOOL.make_seris_value(model_value_A, vtype='change', period=period)
    ms_B,_ = EVTOOL.make_seris_value(model_value_B, vtype='change', period=period)
    ms_base,_ = EVTOOL.make_seris_value(base_value, vtype='change', period=period)
    correlation = np.corrcoef(ms_A, ms_B)[0, 1]
    print(f"***绝对收益的相关系数/{period}***")
    print(f"对比对象,区间起点,区间终点,相关系数")
    print(f"A&B,{dayList[0]},{dayList[-1]},{correlation:.4f}")
    correlation = np.corrcoef(ms_A, ms_base)[0, 1]
    print(f"{args.patha}&{args.indexName},{dayList[0]},{dayList[-1]},{correlation:.4f}")
    correlation = np.corrcoef(ms_B, ms_base)[0, 1]
    print(f"{args.pathb}&{args.indexName},{dayList[0]},{dayList[-1]},{correlation:.4f}")
    print()

    # 指标2: 投资标的的相似度分析
    # 计算平均股票重合度(按权重),每日权重(除现金后)归一化到1
    sameWeight = cal_same_weight(weightsA,weightsB)
    # 生成图片
    titles = [f'投资标的分析-股票重合度-day', '日期', '重合度']
    data_dict = {"投资股票的权重重合度": list(sameWeight.values()),}
    EVTOOL.plot_image(args, titles, list(sameWeight.keys()), data_dict)
    print(f"***投资股票的权重重合度/day***")
    print(f"模型A,模型B,平均权重重合度")
    print(f"{args.patha},{args.pathb},{sum(sameWeight.values())/len(sameWeight):.2f}%")
    print()

