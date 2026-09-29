import argparse
import json
import os
from time import time
import utils as UTILS
import common_utils as CM
import product_eval as EVTOOL

"""
功能: 给定一个路径,该路径下有所有交易日的调仓记录(json格式)
这里读取调仓记录,生成每周相对收益数据,并保存到文件中
"""

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    # 下面这个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')

    args = parser.parse_args()

    assert args.path != '' and os.path.exists(args.path), f"必须指定合法存在的权重文件路径:{args.path}"

    if args.DS == 'day304':
        args.g_csv_path = r"/data/raw_generated_data/wind/304f_pad_csv"
        args.scaler_file = r"/data/yy_data/wind/304f_20100104/scaler_info.json"
    elif args.DS == 'day104':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/104f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/104f_250618/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    elif args.DS == 'day106':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/106f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/106f_250619/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    else:  # day111
        raise ValueError("not support DS")
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path)
    return args

def append_model(path, model_value):
    dbfile = 'dbinfo.json'
    allmodels={}
    if os.path.exists(dbfile):
        with open(dbfile, 'r', encoding='utf-8') as file:
            allmodels = json.load(file)
    allmodels[path]=model_value

    with open(dbfile, 'w', encoding="utf-8") as file:
        json.dump(allmodels, file)
    print(f"入库完成,更新数据文件<{dbfile}>")

if __name__ == '__main__':
    args = get_args()

    # 读取权重数据
    ts = time()
    weights_A = UTILS.read_weigh_from_path(args.path)
    DayCount = len(weights_A)
    dayList=list(weights_A.keys())
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    # 读取所有股票信息
    print(f"{CM.timestr()}读取所有交易日股票信息...")
    ts = time()
    # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    all_stock_info = UTILS.read_all_stock_info(weights_A)
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = EVTOOL.make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # 计算每日收益,这里会计算每日绝对收益(扣除成本)!!!
    # 下面的方法返回:
    # 1.每天模型的相对前一天的价值(税后)--税后是指扣除了交易成本  {k=交易日期,value=value}
    # 2.每天盈利股票
    # 3.每天带来亏损的股票
    model_value_A, _, _ = EVTOOL.cal_trade_series_value(args.tradeCode, weights_A, priceInfo)  # 计算A模型收益
    append_model(args.path,model_value_A)


