# 进行批量测试
import argparse
import json
import os
import time
import numpy as np
import torch
import utils_date as DT
import utils as UTILS
import common_utils as CM
from data_utils import DataNormalizer_SWSJ
import daily_infer_0625 as InferTool
import cal_value as CalTool


def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--GPU', type=int, default=0, help='[0,1,2..]表示用哪一块GPU, 默认为0, -1表示不使用GPU')
    parser.add_argument('--MODELTYPE', type=str, default='A', help='[A,B,SUB01,SUPER]')
    parser.add_argument('--minw', type=float, default=0.002, help='最小权重, 默认为0.001')
    parser.add_argument('--ext_day', type=int, default=0, help='额外向历史方向读取数据多少天(用于过滤)')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--skipLoss', type=float, default=1.0,
                        help='跳过开盘跌幅大于skipLoss的股票, 默认为1.0(即100%,不跳过)')
    parser.add_argument('--cfd', type=str, default='20240101', help='交易日期起始点')
    parser.add_argument('--cld', type=str, default='20250630', help='交易日期终止点')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--price_target', type=str, default='close_hfq', help='交易使用的价格字段,默认为空')
    parser.add_argument('--model', type=str, default='', help='直接指定单一模型')
    parser.add_argument('--model2', type=str, default='', help='直接指定单一模型')
    parser.add_argument('--model3', type=str, default='', help='直接指定单一模型')
    parser.add_argument('--period', type=str, default='week', help='按周期分组')
    parser.add_argument('--G', type=int, default=200, help='股票小组的大小')
    parser.add_argument('--withBJ', type=int, default=1, help='是否包含北交所股票')
    parser.add_argument('--BJMAX', type=int, default=0, help='北交所股票总权重之上限,默认为0-不设限')
    parser.add_argument('--min_total_mv', type=int, default=0, help='限制进入推理池的股票的最小市值,默认为0表示不限制')
    parser.add_argument('--mask_mf', type=int, default=4,
                        help='资金数据处理[0不处理 2全部清零 3只在全局塔上清零 4 直接去除]')
    parser.add_argument('--shuffle', type=int, default=1, help='排序参数')
    parser.add_argument('--split_year', type=int, default=1, help='是否分年,默认为1')
    parser.add_argument('--indexName', type=str, default='全A',
                        help='大盘指数["FOOL","WEIGHT","中证1000","北证50","国证2000","沪深300","万得小市值","自由现金流"]')
    parser.add_argument('--stockpool', type=str, default='',
                        help='指定推理股票池,如"沪深300",则当前路径下需存在名为"沪深300成分股.json文件"')
    args = parser.parse_args()
    args.device = 'cpu'
    if torch.cuda.is_available() and 0 <= args.GPU < torch.cuda.device_count():
        args.device = f'cuda:{args.GPU}'
    args.withBJ = args.withBJ == 1

    args.setZeroIndex = []
    if args.DS == 'day304':
        args.g_csv_path = r"/data/raw_generated_data/wind/304f_pad_csv"
        args.scaler_file = r"/data/yy_data/wind/304f_20100104/scaler_info.json"
    elif args.DS == 'day104':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/104f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/104f_250618/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    elif args.DS == 'day106':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/106f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/106f_250619/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    elif args.DS == 'day1062':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/106f_pad_csv_calendar"  # 每日增量更新
        # args.g_csv_path = r"/data/yy_data/tushare_data/ts_260201_106f/106f_pad_csv_0201" # 冻结到20260130
        args.scaler_file = r"/data/yy_data/tushare_data/ts_260201_106f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 如果使用1062数据集回测,该数据集中的天数据是严格按照交易日历来生成的,没有padd节假日数据了
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
        args.setZeroIndex = [57, 58, 99, 100]
    else:  # day111
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = r"/data/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.g_csv_path)

    spIndex = ["FOOL", "WEIGHT", "中证1000", "北证50", "国证2000", "沪深300", "万得小市值", "自由现金流", "中证全A",
               "同花顺全A", "全A", "8841425"]
    if args.indexName not in spIndex:
        args.indexName = '全A'
    if args.cld >= '20260101':
        args.indexName = '中证1000'

    # 20260226 ADD 欣健的模型中,对资金流通道是直接剔除,对另外4个(信息评分与集合竞价)也是直接剔除,共22个
    # 定义 mask_mf=5 为专用参数
    if args.mask_mf == 5 and args.DS == 'day1062':
        rm_fields = [i + 33 for i in range(18)]  # 18个资金流
        rm_fields.extend(args.setZeroIndex)  # 另外4个
        args.keep_fields = [i for i in range(105)]  # 全部字段序列,一共105个
        for rk in rm_fields:
            # 删除这些要剔除的字段序号
            args.keep_fields.remove(rk)
        args.setZeroIndex = []

    # assert os.path.exists(args.model), f"模型文件{args.model}不存在"

    args.pool = None
    if args.stockpool == '沪深300':
        # 沪深300成分股
        hs_stock_file = "/data/raw_generated_data/ths_index/hushen300/hushen300_code_index.json"
        # hs_stock_file= "沪深300月度成分.json"
        with open(hs_stock_file, 'r') as f:
            ret = json.load(f)
        args.pool = ret
    elif args.stockpool != '':
        spf = f"{args.stockpool}成分股.json"
        assert os.path.exists(spf), f"指定的推理股票池文件{spf}不存在"
        args.pool = UTILS.read_pmfile(spf)

    if 'open' in args.model:
        print(f"*****OPEN交易,当前skipLoss={args.skipLoss},minw={args.minw},请注意这两个参数是否合理!!!")

    return args






def load_from_file(fn):
    if os.path.exists(fn):
        with open(fn, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def show_pool_info(tradeDay,sim_poolFile,real_poolFile):
    sim_pool = load_from_file(sim_poolFile)
    real_pool = load_from_file(real_poolFile)
    if sim_pool is None:
        print(f"{tradeDay}-模拟池文件不存在")
        return
    if real_pool is None:
        print(f"{tradeDay}-实盘池文件不存在")
        return
    same = len(set(sim_pool) & set(real_pool))
    print(f"{tradeDay}--推理股票池分析---")
    print(f"OUT1,{tradeDay},模拟池,{len(sim_pool)},实盘池,{len(real_pool)},重合部,{same},占比,{200*same/(len(sim_pool)+len(real_pool)):.2f}%")


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
    sS = len(sameStock)*2 / (len(stockA) + len(stockB))

    sameW = 0
    for s in sameStock:
        if s == 'CASH':
            continue
        # 将股票权重都分别反归一化(即假定全仓股票,现金为0)
        va = wa[s] / totalA
        vb = wb[s] / totalB
        sameW +=min(va,vb) # 取二者较小值为重合度
    return sameW,sS

def key_up(d):
    new_d = {}
    for k,v in d.items():
        new_d[str.upper(k)] = v
    return new_d


def show_weight_info(tradeDay,sim_pmfile,real_pmfile):
    sim_w = key_up(load_from_file(sim_pmfile))
    real_w = key_up(load_from_file(real_pmfile))
    sameW,sameS = cal_same_day_weight(sim_w,real_w)
    print(f"{tradeDay}--权重相似度分析---")
    print(f"OUT2,{tradeDay},模拟股票数,{len(sim_w)-1},实盘股票数,{len(real_w)-1},股票重合度,{sameS*100:.2f}%,权重重合度,{sameW*100:.2f}%")


if __name__ == '__main__':
    args = get_args()

    tradeDayList = DT.trade_date_list('20251010', '20260224')
    tradeDayList = DT.get_rid_of_none_trading_days(tradeDayList)
    print(tradeDayList)

    for tradeDay in tradeDayList:
        # 模拟盘权重文件和股票池文件路径
        sim_pmfile = os.path.join('./simout', f"{tradeDay}_SCP_Day_1003_BK3_epoch11_G200_s1.0_m0.002_SF1.json")
        sim_poolFile = os.path.join('./simout', f"{tradeDay}_SCP_Day_1003_BK3_epoch11_G200_s1.0_m0.002_SF1.pool")
        if not os.path.exists(sim_pmfile):
            print(f"{sim_pmfile}---not exist")
            continue

        # 实盘的权重文件和股票池文件路径
        real_pmfile = os.path.join('./realout', f"{tradeDay}.json")
        real_poolFile = os.path.join('./realpool', f"{tradeDay}.json")
        if not os.path.exists(real_pmfile):
            print(f"{real_pmfile}---not exist")
            continue

        # 分析两个池子的区别(如果任意有一个文件不存在,则忽略)
        show_pool_info(tradeDay,sim_poolFile,real_poolFile)

        # 分析两个权重图的区别
        show_weight_info(tradeDay,sim_pmfile,real_pmfile)

        sim_value, _, _, _ = CalTool.cal_value_by_pmfile(sim_pmfile, output=False, price_target=args.price_target)
        real_value, _, _, _ = CalTool.cal_value_by_pmfile(real_pmfile, output=False, price_target=args.price_target)

        # 查看收益区别
        print(f"{tradeDay}--预期收益分析--")
        print(f"OUT3,{tradeDay},模拟收益,{sim_value:.4f},实盘收益,{real_value:.4f},差异率,{(sim_value-real_value)/real_value*100:.2f}%")
    print("===DONE===")