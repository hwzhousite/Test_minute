"""
产品组装测试-2
需要三个输入
1. 一个业绩基准指数
2. A模型输出路径(该路径下应该包括本次测试所有交易日的json调仓记录文件)
3. B模型输出路径
"""
import argparse
import copy
import json
import os
from time import time
import numpy as np
import utils as UTILS
import common_utils as CM
import product_eval as EVTOOL
import utils_date as DT

# 当前支持的业绩比较基准指数(请注意,并不是所有指数都覆盖了所有的测试时段)
g_indexBase = {
    'zz1000': "中证1000.json",
    'bz50': "北证50.json",
    'gz2000': "国证2000.json",
    'hs300': "沪深300.json",
    'zyxjl': "自由现金流.json",
    'wd': "万得小市值.json",
    'zzqa': "中证全A.json",
    'ths': "同花顺全A.json",
    'wdclose': "8841425.json",
    'qa': "全A.json",
    '159201': "zyxjl159201.json"
}


def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    parser.add_argument('--out_path', type=str, default='./out', help='如果有输出文件时,保存到这个路径下')
    parser.add_argument('--save_json', type=int, default=0, help='保存拼装文件')

    # 下面这个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path_a', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--match_a', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--path_b', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--match_b', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='更优先的匹配关键字表示年份')

    parser.add_argument('--mark', type=str, default='ASSEMBLE', help='标识,输出某些文档时以此命名')
    parser.add_argument('--indexName', type=str, default='qa',
                        help='业绩比较基准,请确保/data/indexData/下有同名的json文件')

    parser.add_argument('--f', type=float, default=0.5, help='ETF基金(或者是A产品)在总资产中的占比')
    parser.add_argument('--minw', type=float, default=100,
                        help='拼装完成后做平滑处理的最小权重,如果值大于1则表示不做平滑处理')
    parser.add_argument('--only_same', type=int, default=0,
                        help='是否只保留两个产品的共同部分,只对两个产品的拼装有效,默认为0')

    args = parser.parse_args()
    args.only_same = args.only_same > 0
    assert args.path_b != '' and os.path.exists(args.path_b), f"必须指定合法存在的权重文件路径:{args.path_b}"
    assert 0 <= args.f <= 1.0, f"请指定合法的f系数{args.f}"
    if args.out_path != '' and (not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

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
    else:  # day111
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = r"/data/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path, args.indexName)

    args.indexFile = os.path.join("/data/indexData/", f"{args.indexName}.json")
    print(f"业绩比较基准为<{args.indexName}>,数据文件<{args.indexFile}>")
    assert os.path.exists(args.indexFile), "指数数据文件不存在"

    return args


def make_abs_value_seris(vlist):
    # 根据相对收益序列生成绝对净值序列(累乘)
    abs_values = []
    v = 1.
    for x in vlist:
        v = v * x
        abs_values.append(v)
    return abs_values


def cal_year_info(wA, all_stock_info):
    # 计算每日持股数量
    holdStock = []
    cash_weight = []
    T1 = []
    TR = []
    CMV = []
    BJW = []

    # 平均持股数 / T1 / TR / 市值 /北交所股票权重
    dayList = list(wA.keys())
    DL = len(dayList)
    prevW = None
    for i in range(DL):
        day = dayList[i]
        dayW = wA[day]  # 今日权重

        cash_weight.append(dayW['CASH'])  # 今日现金权重

        hs = len(dayW) - 1  # 持股数
        holdStock.append(hs)

        top1 = get_top1_stock(dayW)  # TOP1
        T1.append(top1)

        tr = cal_tr(prevW, dayW)  # 换手率
        prevW = dayW
        TR.append(tr)

        mv = cal_cmv(day, dayW, all_stock_info)
        if mv >= 0:
            CMV.append(mv)

        bjsw = cal_bjweight(dayW)
        BJW.append(bjsw)

    hs = sum(holdStock) / len(holdStock)
    t1 = sum(T1) / len(T1)
    tr = sum(TR) / len(TR)
    mv = sum(CMV) / len(CMV)
    bjw = sum(BJW) / len(BJW)
    cash = sum(cash_weight) / len(cash_weight)

    return [hs, cash, t1, tr, mv, bjw]


def anaylist_seris(wDayList, aW_change, bW_change, indexBase, weight_AllDay, all_stock_info):
    """
    :param wDayList  这是一个时间序列[[周内首个交易日期,周内最后交易日期]....]
    :param aW_change 这是一个相对收益序列,表示每一周模型的涨跌幅 0附近
    :param bW_change 这是一个作为对照的业绩基准的每周涨跌幅 0附近
    :param indexBase 业绩基准指数(每天的),时间跨度可能远超过当前这个时间段
    :param weight_AllDay 在区间内每一天的权重
    :param all_stock_info 所有股票信息
    """

    firstDay = wDayList[0][0]  # 全序列的首个交易日
    lastDay = wDayList[-1][1]  # 全序列的末尾交易日

    cEx = [x - y for x, y in zip(aW_change, bW_change)]  # 每周超额
    absEx = [abs(x) for x in cEx]
    # 指标2: 计算技术指标
    aW_rel_value = [x + 1 for x in aW_change]  # 每周的相对收益序列,1附近
    bW_rel_value = [x + 1 for x in bW_change]  # 每周的基准相对收益序列,1附近

    # 每周的净值序列,真实价值, 根据每周相对价值累乘得来
    aW_abs_value = make_abs_value_seris(aW_rel_value)

    # 根据日期计算业绩基准指数的净值
    llday = DT.get_real_next_trade_day(lastDay, True)  # 最末尾交易日再向未来方向的一个交易日
    inv = indexBase[llday] / indexBase[firstDay]
    bW_abs_value = [inv]

    sr = CM.get_SR(aW_rel_value, period='week')
    ir = CM.get_ir(aW_rel_value, bW_rel_value, period='week')
    LAP = CM.get_bigloss_and_return(aW_rel_value, bW_rel_value)
    beta = CM.get_beta(aW_rel_value, bW_rel_value)  # 计算beta系数
    offensive = CM.get_offensive_power(aW_rel_value, bW_rel_value)
    defensive = CM.get_defensive_power(aW_rel_value, bW_rel_value)
    downrisk = CM.get_down_risk(aW_rel_value, period='week', no_risk_year_p=0.02)

    mdd = UTILS.max_drawdown(aW_abs_value)
    mee = sum(cEx) / len(cEx)  # 平均每周超额
    std_ex = np.std(np.array(cEx), ddof=1)  # 周度超额的标准差
    victory = sum(x > 0 for x in cEx) / len(cEx)  # 胜率

    ex_value_list = [1.]
    ex_value = 1.
    for i in range(len(aW_rel_value)):
        x = aW_rel_value[i] / bW_rel_value[i]
        ex_value = ex_value * x
        ex_value_list.append(ex_value)
    emdd = UTILS.max_drawdown(ex_value_list)
    absEE = sum(absEx) / len(absEx)
    year_info = cal_year_info(weight_AllDay, all_stock_info)

    abs_win_rate, win_loss_rate = CM.cal_win_loss_rate(aW_rel_value)  # 绝对胜率与盈亏比
    # avg_hold_days = CM.cal_avg_hold_days(weight_AllDay)

    info = f"{wDayList[0][0]}-{wDayList[-1][1]},{len(wDayList)},{aW_abs_value[-1]:.4f},{bW_abs_value[-1]:.4f},{(aW_abs_value[-1] - bW_abs_value[-1]) / bW_abs_value[-1] * 100:.2f}%,{mee * 100:.2f}%,{std_ex:.4f},{absEE * 100:.2f}%"
    info = info + f",{sr:.4f},{ir:.4f},{mdd * 100:.2f}%,{emdd * 100:.2f}%"
    info = info + f",{LAP[0]:.1f},{LAP[1]:.1f},{LAP[2]:.1f},{LAP[3]:.1f}"
    info = info + f",{downrisk:.4f},{beta:.4f},{offensive:.4f},{defensive:.4f},{victory * 100:.2f}%,{abs_win_rate * 100:.2f}%,{win_loss_rate:.2f}"
    info = info + f",{year_info[0]:.1f},{year_info[1]:.2f},{year_info[2]:.3f},{year_info[3]:.2f},{year_info[4]:.1f},{year_info[5]:.2f}"
    # print(f"{info}")
    return info


def split_year_series(wDayList, aW_change, bW_change):
    ret_dict = {}
    DL = len(wDayList)
    for i in range(DL):
        day = wDayList[i]
        year = int(day[0][:4])
        if not ret_dict.__contains__(year):
            ret_dict[year] = {'daylist': [],
                              'model': [],
                              'base': [],
                              }
        ret_dict[year]['daylist'].append(day)
        ret_dict[year]['model'].append(aW_change[i])
        ret_dict[year]['base'].append(bW_change[i])
    return ret_dict


def is_same_day_list(a, b):
    LA = len(a)
    LB = len(b)
    LC = len(set(a) & set(b))
    return LA == LB and LA == LC


def combin_weight_oneday(wa, wb):
    w = copy.deepcopy(wa)
    for k, v in wb.items():
        if w.__contains__(k):
            continue
        w[k] = v
    return w


def combin_weight(wa, wb):
    # 将两个权重图合并(不要求数据一致性)
    wr = {}
    daylist = list(wa.keys())
    for d in daylist:
        wr[d] = combin_weight_oneday(wa[d], wb[d])
    return wr


def get_sub_dict(weight, year):
    wy = {}
    for k, v in weight.items():
        ky = int(k[:4])
        if ky == year:
            wy[k] = copy.deepcopy(v)
    return wy


def get_top1_stock(w):
    mv = 0.
    for k, v in w.items():
        if k == 'CASH' or k == 'ETF':
            continue
        if v > mv:
            mv = v
    return mv


def cal_tr(wa, wb):
    if wa is None:
        return 1.
    sa = list(set(wa.keys()) | set(wb.keys()))
    tr = 0.
    for code in sa:
        if code == 'CASH':
            continue
        va = 0.
        if wa.__contains__(code):
            va = wa[code]
        vb = 0.
        if wb.__contains__(code):
            vb = wb[code]
        tr += abs(va - vb)
    return tr / 2.


def cal_bjweight(w):
    bjw = 0.
    for k, v in w.items():
        if k.endswith("BJ") or k.endswith("bj"):
            bjw += v
    return bjw


def smooth_stock_mv(stock_mv):
    """
    对股票的流通市值做平滑分档处理,以避免极端值对统计数据的影响
    """

    return stock_mv if stock_mv < 1000 else 1000

    # mv_list = [0, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # if stock_mv <= 0:
    #     # 数据错误
    #     return 10.
    # if stock_mv >= mv_list[-1]:
    #     return mv_list[-1]
    # for i in range(0, len(mv_list) - 1):
    #     if mv_list[i] <= stock_mv < mv_list[i + 1]:
    #         return (mv_list[i] + mv_list[i + 1]) / 2.
    # never goes here


def cal_cmv(day, dayW, all_stock_info):
    """
    统计平均CMV
    # all_stock_info: # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    """

    if dayW['CASH'] >= 0.999:
        return -1
    totalStockWeight = 1. - dayW['CASH']
    if dayW.__contains__('ETF'):
        totalStockWeight = 1. - dayW['ETF']
    cmv = 0.
    datafiels = list(UTILS.g_field_index.keys())
    pi = CM.find_str_in_list(datafiels, 'mv')
    for code, w in dayW.items():
        if code == 'CASH' or code == 'ETF':
            continue
        codeInfo = all_stock_info[code][day]
        stock_mv = codeInfo[pi] / 10000.  # 流通市值单位为万,转换成亿
        stock_mv = smooth_stock_mv(stock_mv)  # 做平滑处理
        tmv = stock_mv * w
        tmv = tmv / totalStockWeight
        cmv += tmv
    return cmv


def output_ex(args, wDL, cmodel, cbase):
    fn = f'{args.mark}_每周超额数据.csv'
    fn = os.path.join(args.out_path, fn)
    with open(fn, "w", encoding='utf-8-sig') as f:
        print(f"日期,模型累计净值,{args.indexName}累计净值,模型当期收益,{args.indexName}当期收益,超额收益,输赢", file=f)
        DL = len(wDL)
        v1 = 1.0
        v2 = 1.0
        for i in range(DL):
            ex = cmodel[i] - cbase[i]
            w = 1 if ex >= 0 else 0
            v1 = v1 * (1.0 + cmodel[i])
            v2 = v2 * (1.0 + cbase[i])
            print(
                f"{wDL[i][0]}-{wDL[i][1]},{v1:.4f},{v2:.4f},{cmodel[i] * 100:.2f}%,{cbase[i] * 100:.2f}%,{ex * 100:.2f}%,{w}",
                file=f)


def keep_same_trade_days(wa, wb):
    """
    对输入的两个交易序列进行对比,只保留相同的交易日期
    """

    dayListA = list(wa.keys())
    dayListB = list(wb.keys())
    if is_same_day_list(dayListA, dayListB):
        # 日期序列完全相同
        return wa, wb

    # 日期序列不相同,则只取重复的部分
    sameKeys = list(set(dayListA) & set(dayListB))
    sameKeys = sorted(sameKeys)
    newA = {}
    newB = {}
    for k in sameKeys:
        newA[k] = wa[k]
        newB[k] = wb[k]
    return newA, newB


def output_json(args, weights_S):
    for day, w in weights_S.items():
        fn = os.path.join(args.out_path, f"{day}.json")
        with open(fn, 'w', encoding="utf-8") as file:
            json.dump(w, file, indent=2)


def assemble_product(args, path_a, match_a, path_b, match_b, match_year, period='week'):
    """
    按指定的比例f拼装两个不同的产品
    """

    # 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights_A = UTILS.read_weigh_from_path(path_a, match_a, match_year)
    weights_B = UTILS.read_weigh_from_path(path_b, match_b, match_year)
    weights_A, weights_B = keep_same_trade_days(weights_A, weights_B)
    DayCount = len(weights_A)
    dayList = list(weights_A.keys())
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    # 生成组合权重图
    weights_S = EVTOOL.make_combin_weight(weights_A, weights_B, args.f, only_same=args.only_same, minw=args.minw)

    if args.save_json == 1:
        output_json(args, weights_S)

    # 下面读取数据时需要用到全部出现过的股票
    weights_GG = EVTOOL.make_combin_weight(weights_A, weights_B, args.f, only_same=False, minw=100)

    # 读取指数数据
    indexBase = CM.load_index(args.indexFile)

    # 读取所有股票信息
    print(f"{CM.timestr()}读取所有交易日股票信息...")
    ts = time()
    # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    all_stock_info = UTILS.read_all_stock_info(weights_GG)
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = EVTOOL.make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # 计算每日收益,这里会计算每日绝对收益(扣除成本)!!!
    # 下面的方法返回:
    # 1.每天模型的相对前一天的价值(税后)--税后是指扣除了交易成本
    # 2.每天盈利股票
    # 3.每天带来亏损的股票
    model_value_A, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights_A, priceInfo)  # 计算A模型收益
    model_value_B, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights_B, priceInfo)  # 计算B模型收益
    base_value = EVTOOL.cal_seris_value(indexBase, dayList)  # 计算同期的基准收益

    # A/B两个模型的相似度信息,返回信息list:[IC系数,股票重合度,权重重合度]
    similarity = EVTOOL.cal_similarity(weights_A, model_value_A, weights_B, model_value_B, base_value, period='day')

    # 下面根据f参数指数的比例对etf_value和model_value进行配比,从而得到新的组合产品的价值序列
    assemble_value, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights_S, priceInfo)

    S_change, wDayList = EVTOOL.make_seris_value(assemble_value, vtype='change', period=period)
    base_change, _ = EVTOOL.make_seris_value(base_value, vtype='change', period=period)

    S_change_Day, _ = EVTOOL.make_seris_value(assemble_value, vtype='change', period='day')
    base_change_Day, _ = EVTOOL.make_seris_value(base_value, vtype='change', period='day')

    print(
        f"***每日涨跌幅序列的相关性[{dayList[0]}-{dayList[-1]}]***交易价格-{args.targetPrice}-交易成本{args.tradeCost}")
    print(f"{args.f:.2f}:{path_a}---{(1 - args.f):.2f}:{path_b}")
    print(
        f"收益IC:{similarity[0]} 超额IC:{similarity[1]} 股票重合度:{similarity[2]:.4f} 权重重合度:{similarity[3]:.4f}")
    correlation = np.corrcoef(S_change_Day, base_change_Day)[0, 1]
    correlation = round(float(correlation), 4)
    print(f"组合模型与业绩基准{args.indexName}的收益IC:{correlation:.4f}")

    # 将整个序列按年切分, 生成两重字典{key=year,value={'daylist':[],'model':[],'base':[]}}
    year_data = split_year_series(wDayList, S_change, base_change)
    total_years = len(year_data)
    print(
        f"区间,周期数,累计价值,大盘{args.indexName},总超额,MEAN_EX,STD_EX,MAE,SR,IR,MDD,EMDD,MaxLossA,MaxLossR,RepairA,RepairR,下行风险,BETA,进攻能力,防守能力,相对胜率,绝对胜率,盈亏比,持股数,CASH,T1,TR,CMV,BJW")
    for y, d in year_data.items():
        wA = get_sub_dict(weights_S, y)
        info = anaylist_seris(d['daylist'], d['model'], d['base'], indexBase, wA, all_stock_info)
        print(f"{info}")
    if total_years > 1:
        info = anaylist_seris(wDayList, S_change, base_change, indexBase, weights_S, all_stock_info)
        print(f"{info}")

    output_ex(args, wDayList, S_change, base_change)


def assemble_product_for_ETF(args, etfName, path_b, match_b, match_year, period='week'):
    """
    按指定的比例f拼装两个不同的产品(且第1个产品是一个指定的指数或者etf)
    """

    # 读取ETF数据
    etfFile = g_indexBase[etfName]
    etfBase = CM.load_index(etfFile)

    # 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights_B = UTILS.read_weigh_from_path(path_b, match_b, match_year)
    DayCount = len(weights_B)
    dayList = list(weights_B.keys())
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    # 生成组合权重图
    weights_S = EVTOOL.make_combin_weight_with_ETF(weights_B, args.f)

    # 读取业绩基准指数数据
    indexBase = CM.load_index(args.indexFile)

    # 读取所有股票信息
    print(f"{CM.timestr()}读取所有交易日股票信息...")
    ts = time()
    # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    all_stock_info = UTILS.read_all_stock_info(weights_S)
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = EVTOOL.make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # 计算同期的业绩基准收益
    base_value = EVTOOL.cal_seris_value(indexBase, dayList)
    # 下面根据f参数指数的比例对etf_value和model_value进行配比,从而得到新的组合产品的价值序列
    assemble_value, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights_S, priceInfo, etf=etfBase)

    S_change, wDayList = EVTOOL.make_seris_value(assemble_value, vtype='change', period=period)
    base_change, _ = EVTOOL.make_seris_value(base_value, vtype='change', period=period)

    # 20260305 相关性计算改为用日数据计算(避免被周数据平滑掉)
    S_change_Day, _ = EVTOOL.make_seris_value(assemble_value, vtype='change', period='day')
    base_change_Day, _ = EVTOOL.make_seris_value(base_value, vtype='change', period='day')
    print(
        f"***每日涨跌幅序列的相关性[{dayList[0]}-{dayList[-1]}]***交易价格-{args.targetPrice}-交易成本{args.tradeCost}")
    print(f"{args.f:.2f}:{etfName}---{(1 - args.f):.2f}:{path_b}")
    correlation = np.corrcoef(S_change_Day, base_change_Day)[0, 1]
    correlation = round(float(correlation), 4)
    print(f"组合模型与业绩基准{args.indexName}的IC:{correlation:.4f}")

    # 将整个序列按年切分, 生成两重字典{key=year,value={'daylist':[],'model':[],'base':[]}}
    year_data = split_year_series(wDayList, S_change, base_change)
    total_years = len(year_data)
    print(
        f"区间,周期数,累计价值,大盘{args.indexName},总超额,MEAN_EX,STD_EX,MAE,SR,IR,MDD,EMDD,MaxLossA,MaxLossR,RepairA,RepairR,下行风险,BETA,进攻能力,防守能力,总胜率,持股数,CASH,T1,TR,CMV,BJW")
    for y, d in year_data.items():
        wA = get_sub_dict(weights_S, y)
        info = anaylist_seris(d['daylist'], d['model'], d['base'], indexBase, wA, all_stock_info)
        print(f"{info}")
    if total_years > 1:
        info = anaylist_seris(wDayList, S_change, base_change, indexBase, weights_S, all_stock_info)
        print(f"{info}")

    output_ex(args, wDayList, S_change, base_change)


if __name__ == '__main__':
    args = get_args()
    # 按照命令行参数给定的路径和参数进行模型拼装,并输出分年统计信息
    if g_indexBase.__contains__(args.path_a):
        # 如果参数path_a指向的是一个ETF或者指数,则表明当前用于拼装的A产品是一个指数基金
        assemble_product_for_ETF(args, args.path_a, args.path_b, args.match_b, args.match_year)
    else:
        assert args.path_a != '' and os.path.exists(args.path_a), f"必须指定合法存在的权重文件路径:{args.path_a}"
        assemble_product(args, args.path_a, args.match_a, args.path_b, args.match_b, args.match_year)
