"""
产品分析--极端情况(盈亏)
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


def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    parser.add_argument('--out_path', type=str, default='', help='如果有输出文件时,保存到这个路径下')

    # 下面这个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='更优先的匹配关键字表示年份')

    parser.add_argument('--mark', type=str, default='ASSEMBLE', help='标识,输出某些文档时以此命名')
    parser.add_argument('--indexName', type=str, default='qa',
                        help='业绩比较基准,请确保/data/indexData/下有同名的json文件')

    args = parser.parse_args()

    assert args.path != '' and os.path.exists(args.path), f"必须指定合法存在的权重文件路径:{args.path}"
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
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/106f_pad_csv_calendar"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/ts_260201_106f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 如果使用1062数据集回测,该数据集中的天数据是严格按照交易日历来生成的,没有padd节假日数据了
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
    else:  # day111
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = r"/data/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.DS, args.g_csv_path, args.indexName)

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

    return [hs, t1, tr, mv, bjw]


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

    info = f"{wDayList[0][0]}-{wDayList[-1][1]},{len(wDayList)},{aW_abs_value[-1]:.4f},{bW_abs_value[-1]:.4f},{(aW_abs_value[-1] - bW_abs_value[-1]) / bW_abs_value[-1] * 100:.2f}%,{mee * 100:.2f}%,{std_ex:.4f},{absEE * 100:.2f}%"
    info = info + f",{sr:.4f},{ir:.4f},{mdd * 100:.2f}%,{emdd * 100:.2f}%"
    info = info + f",{LAP[0]:.1f},{LAP[1]:.1f},{LAP[2]:.1f},{LAP[3]:.1f}"
    info = info + f",{downrisk:.4f},{beta:.4f},{offensive:.4f},{defensive:.4f},{victory * 100:.2f}%,{year_info[0]:.1f},{year_info[1]:.3f},{year_info[2]:.2f},{year_info[3]:.1f},{year_info[4]:.2f}"
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
        tmv = codeInfo[pi] * w
        tmv = tmv / totalStockWeight
        cmv += tmv
    return cmv / 10000.


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


def get_top_key(some_dict, top, biggest=True):
    """
    给定一个字典,从中取出最大或者最小的前top个数据的key
    :param some_dict 给定的一个数据字典
    :param top 要取回的top数量
    :param biggest 表示是要取最大的还是取最小的, 默认为True取最大的
    """

    sorted_w = sorted(some_dict.items(), key=lambda x: x[1], reverse=biggest)
    topKeys = []
    size = min(top, len(sorted_w))
    for i in range(size):
        topKeys.append(sorted_w[i][0])
    return topKeys


def cal_cash_value(weights_A, base_value):
    """
    weights_A 是连续多天的权重图{key=day,value={weight}}
    base_value 是连续多天的基准指数相对收益(1附近的值) {key=day,value=v}
    """

    CMIN = 0.001  # 持有现金超过这个值的才算有意义的现金,否则都忽略
    Day = len(weights_A)
    CDay = 0
    CashPos = 0
    daylist = list(weights_A.keys())
    CV = 1.
    CVP = 1.
    CVN = 1.
    for i in range(len(daylist) - 1):
        day = daylist[i]
        w = weights_A[day]
        wc = w['CASH']
        if wc < CMIN:
            continue
        nextday = daylist[i + 1]
        if not base_value.__contains__(nextday):
            continue
        marketValue = base_value[nextday]
        CDay += 1
        if marketValue < 1.:
            # 如果大盘跌了,则认为持有现金是正确的避险行为
            CashPos += 1
        # 假定全部股票(权重)都按照大盘涨跌来计算收益, 而wc(cash)部分维持现金价值, 求和之后再与大盘比较(相除)
        # 如果大于1,则表明持有现金带来了正收益,反正则是折损
        x = ((1. - wc) * marketValue + wc) / marketValue
        CV = CV * x
        if x >= 1.:
            CVP = CVP * x
        else:
            CVN = CVN * x
    print("*****检测现金项所产生的作用*****")
    print("假定全部股票(权重)都按照大盘涨跌来计算收益, 而wc(cash)部分维持现金价值, 求和之后再与大盘比较(相除),如果大于1,则表明持有现金带来了正收益,反正则是折损")
    print(f"总交易天数{Day},其中持有现金大于{CMIN}的天数为{CDay},占比{CDay / Day * 100:.2f}%")
    print(f"现金避险正确(大盘下跌,持有现金大于{CMIN})天数{CashPos},占比{CashPos / CDay * 100:.2f}%")
    print(f"现金避险错误(大盘上涨,持有现金大于{CMIN})天数{CDay - CashPos},占比{(CDay - CashPos) / CDay * 100:.2f}%")
    print(f"正作用累计价值{CVP:.2f},负作用累计价值{CVN:.2f},现金行为累计价值{CV:.2f}\n")


def analysis_product(args, path_a, match_a, match_year, period='week'):
    """
    极端盈亏情况分析
    """

    # 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights_A = UTILS.read_weigh_from_path(path_a, match_a, match_year)
    DayCount = len(weights_A)
    dayList = list(weights_A.keys())
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    # 读取指数数据
    indexBase = CM.load_index(args.indexFile)

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
    # 1.每天模型的相对前一天的价值(税后)--税后是指扣除了交易成本
    # 2.每天盈利股票
    # 3.每天带来亏损的股票
    model_value_A, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights_A, priceInfo)  # 计算A模型收益
    base_value = EVTOOL.cal_seris_value(indexBase, dayList)  # 计算同期的基准收益

    # 20260319 ADD 计算现金项带来的收益
    cal_cash_value(weights_A, base_value)

    # 生成按周的收益率(0附近)
    model_change, wDayList = EVTOOL.make_seris_value(model_value_A, vtype='change', period=period)
    base_change, _ = EVTOOL.make_seris_value(base_value, vtype='change', period=period)

    # 生成一个超额数据字典, key=[],value=ex
    ex_dict = {}
    plen = len(wDayList)
    for i in range(plen):
        k = f"{wDayList[i][0]},{wDayList[i][1]}"
        ex_dict[k] = model_change[i] - base_change[i]

    print(f"*****{path_a}*****")

    # 排序并取出超额最高的前N个数据
    N = 20
    print(f"单周算术超额最大top{N}")
    topKK = get_top_key(ex_dict, top=N, biggest=True)
    for k in topKK:
        print(f"{k} {ex_dict[k] * 100:.2f}%")
    print()

    # 排序并取出超额最低的前N个数据
    print(f"单周算术超额最小top{N}")
    topKK = get_top_key(ex_dict, top=N, biggest=False)
    for k in topKK:
        print(f"{k} {ex_dict[k] * 100:.2f}%")
    print()

    # 查找是否存在涨跌且加仓或者跌停且减仓的行为
    print("检查是否有非法的加减仓行为...")
    EVTOOL.check_up_down_stop(weights_A, all_stock_info)
    print("检查是否有非法的加减仓行为,DONE")
    print()

    # 查找在买入后出现连
    print("查找在买入后至少吃到一个涨停的情况...")
    EVTOOL.check_upstop_earn(weights_A, all_stock_info)

    # 查找历史上最大的若干次换仓行为,后续人工查询冲击情况
    # amount 成交额--原数据中单位为千元
    # 即: 假定本基金总额为1000万,看当日买卖的金额/amount得到一个比例, 看这个比例是否超过10%,如果超过则有巨大风险
    FM = 1000
    print(f"查询剧烈的换仓行为,假定基金规模为{FM}万元.")
    EVTOOL.check_big_change_weight(weights_A, all_stock_info, FM)  # 需要修改为: 每天的最大买和卖出(分开算), 有多少天触达警戒线


if __name__ == '__main__':
    args = get_args()
    analysis_product(args, args.path, args.match, args.match_year)
