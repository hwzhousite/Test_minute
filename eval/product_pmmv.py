"""
尝试建立更多维度的评估指标
为了方便团队协作共享, 我们规定当前脚本的基本输入是一个模型回测时一段时间内连续N个交易日的持仓文件
文件为json格式
"""
import argparse
import copy
import json
import os
import random
from time import time
import numpy as np
import utils as UTILS
import common_utils as CM
import matplotlib

matplotlib.use('Agg')  # 设置后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import utils_date as DT
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    parser.add_argument('--out_path', type=str, default='', help='如果有输出文件时,保存到这个路径下')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    # 程序会从上面这个path指定的路径下去查找所有的json文件, 并且要求json文件名中包含下面的match字符串
    # 如果match字符串为空,则默认查找path路径下所有的json文件
    # 另一个默认前提: 所有json文件名都以8位的日期开头
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='', help='标识,输出某些文档时以此命名')
    parser.add_argument('--pdf', type=int, default=0, help='是否输出PDF')

    args = parser.parse_args()

    assert args.path != '' and os.path.exists(args.path), f"必须指定合法存在的权重文件路径:{args.path}"
    if args.out_path != '' and (not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    if args.mark == '':
        args.mark = Path(args.path).name

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
        # 检查虚拟内存数据文件是否存在, 如果存在,则指向这个内存数据文件目录,加快读取速度
        # 需要运行当前目录下的memdata.sh来加载和初始化
        # 服务器未重启的话,生命期内只需要运行一次, 当然,如果真实的数据更新, 这个内存文件并不会自动同步, 需要手动更新
        # virtual_mem_data_path = '/memdata_virtual/106f_pad_data/'
        # # 这里的106f_pad_data是在memdata.sh中定义的
        # if os.path.exists(virtual_mem_data_path):
        #     args.g_csv_path = virtual_mem_data_path
    else:  # day111
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = r"/data/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path)
    print(f"数据路径:{args.g_csv_path}")
    return args


def reverse_dict_valueIsInt(ad):
    reversed_dic = {}
    for k, v in ad.items():
        v = int(round(v))
        reversed_dic[v] = k
    return reversed_dic


def check_up_down_stop(weights, priceInfo):
    # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    dayslist = list(weights.keys())
    DL = len(weights)
    for i in range(1, DL):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日-交易日
        wA = weights[dA]  # 今日-持仓
        dB = DT.get_real_next_trade_day(dA, True)
        if weights.__contains__(dB):
            wB = weights[dB]
        else:
            # wB = copy.deepcopy(wA)
            continue
        check_up_down_stop_day(dA, wA, dB, wB, priceInfo)


def find_biggist_change(day, wA, wB, all_stock_info, FM, amI):
    """
    查找最大的买入和最大的(这里的最大/小是指金额之于当日总成交量的占比)
    """

    code = set(wA.keys()) & set(wB.keys())
    big_buy = None
    big_buy_w = 0.
    big_sell = None
    big_sell_w = 0.

    for s in code:
        if s == 'CASH' or s == 'ETF':
            continue

        stock_amount = all_stock_info[s][day][amI] * 0.1  # 当日总成交额
        if stock_amount < 0.1:
            print(f"---严重数据错误---{s}-{day}-总成交额{stock_amount}万")
            continue

        aw = wA.get(s, 0)
        bw = wB.get(s, 0)

        if aw < bw:
            # 买入
            cw = bw - aw
            op_money = cw * FM  # 操作买卖的金额(万)
            opmr = op_money / stock_amount * 100.
            if opmr > big_buy_w:
                big_buy_w = opmr
                big_buy = s
        elif aw > bw:
            # 卖出
            cw = aw - bw
            op_money = cw * FM  # 操作买卖的金额(万)
            opmr = op_money / stock_amount * 100.
            if opmr > big_sell_w:
                big_sell_w = opmr
                big_sell = s
    return big_buy, big_buy_w, big_sell, big_sell_w


def check_big_change_weight(weights, all_stock_info, FM):
    """
    统计排名前N的最大换仓行为,查询它是否有过大的冲击(成本)风险
    判断依据是估算其换手率是否占全天换手率的10%
    FM是假定的基金规模(万元)
    all_stock_info 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    """

    daylist = list(weights.keys())
    DL = len(daylist)
    bigDict = {}  # 把每个交易日中最大的买入占比和最大的卖出占比找出来放到字典

    datafiels = list(UTILS.g_field_index.keys())
    # fmv = CM.find_str_in_list(datafiels, 'mv')  # 流通市值索引
    # fcr = CM.find_str_in_list(datafiels, 'turnover_rate_f')  # 换手率(自由流通股)
    amI = CM.find_str_in_list(datafiels, 'amount')  # 当日成交额(千元)

    for i in range(1, DL):
        pw = weights[daylist[i - 1]]
        cw = weights[daylist[i]]
        day = daylist[i]

        # 查找当天最大的买入最大的卖出权重, 返回值为 bigin_stock,计划买入金额占当日总成交额的占比,bigout_stock,...
        bi, biw, bo, bow = find_biggist_change(day, pw, cw, all_stock_info, FM, amI)
        if bi is not None:
            k = f"{bi}-{day}-BUY"
            bigDict[k] = biw
        if bo is not None:
            k = f"{bo}-{day}-SELL"
            bigDict[k] = bow

    # 对整个字典进行排序,按value(变更权重)来倒排
    bigDict = resort_dict(bigDict, rerverse=True)
    check_warning = 0
    wdays = {}
    for comb_k, wc in bigDict.items():
        info = comb_k.split('-')
        stock = info[0]  # 股票代码
        day = info[1]  # 日期
        op = info[2]  # 操作(买or卖)

        stock_amount = all_stock_info[stock][day][amI] * 0.1  # 该股票当日总成交额(万)
        if stock_amount < 0.1:
            print(f"---严重数据错误---{stock}-{day}-总成交额{stock_amount}万")
            continue

        op_money = wc * stock_amount / 100.  # 操作买卖的金额(万元)   wc是百分数,所以要除个100.
        if wc >= 5:
            print(
                f"WARNING#{check_warning}-{stock}-{day}-{op}-总成交额{stock_amount:.1f}万-操作金额{op_money:.1f}万-占比{wc:.2f}%")
            check_warning += 1
            wdays[day] = 1
    tds = len(wdays)
    totalPct = tds / DL * 100.
    print(f"总交易日{DL}--出现疑似巨大交易冲击的次数{check_warning},天数{tds}--占比{totalPct:.2f}%")


def check_upstop_earn(weights, priceInfo):
    dayslist = list(weights.keys())
    DL = len(weights)
    for i in range(DL):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日-交易日
        wA = weights[dA]  # 今日-持仓
        dB = DT.get_real_next_trade_day(dA, True)
        check_upstop_earn_day(dA, wA, dB, priceInfo)


def cal_trade_series_value(tradeCost, weights, priceInfo, etf=None):
    # 计算每日收益,这里会计算每日绝对收益(扣除成本)!!!
    # 参考大盘的价值计算, 只有一个交易日T, 我们计算大盘价值的方法是以指数在T+1日的收盘去除T日的收盘作为大盘在这一天的价值
    # 因此,我们在计算模型收益的时候也应该使用相同的逻辑,T日调仓,则T日的收益应该用T+1日的价格来计算
    # 这样,所有价值序列的首值都不是1
    # etf如不为None,则表示它是一个ETF基数(指数)数据字典,当权重图中出现股票名称为ETF时,用这个数据来计算其价值
    win_dict = {}  # 每天盈利股票
    lost_dict = {}  # 每天带来亏损的股票
    model_value = {}  # 每天模型的相对价值(税后)--税后是指扣除了交易成本,下同
    dayslist = list(weights.keys())
    DL = len(weights)
    for i in range(DL):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日-交易日
        wA = weights[dA]  # 今日-持仓
        dB = DT.get_real_next_trade_day(dA, True)
        if weights.__contains__(dB):
            wB = weights[dB]
        else:
            wB = copy.deepcopy(wA)
        # 计算今日换仓之后到明天相对盈亏, 具体到每一次股票(以前一日所有持仓股票的视角)
        win, lost, v = cal_trade_win_lost(dA, wA, dB, wB, priceInfo, tradeCost, etf=etf)
        win_dict[dA] = win
        lost_dict[dA] = lost
        model_value[dA] = v
    return model_value, win_dict, lost_dict


def make_day_total_net_value(model_value):
    dayList = list(model_value.keys())
    v = model_value[dayList[0]]
    valueList = [v]
    for i in range(1, len(dayList)):
        v = v * model_value[dayList[i]]
        valueList.append(v)
    return valueList, dayList


def cal_seris_value(serisDict, dayslist):
    """
    根据指数序列生成一个指定时间周期内每天的相对收益
    :param serisDict 指数序列{key=日期,value=指数}
    :param dayslist 一个日期序列list
    :return dict{key=day,value=value}
    """

    serisValue = {}
    DL = len(dayslist)
    for i in range(DL):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日
        dB = DT.get_real_next_trade_day(dA, True)
        v = serisDict[dB] / serisDict[dA]
        serisValue[dA] = v
    return serisValue


def make_seris_value(serisDict, vtype='change', period='week'):
    dayList = list(serisDict.keys())
    serisValue = []
    keylist = []
    v = 1.0
    mv = 0.
    if vtype == 'change':
        # 需要生成的是涨跌幅数据,则下面累乘出来的小周期价值v,都需要减去1.从而变成涨跌幅
        mv = 1.
    startDay = None
    for i in range(len(dayList)):
        day = dayList[i]
        if startDay is None:
            startDay = day
        v = v * serisDict[day]
        if is_period_end_day_in_list(dayList, i, period):
            serisValue.append(v - mv)
            keylist.append([startDay, day])
            v = 1.
            startDay = None
    return serisValue, keylist


def cal_repeat_weight_day(wa, wb):
    """
    计算两天权重的重合度
    股票重合度: 重复股数*2/(a的股票数+b的股票数)
    权重重合度: 重复股票在a和b中的权重(较小者)求和
    """
    sa = list(wa.keys())
    sb = list(wb.keys())
    if len(sa) <= 1 or len(sb) <= 1:
        return 0, 0
    scommon = list(set(sa) & set(sb))

    count = 0
    weight = 0
    for stock in scommon:
        if stock == 'CASH':
            continue
        count += 1
        vwa = wa[stock]
        vwb = wb[stock]
        weight += min(vwa, vwb)
    return 2 * count / (len(sa) + len(sb) - 2), weight


def cal_repeat_weight(wA, wB):
    stock_repeat = []
    weight_repeat = []
    for day, wa in wA.items():
        s, w = cal_repeat_weight_day(wa, wB[day])
        stock_repeat.append(s)
        weight_repeat.append(w)
    return sum(stock_repeat) / len(stock_repeat), sum(weight_repeat) / len(weight_repeat)


def cal_similarity(weights_A, model_value_A, weights_B, model_value_B, base_value, period='week'):
    """
    计算两个模型权重图的相似度
    """

    ret = []
    # 计算两个模型按周度的价值涨跌幅
    A_change, wDayList = make_seris_value(model_value_A, vtype='change', period=period)  # 每周的涨跌幅序列,0附近
    B_change, _ = make_seris_value(model_value_B, vtype='change', period=period)  # 每周的涨跌幅序列,0附近
    base_change, _ = make_seris_value(base_value, vtype='change', period=period)  # 每周的涨跌幅序列,0附近

    # 计算这两个涨跌幅序列的相关系数
    correlation = np.corrcoef(A_change, B_change)[0, 1]
    ret.append(round(float(correlation), 4))

    # 计算两个超额序列的相关系数
    exa = [x - y for x, y in zip(A_change, base_change)]
    exb = [x - y for x, y in zip(B_change, base_change)]
    correlation = np.corrcoef(exa, exb)[0, 1]
    ret.append(round(float(correlation), 4))

    # 计算股票数量重合度和权重重合度
    sw, ww = cal_repeat_weight(weights_A, weights_B)
    ret.append(sw)
    ret.append(ww)
    return ret


def read_price(priceType, weights):
    """
    读取所有交易日的股票交易价格,对每个交易日来说,都要先生成一个前日/今日这2天
    所有持仓股票并集的所有价格信息,因为后续在计算邻近两日的相对价值时要用到
    """

    priceDict = {}  # 生成价格数据字典, key为日期, value是另一个字典[key=stockcode,value=price]
    dayslist = list(weights.keys())
    daysLen = len(dayslist)
    for i in range(daysLen):
        dA = dayslist[i]  # 交易日
        sA = list(weights[dA].keys())  # 交易日持股
        if i > 0:
            # 从第二个交易日起, 当天的价格信息必须得包含前一天所有持仓的股票
            # 因为对于今日清仓的股票, 在权重表中是不存在的, 必须要加进来, 才能在下面的方法中去取到它在今天的卖出价
            dPrev = dayslist[i - 1]
            sPrev = list(weights[dPrev].keys())
            sA = set(sA) | set(sPrev)
            sA = list(sA)
        if 'CASH' in sA:
            sA.remove('CASH')
        # 从文件中读取这个股票列表在当日的价格
        price = UTILS.read_group_price(sA, priceType, dA, force_read_file=True)
        priceDict[dA] = price
        print(f"{CM.timestr()}#{i + 1}/{daysLen}-{dA}...")
    return priceDict


def make_no_cash_weight(w):
    totalW = sum(w.values())
    totalW -= w['CASH']
    nw = {}
    for c, v in w.items():
        if c == 'CASH':
            continue
        nw[c] = v / totalW
    return nw


def cal_weight_change(wA, wB):
    """
    计算两个权重图之间的股票换手率
    """

    allStock = list(set(wA.keys()) | set(wB.keys()))
    wc = 0.
    for code in allStock:
        if code == 'CASH':
            continue
        if wA.__contains__(code):
            ca = wA[code]
        else:
            ca = 0.
        if wB.__contains__(code):
            cb = wB[code]
        else:
            cb = 0.
        c = abs(ca - cb)
        wc += c
    return wc


def up_or_down_stop(tradeDay, pClose, pOpen, pType, updown):
    """
    给定前收,今开,类型,判断是否涨跌停
    """

    rate_limit_t1 = 0.1  # 主板股票的涨跌幅限制 10p (也是默认值)
    chuangyeban = 0.2  # 创业板默认为 0.2
    if tradeDay < '20200824':
        chuangyeban = 0.1  # 但是在20200824之前创业板是0.1
    pType = int(pType)
    if pType == 2:
        rate_limit_t1 = chuangyeban
    elif pType == 3:
        rate_limit_t1 = 0.3  # 北交所股票的涨跌幅限制 30p
    elif pType == 4:
        rate_limit_t1 = 0.2  # 科创板股票的涨跌幅限制 20p
    elif pType == 5:
        rate_limit_t1 = 0.05  # ST股票的涨跌幅限制 5p

    if updown == 'UP':
        # 判断是否为涨停 # t1交易时刻价格相对t0时刻收盘价的涨跌幅超过涨跌幅限制的股票, 则认为是涨停
        rate_trade = (pOpen - pClose + 0.01) / pClose
        return rate_trade >= rate_limit_t1
    else:
        # 判断是否为跌停
        rate_limit_t1 = -1. * rate_limit_t1  # 取相反数变成下跌幅限制
        rate_trade = (pOpen - pClose - 0.01) / pClose  # [B,M]
        return rate_trade <= rate_limit_t1


def check_upstop_earn_day(dA, wA, dB, priceInfo):
    """
    查找持仓拿到涨停收益的情形
    priceInfo 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    """

    codeS = list(wA.keys())

    datafiels = list(UTILS.g_field_index.keys())
    piOpen = CM.find_str_in_list(datafiels, 'open')
    piClose = CM.find_str_in_list(datafiels, 'close')
    piLimitType = CM.find_str_in_list(datafiels, 'limit_mark')

    # 遍历所有股票,计算总价值
    for code in codeS:
        if code == 'CASH' or code == 'ETF':
            continue
        vA = wA.get(code, 0)
        if vA <= 0.00001:
            continue

        codeInfoA = priceInfo[code][dA]
        codeInfoB = priceInfo[code][dB]

        pClose = codeInfoA[piClose]
        pOpen = codeInfoB[piOpen]
        pType = codeInfoB[piLimitType]
        if up_or_down_stop(dB, pClose, pOpen, pType, 'UP'):
            pr = (pOpen - pClose) / pClose
            print(f"{code}-{dA}-w{vA:.4f}-close_at{pClose:.2f}-{dB}open_at{pOpen}-R{pr * 100:.2f}%")


def check_up_down_stop_day(dA, wA, dB, wB, priceInfo):
    """
    检查是否存在异常加减仓
    priceInfo 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    """

    codeS = list(set(wA.keys()) & set(wB.keys()))

    datafiels = list(UTILS.g_field_index.keys())
    piOpen = CM.find_str_in_list(datafiels, 'open')
    piClose = CM.find_str_in_list(datafiels, 'close')
    piLimitType = CM.find_str_in_list(datafiels, 'limit_mark')

    # 遍历所有股票,计算总价值
    for code in codeS:
        if code == 'CASH' or code == 'ETF':
            continue

        vA = wA.get(code, 0)
        vB = wB.get(code, 0)

        codeInfoA = priceInfo[code][dA]
        codeInfoB = priceInfo[code][dB]
        if vA - vB > 0.00001:
            # 有减仓行为
            pClose = codeInfoA[piClose]
            pOpen = codeInfoB[piOpen]
            pType = codeInfoB[piLimitType]
            if up_or_down_stop(dB, pClose, pOpen, pType, 'DOWN'):
                print(f"{code}-{dA}close at {pClose:.2f}-{dB}open at {pOpen}-ERRORSUB-{vA:.4f}-{vB:.4f}")
        elif vB - vA > 0.00001:
            # 有加仓行为
            pClose = codeInfoA[piClose]
            pOpen = codeInfoB[piOpen]
            pType = codeInfoB[piLimitType]
            if up_or_down_stop(dB, pClose, pOpen, pType, 'UP'):
                print(f"{code}-{dA}close at {pClose:.2f}-{dB}open at {pOpen}-ERRORADD-{vA:.4f}-{vB:.4f}")


def cal_trade_win_lost(dA, wA, dB, wB, priceInfo, TC, etf=None):
    """
    计算前后两个交易日之间的交易结果
    dA为交易日期(今日),wA为今日权重...
    dB为次日, wB为今日权重,
    priceInfo为价格字典
    etf如不为None,则表示它是一个ETF基数(指数)数据字典,当权重图中出现股票名称为ETF时,用这个数据来计算其价值
    返回两个字典 win[key=stockcode,value=winvalue], lost[key=stockcode,value=lostvalue], 总盈亏
    计算dA所持有的股票以dB日的目标价格计算的相对价值(并扣除从wA变为wB时的交易成本)
    """

    if priceInfo.__contains__(dA):
        priceA = priceInfo[dA]  # 今日价格字典
    else:
        priceA = None
    if priceInfo.__contains__(dB):
        priceB = priceInfo[dB]  # 明日价格字典
    else:
        priceB = None

    win = {}  # 盈利信息
    lost = {}  # 亏损信息

    # 遍历今日所有股票,计算总价值
    totalValue = 0.
    for codeA, weightA in wA.items():
        if codeA == 'CASH':
            totalValue += weightA
            continue
        # 今日持仓的一只股票
        if codeA == 'ETF':
            todayPrice = etf[dA]
            nextPrice = etf[dB]
        else:
            todayPrice = priceA[codeA]
            nextPrice = priceB[codeA]
        if todayPrice < 1e-5 or nextPrice < 1e-5:
            # 有任何一天的价格异常
            pc = 1.
        else:
            pc = nextPrice / todayPrice
        if pc >= (1.0 + 1e-8):
            win[codeA] = (pc - 1.) * weightA  # 记录胜出的股票及胜出的收益
        else:
            lost[codeA] = (1. - pc) * weightA  # 记录输掉的股票及损失
        code_value = pc * weightA  # 这只股票的相对价值
        totalValue += code_value  # 累计收益

    # 计算换手率
    weight_change = cal_weight_change(wA, wB)
    cost = weight_change * TC

    return win, lost, totalValue * (1. - cost)


def plot_imge_2line(args, title, tdays, data):
    assert len(data) == 2, f"数据序列必须为2"
    keys = list(data.keys())
    # 示例数据：序列A（0-100），序列B（-5-5）
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]
    y1 = data[keys[0]]
    y2 = data[keys[1]]

    # 创建图形和第一个纵坐标轴
    fig, ax1 = plt.subplots(figsize=(32, 16))

    # 绘制第一个数据序列（左侧Y轴）
    color1 = 'tab:red'
    ax1.set_xlabel('时间（单位）')
    ax1.set_ylabel(keys[0], color=color1, fontsize=12)
    line1 = ax1.plot(x, y1, color=color1, linewidth=2, label=keys[0])
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(y1), max(y1))  # 明确设置左侧Y轴范围

    # 创建第二个纵坐标轴（共享同一横轴）
    ax2 = ax1.twinx()

    # 绘制第二个数据序列（右侧Y轴）
    color2 = 'tab:blue'
    ax2.set_ylabel(keys[1], color=color2, fontsize=12)
    line2 = ax2.plot(x, y2, color=color2, linestyle='--', linewidth=2, label=keys[1])
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(min(y2), max(y2))  # 明确设置右侧Y轴范围

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 设置日期格式
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 设置刻度间隔，避免过于密集
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签以避免重叠

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 添加标题并展示
    plt.title(f"{title[0]}-{args.mark}")
    plt.tight_layout()
    # plt.show()

    # 保存图片到文件
    imgname = f'{args.mark}_{title[0]}_{tdays[0]}_{tdays[-1]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_image(args, title, tdays, data):
    """
    title[]  t0为当前这个图片展示的主要信息,它也会是文件名的构成部分
             t1为横坐标名称
             t2为纵坐标名称
    tdays 作为横坐标的日期序列
    data  数据序列字典[key=name, value=[]]
    """

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # kksize = 8
    # if len(tdays) < 50:
    #     kksize = 16
    # plt.rcParams['xtick.labelsize'] = kksize  # 设置所有图表x轴刻度标签的默认大小
    # plt.rcParams['ytick.labelsize'] = kksize  # 设置所有图表y轴刻度标签的默认大小

    # 1. 将日期字符串转换为datetime对象
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]  # 请根据实际格式修改日期解析格式

    # 2. 创建图形并绘制折线
    MK = ['.', ',', 'o', 'v', '^', '<', '>']
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    LS = ['-', '--', '-.', ':']
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    plt.figure(figsize=(32, 16))  # 设置图片大小，根据需要调整
    for dn, dv in data.items():
        p_mk = MK[mi]
        mi = (mi + 1) % len(MK)
        p_line = LS[ml]
        ml = (ml + 1) % len(LS)
        p_color = CO[mc]
        mc = (mc + 1) % len(CO)
        plt.plot(x, dv, label=dn, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)  # 绘制数据b

    # 3. 设置横坐标日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 设置日期格式
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 设置刻度间隔，避免过于密集
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签以避免重叠

    # 4. 添加标题和标签
    plt.title(f'{title[0]}-{args.mark}', fontsize=48)
    plt.xlabel(title[1], fontsize=24)
    plt.ylabel(title[2], fontsize=24)
    plt.legend()  # 显示图例

    # 5. 保存图片到文件
    imgname = f'{args.mark}_{title[0]}_{tdays[0]}_{tdays[-1]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_image_rect(args, title, tdays, weights):
    """
    title[]  t0为当前这个图片展示的主要信息,它也会是文件名的构成部分
             t1为横坐标名称
             t2为纵坐标名称
    tdays 作为横坐标的序列(字符串,并不要求它是日期)
    data  数据序列字典[key=name, value=[]]
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # kksize = 10
    # if len(tdays)<50:
    #     kksize = 16
    # plt.rcParams['xtick.labelsize'] = kksize  # 设置所有图表x轴刻度标签的默认大小
    # plt.rcParams['ytick.labelsize'] = kksize  # 设置所有图表y轴刻度标签的默认大小

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(32, 16))
    bars = ax.bar(tdays, weights, color=plt.cm.tab20c(np.arange(len(tdays))))
    # 设置x轴刻度标签的字体大小
    ax.tick_params(axis='x', labelsize=24)  # 设置x轴刻度标签的字体大小为12

    # 设置图表标题和轴标签
    plt.title(f'{title[0]}-{args.mark}', fontsize=48)
    plt.xlabel(title[1], fontsize=32)
    plt.ylabel(title[2], fontsize=32)

    # 在每个柱子上显示权重值
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{weight:.2f}',
                ha='center', va='bottom', fontsize=32)

    # 旋转x轴标签以便更好地显示
    plt.xticks(rotation=45)

    # 调整布局并保存图片
    plt.tight_layout()

    imgname = f'{args.mark}_{title[0]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_image_rect_multi(args, title, tdays, data_dict):
    # tdays为横坐标序列,  data_dict{key=子序列名,value=[子序列]}
    # 多个数据序列（例如：不同时间段的权重数据）
    series_labels = list(data_dict.keys())
    data_series = list(data_dict.values())

    # 调用函数绘制多序列柱状图
    plot_multi_series_bar_chart(
        tdays=tdays,
        data_series=data_series,
        series_labels=series_labels,
        title=f'{title[0]}',
        xlabel=f'{title[1]}',
        ylabel=f'{title[2]}',
    )

    # 保存图片
    imgname = f'{args.mark}_{title[0]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_multi_series_bar_chart(tdays, data_series, series_labels, title, xlabel, ylabel, figsize=(32, 16)):
    """
    绘制多序列柱状图

    参数:
    tdays: 横坐标标签（类别）
    data_series: 包含多个数据序列的列表，每个序列对应一个数据系列
    series_labels: 每个数据系列的标签（用于图例）
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    figsize: 图表尺寸
    """

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=figsize)

    n_series = len(data_series)  # 数据系列数量
    n_categories = len(tdays)  # 类别数量

    # 计算柱子的位置和宽度[1,2](@ref)
    x = np.arange(n_categories)  # 类别位置
    width = 0.8 / n_series  # 柱子宽度，根据系列数量动态调整

    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    # 为每个数据系列绘制柱子[1](@ref)
    bars = []
    for i, (data, label) in enumerate(zip(data_series, series_labels)):
        p_color = CO[mc]
        mc = (mc + 1) % len(CO)
        # 计算当前系列柱子的x轴位置（居中排列）[1](@ref)
        x_pos = x + width * (i - (n_series - 1) / 2)
        bar = ax.bar(x_pos, data, width, label=label, color=p_color)  # color=plt.cm.tab20c(i))  # 使用不同的颜色
        bars.append(bar)

        # 在每个柱子上显示数值[1](@ref)
        for j, (bar_obj, value) in enumerate(zip(bar, data)):
            height = bar_obj.get_height()
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10)

    # 设置图表标题和轴标签
    ax.set_title(f'{title}', fontsize=48)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    # 设置x轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(tdays, rotation=45)

    # 添加图例
    ax.legend(fontsize=20)

    # 调整布局
    plt.tight_layout()

    return fig, ax


def is_period_end_day_in_list(daylist, index, period):
    if period == 'day':
        return True
    DLen = len(daylist)
    if index < 0 or index >= DLen:
        return False
    if index == DLen - 1:
        return True
    year, month, day, weekday = CM.parse_date(daylist[index])
    y2, m2, d2, w2 = CM.parse_date(daylist[index + 1])
    if period == 'month':
        return m2 != month  # day(i)与day(i+1)不是同一个月了,表明i是月末

    if period == 'week':
        if year != y2:
            return True

    di = datetime.strptime(daylist[index], "%Y%m%d")
    wi = int(di.strftime('%V'))

    dip1 = datetime.strptime(daylist[index + 1], "%Y%m%d")
    wip1 = int(dip1.strftime('%V'))

    return wi != wip1


def make_csv_week(args, tdays, weights, model_value, market_value):
    """
        market_value = {
        '中证1000': market_value_zz,
        '国证2000': market_value_gz,
        '全A': market_value_qa,
    }
    """

    imgname = f'{args.mark}_每周净值数据.csv'
    imgname = os.path.join(args.out_path, imgname)

    DLen = len(tdays)

    # 构造出模型的每周累计净值
    vm = 1.
    mdv = []  # 模型在每个期末时的累计净值
    cash = []
    mkt = {}
    outDays = []
    for k, _ in market_value.items():
        mkt[k] = {'CV': 1., 'VL': []}

    cc = []
    for i in range(DLen):
        day = tdays[i]
        vm = vm * model_value[day]  # 模型截止到今天的累计净值
        w = weights[day]
        cc.append(w['CASH'])
        for k in list(mkt.keys()):
            mkt[k]['CV'] = mkt[k]['CV'] * market_value[k][day]
        if is_period_end_day_in_list(tdays, i, 'week'):
            mdv.append(vm)
            for k in list(mkt.keys()):
                mkt[k]['VL'].append(mkt[k]['CV'])
            outDays.append(day)
            cs = sum(cc) / len(cc)
            cash.append(cs)
            cc = []

    f = open(imgname, 'w', encoding='utf-8-sig')
    print(f"日期(周末),现金项(周内日均),模型净值,{','.join(list(mkt.keys()))}", file=f)
    DLen = len(outDays)
    for i in range(DLen):
        print(f"{outDays[i]},{cash[i]:.4f},{mdv[i]:.4f}", end='', file=f)
        for k in list(mkt.keys()):
            print(f",{mkt[k]['VL'][i]:.4f}", end='', file=f)
        print(f"", file=f)
    f.close()

    make_down_week_info(model_value, market_value['全A'], '全A', top=0.2)
    make_down_week_info(model_value, market_value['中证1000'], '中证1000', top=0.2)


def make_down_week_info(model_value, market_value, market_name, top=0.1):
    # 指数重大回撤时本模型的表现(按周/取前10%)...
    model_change, wDayList = make_seris_value(model_value, vtype='v', period='week')
    qa_change, _ = make_seris_value(market_value, vtype='v', period='week')

    DL = len(wDayList)
    m_c_d = {}  # key=week_last_day, value=week_model_value
    for i in range(DL):
        wld = wDayList[i][1]
        m_c_d[wld] = model_change[i]
    qa_c_d = {}  # ke=week_last_day, value=week_market_value
    for i in range(DL):
        wld = wDayList[i][1]
        qa_c_d[wld] = qa_change[i]

    sorted_qa = sorted(qa_c_d.items(), key=lambda x: x[1], reverse=False)
    kL = int(DL * top)
    topK = []
    for i in range(kL):
        wld = sorted_qa[i][0]
        topK.append(wld)
    topK = sorted(topK)

    imgname = f'{args.mark}_{market_name}指数下行时模型表现.csv'
    imgname = os.path.join(args.out_path, imgname)
    f = open(imgname, 'w', encoding='utf-8-sig')
    print(f"日期(周末),{market_name}指数价值,模型价值,W/L", file=f)
    w = 0
    for x in topK:
        if m_c_d[x] >= qa_c_d[x]:
            w += 1
            wl = '是'
        else:
            wl = '否'
        print(f"{x},{qa_c_d[x]:.4f},{m_c_d[x]:.4f},{wl}", file=f)
    print(f"胜率,,,{w / kL * 100:.2f}%", file=f)
    f.close()


def function01(args, tdays, model_value, market_value_dict, period='day'):
    # 将净值与大盘绘制到同一图内
    # model_value是每天相对前一天的相对价值, market_value也是

    # 构造出模型的净值序列
    v = 1.
    mdv = []
    for x in model_value.values():
        v = v * x
        mdv.append(v)
    # print("MODEL:",mdv)

    # 构造出大盘的净值序列---对每种指标都生成一个序列
    fdv = {}
    for mz, mzvlist in market_value_dict.items():
        v = 1.
        mvlist = []
        for x in list(mzvlist.values()):
            v = v * x
            mvlist.append(v)
        fdv[mz] = mvlist

    DLen = len(tdays)
    if period == 'week' or period == 'month':
        # 按周/月取值,则需要从完整的日序列数据中挑选出所有周/月末数据项(即每月的最后一个交易日)
        p_end_mdv = []
        # p_end_fdv = []
        p_end_day = []
        newMZ = {}
        for mz in list(market_value_dict.keys()):
            newMZ[mz] = []
        for i in range(DLen):
            if is_period_end_day_in_list(tdays, i, period):
                # 如果第i天刚好是一个周末/月末
                p_end_mdv.append(mdv[i])
                for mz in list(newMZ.keys()):
                    newMZ[mz].append(fdv[mz][i])
                p_end_day.append(tdays[i])
        mdv = p_end_mdv
        fdv = newMZ
        tdays = p_end_day

    # 生成图片
    titles = [f'整体性能-净值曲线-{period}', '日期', '净值']
    data_dict = {
        "基金净值": mdv,
    }
    data_dict.update(fdv)
    return plot_image(args, titles, tdays, data_dict)


def search_value_in_list(x, ex_type_seg):
    """
    在一个有序列表中查找x所在的位置, 即ex_type_seg中n个点把数轴分为n+1份
    返回x在这n+1个区间中的哪一个, 从0开始计数.
    """
    DL = len(ex_type_seg)
    for i in range(DL):
        if x < ex_type_seg[i]:
            return i
    return DL


def function02(args, tdays, model_value, market_index_name, market_value, period='week'):
    DLen = len(tdays)

    # 构造出模型的每周累计净值
    vm = 1.
    vf = 1.
    mdv = []  # 模型在每个期末时的累计净值(本周内的)
    fdv = []  # 大盘在每个期末时的累计净值(本周内的)
    ddv = []  # 期末的日期
    for i in range(DLen):
        if period == 'day':
            mdv.append(model_value[tdays[i]])
            fdv.append(market_value[tdays[i]])
            ddv.appen(tdays[i])
        else:
            vm = vm * model_value[tdays[i]]
            vf = vf * market_value[tdays[i]]
            if is_period_end_day_in_list(tdays, i, period):
                mdv.append(vm)
                fdv.append(vf)
                ddv.append(tdays[i])
                vm = 1.
                vf = 1.

    # 计算每期(日/周/月)的超额
    DLen = len(ddv)
    for i in range(DLen):
        mdv[i] = (mdv[i] - fdv[i]) / fdv[i] * 100.

    # 把每周的超额进行分箱,观察其出现的次数
    # 算法 1.求出最大/最小值, 2.将其均分为20份得到19个点位值
    ex_type = {}
    ex_name = []
    ex_type_seg = []
    ex_max = 10.  # max(mdv)
    ex_min = -10.  # min(mdv)
    tpN = 20
    segLength = (ex_max - ex_min) / tpN
    for i in range(tpN):
        segMin = ex_min + i * segLength
        segMax = ex_min + (i + 1) * segLength
        kn = f"[{segMin:.2f}%,{segMax:.2f}%)"
        ex_name.append(kn)
        ex_type[i] = 0
        if i > 0:
            ex_type_seg.append(segMax)  # 生成了tpN-1个点位值
    for x in mdv:
        xi = search_value_in_list(x, ex_type_seg)  # 查找x在这tpN个区间中的哪一个, 从0开始计数,最大为tpN-1
        ex_type[xi] += 1
    totalTypes = sum(ex_type.values())
    for i in range(len(ex_type)):
        ex_type[i] = (ex_type[i] / totalTypes) * 100

    img = plot_image(args, [f"整体性能-超额收益-{period}-{market_index_name}", '日期', '超额收益%'], ddv,
                     {'超额收益': mdv})
    img2 = plot_image_rect(args,
                           [f"整体性能-超额收益-{period}-{market_index_name}-{tpN}档分布", '超额区间', '次数占比%'],
                           ex_name,
                           list(ex_type.values()))
    return [img, img2]


def resort_dict(unsorted_dict, rerverse=False):
    """
    对于key-value字典进行排序
    """

    sw = {}
    sorted_w = sorted(unsorted_dict.items(), key=lambda x: x[1], reverse=rerverse)
    for i in range(len(sorted_w)):
        stock_code = sorted_w[i][0]
        stock_weight = sorted_w[i][1]
        sw[stock_code] = stock_weight
    return sw


def cal_bj_stock_weight(w):
    bjw = 0
    for k, v in w.items():
        if str(k).endswith('.BJ'):
            bjw += v
    return bjw


def function07(args, weights, indexName, index_value):
    """
    分析模型的现金持有占比与某个指数的对应关系(按周计算)
    """

    days = []
    cash_w = []
    index_v = []

    allDayList = list(weights.keys())

    pCash = []
    pIndex = 1.
    for di in range(len(allDayList)):
        day = allDayList[di]
        w = weights[day]
        if not index_value.__contains__(day):
            continue
        pCash.append(w['CASH'])
        pIndex *= index_value[day]
        if is_period_end_day_in_list(allDayList, di, 'week'):
            cash_w.append(sum(pCash) / len(pCash))
            index_v.append(pIndex)
            days.append(day)

            pCash = []
            pIndex = 1.
    titles = [f'避险分析-{days[0]}-{days[-1]}', '日期', f'周均现金占比与指数{indexName}对照']
    data_dict = {
        "现金占比": cash_w,
        "指数涨跌幅": index_v,
    }
    return plot_imge_2line(args, titles, days, data_dict)


def function0301(args, weights, market_value_bj):
    """
    对北交所股票权重占比进行分析
    """

    bj_days = []  # 整个交易期中,恰好与北交所日期区间重合的部分
    bj_weights = []  # 该日期的北交所权重占比
    bj_change = []  # 该日期的北证指数涨跌幅
    for day, w in weights.items():
        # 遍历每一天的投资权重图
        if not market_value_bj.__contains__(day):
            continue
        bj_days.append(day)
        bj_weights.append(cal_bj_stock_weight(w) * 100.)
        bj_change.append((market_value_bj[day] - 1) * 100)
    titles = [f'投资分析-北交所股票权重占比{bj_days[0]}-{bj_days[-1]}', '日期', '权重-指数涨跌幅']
    data_dict = {
        "北交所股票占比": bj_weights,
        "北证指数涨跌幅": bj_change,
    }
    imgname = f'{args.mark}_北交所权重占比.csv'
    imgname = os.path.join(args.out_path, imgname)
    f = open(imgname, "w", encoding='utf-8-sig')
    print(f"日期,权重", file=f)
    for d, w in zip(bj_days, bj_weights):
        print(f"{d},{w}", file=f)
    f.close()
    return plot_imge_2line(args, titles, bj_days, data_dict)


def make_tmv_string(cate):
    s = [f"小于{cate[0]}"]
    for i in range(len(cate) - 1):
        s.append(f"{cate[i]}~{cate[i + 1]}")
    s.append(f"大于{cate[-1]}")
    return s


def value_in_list(cate, v):
    CL = len(cate)
    for i in range(CL):
        if v <= cate[i]:
            return i
    return CL


def function03(args, weights, all_stock_info,market_info):
    """
    对投资标的进行归类分析
    """

    images = []

    cmv_cate = [25, 50, 200, 500, 1000]
    cmv_type = make_tmv_string(cmv_cate)
    cmv_weight = {k: 0 for k in range(len(cmv_type))}
    market_weight = {}  # 各板块总权重

    datafiels = list(UTILS.g_field_index.keys())
    mvid = CM.find_str_in_list(datafiels, 'total_mv')
    maid = CM.find_str_in_list(datafiels, 'gen_market')
    for day, w in weights.items():
        # 遍历每一天的投资权重图
        for stock, stock_weight in w.items():
            if stock == 'CASH':
                continue
            # 获取股票所在的行业和板块
            if all_stock_info.__contains__(stock):
                stockdata = all_stock_info[stock]
                sday = list(stockdata.keys())
                sday = sday[0]  # 最早的一天,不能取最后一天,因为最后一天可能已经退市,全字段都是0
                sdv = stockdata[sday]
                stock_cmv = float(sdv[mvid]) / 10000.  # 总市值
                MV = int(round(sdv[maid]))
            else:
                # raise ValueError(f"未能读取股票{stock}数据")
                print(f"未能读取股票{stock}数据")
                continue

            tmv_type_i = value_in_list(cmv_cate, stock_cmv)
            cmv_weight[tmv_type_i] += stock_weight

            if market_weight.__contains__(MV):
                market_weight[MV] = market_weight[MV] + stock_weight
            else:
                market_weight[MV] = stock_weight

    # 分板块看投资标的的分布
    insList = list(market_weight.keys())
    insList = sorted(insList)  # 从小到大排序
    insName = []
    insValue = []
    totalWeight = sum(market_weight.values())
    for x in insList:
        if market_info.__contains__(x):
            xname = market_info[x]
        else:
            xname = '其他'
        insName.append(xname)
        insValue.append(market_weight[x] / totalWeight * 100)
    # print("---", len(insName), insName)
    # print(len(insValue), insValue)
    # plot_image_rect(args, [f"投资分析-板块分布", '板块名称', '权重占比%'], insName, insValue)

    # 分按市值看投资标的的分布
    totalWeight = sum(cmv_weight.values())
    insValue = []
    for x, v in cmv_weight.items():
        insValue.append(v / totalWeight * 100)
    stockImg = plot_image_rect(args, [f"投资分析-总市值分布", '按市值分类', '权重占比%'], cmv_type, insValue)
    images.append(stockImg)

    return images


def make_net_win_lost(win_dict, lost_dict):
    stock_net_value = {}
    for day, data in win_dict.items():
        for stock, wl in data.items():
            if stock_net_value.__contains__(stock):
                stock_net_value[stock] += wl
            else:
                stock_net_value[stock] = wl
    for day, data in lost_dict.items():
        for stock, wl in data.items():
            if stock_net_value.__contains__(stock):
                stock_net_value[stock] -= wl
            else:
                stock_net_value[stock] = -wl
    wnet = {}
    lnet = {}
    for x, v in stock_net_value.items():
        if v >= 0:
            wnet[x] = v
        else:
            lnet[x] = -v
    return wnet, lnet



def make_period_avg_datalist(dayList, bV1, bV2, period):
    DL = len(dayList)
    V1 = []
    V2 = []
    ret_V1 = []
    ret_V2 = []
    ret_daylis = []
    for i in range(DL):
        day = dayList[i]
        V1.append(bV1[i])
        V2.append(bV2[i])
        if is_period_end_day_in_list(dayList, i, period):
            ret_daylis.append(day)
            ret_V1.append(sum(V1) / len(V1))
            ret_V2.append(sum(V2) / len(V2))
            V1.clear()
            V2.clear()
    return ret_daylis, ret_V1, ret_V2


def read_field_from_all(all_stock_info, stock, day, fieldName):
    """
    从缓存的全局数据中读取一项, 这里假定所有的参数都是合法有效的
    全局数据格式为{key=stock,value={key=day,value=[...]}}
    """

    if not all_stock_info.__contains__(stock):
        raise ValueError(f"全局缓存中不存在股票{stock}")
    stockData = all_stock_info[stock]
    if not stockData.__contains__(day):
        raise ValueError(f"全局缓存中股票{stock}数据中不存在日期{day}")
    stockData = stockData[day]  # 这里得到了一个valueList

    # 得到当下要读取的这个字段在整个valueList中的下标
    datafiels = list(UTILS.g_field_index.keys())
    pi = CM.find_str_in_list(datafiels, fieldName)
    return stockData[pi]


def make_combin_weight_day_onlySame(wA, wB, f):
    """
    只保留共同的股票
    """
    newWeight = {'CASH': 0.}
    totalStockWeight = 0.
    sa = list(set(wA.keys()) & set(wB.keys()))  # 共同的股票
    for s in sa:
        if s == 'CASH' or s == 'cash':
            continue
        sw = wA[s] * f + (wB[s] * (1. - f))
        totalStockWeight += sw
        newWeight[s] = sw

    if len(newWeight) == 1:
        # 没有共同股票
        newWeight['CASH'] = 1.
        return newWeight

    # 确定应该保留多少现金, 取二者的CASH均值作为两个模型对风险的共识
    cash_w = (wA['CASH'] + wB['CASH']) / 2.
    # 除了现金之外的所有权重都应该被分配, 因此需要将现有共同股票的权重等比例放大或者缩小
    r = (1. - cash_w) / totalStockWeight
    totalStockWeight = 0.
    for k, v in newWeight.items():
        if k == 'CASH':
            continue
        kv = v * r
        newWeight[k] = kv  # 每只股票乘上缩放系数
        totalStockWeight += kv  # 重新累计所有股票权重(为了最后重设现金权重,避免精度累积误差)
    cash_w = 1. - totalStockWeight
    if cash_w < 0:
        cash_w = 0.
    newWeight['CASH'] = cash_w
    return newWeight


def make_combin_weight_day(wA, wB, f):
    newWeight = {'CASH': 0.}
    totalStockWeight = 0.
    for k, v in wA.items():
        if k == 'CASH':
            continue
        sv = v * f
        newWeight[k] = sv
        totalStockWeight += sv

    for k, v in wB.items():
        if k == 'CASH':
            continue
        sv = v * (1. - f)
        if newWeight.__contains__(k):
            newWeight[k] += sv
        else:
            newWeight[k] = sv
        totalStockWeight += sv

    newWeight['CASH'] = 1. - totalStockWeight
    return newWeight


def make_combin_weight_with_ETF(wModel, f):
    newW = {}
    dayList = list(wModel.keys())
    for day in dayList:
        wd = make_combin_weight_day_with_ETF(wModel[day], f)
        newW[day] = wd
    return newW


def make_combin_weight_day_with_ETF(wModel, f):
    """
    将原有模型的权重图全部压缩到原来的1-f, 增加一个ETF股票权重为f
    """
    newW = {}
    for k, v in wModel.items():
        newW[k] = v * (1. - f)
    newW['ETF'] = f
    return newW


def smooth_weight(w, minw=0.002):
    """
    对权重进行平滑处理
    """

    newW = {'CASH': 0.}
    sc_w_total = 0.
    total_small = 0.
    findSmall = False
    findStock = False
    for k, v in w.items():
        if k == 'CASH':
            continue
        if v < minw:
            total_small += v
            findSmall = True
        else:
            sc_w_total += v
            findStock = True
            newW[k] = v  # 超过minw的股票,保留下来
    if not findSmall:
        return w
    if not findStock:
        newW['CASH'] = 1.
        return newW

    # 将保留下来的股票整体放大r倍
    r = (sc_w_total + total_small) / sc_w_total
    sc_w_total = 0
    for k, v in newW.items():
        if k == 'CASH':
            continue
        v = v * r
        newW[k] = v
        sc_w_total += v
    newW['CASH'] = 1. - sc_w_total
    return newW


def make_combin_weight(wA, wB, f, only_same=False, minw=0.002):
    newW = {}
    dayList = list(wA.keys())
    for day in dayList:
        if only_same:
            wd = make_combin_weight_day_onlySame(wA[day], wB[day], f)
        else:
            wd = make_combin_weight_day(wA[day], wB[day], f)
        # 20260305 ADD
        if minw < 1:
            wd = smooth_weight(wd, minw=minw)
        newW[day] = wd
    return newW


def make_price_info_from_totalinfo(targetPrice, all_stock_info):
    # 从全局数据中读取价格数据,全局数据格式为{key=stock,value={key=day,value=[...]}}
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = {}
    datafiels = list(UTILS.g_field_index.keys())
    piClose = CM.find_str_in_list(datafiels, 'close')
    piCloseHfg = CM.find_str_in_list(datafiels, 'close_hfq')
    if targetPrice == 'AVGHFQ':
        hfgseg = ['open_hfq', 'high_hfq', 'low_hfq', 'close_hfq']
        piA = [CM.find_str_in_list(datafiels, x) for x in hfgseg]
        for stock, data in all_stock_info.items():
            # data{key=date,value=[]}
            for day, dv in data.items():
                if not priceInfo.__contains__(day):
                    priceInfo[day] = {}
                val = [dv[pi] for pi in piA]
                priceInfo[day][stock] = sum(val) / len(val)  # 求出这几个字段的均值并返回
    else:
        pi = CM.find_str_in_list(datafiels, targetPrice)
        for stock, data in all_stock_info.items():
            # data{key=date,value=[]}
            for day, dv in data.items():
                if not priceInfo.__contains__(day):
                    priceInfo[day] = {}
                priceInfo[day][stock] = dv[pi]

                if dv[piCloseHfg] < dv[piClose]:
                    print(f"ERRDATA:{day}-{stock}-{dv[piClose]}-{dv[piCloseHfg]}")

    return priceInfo


def make_barra_factor(weights, baseIndexName):
    """
    生成本投资组合及业绩基准指数的每日因子暴露
    返回数据类型为二级字典 {key=factor,value={key=day,value=v}}
    """

    pmFactor = {}
    baseFactor = {}
    for day, w in weights.items():
        # 遍历所有的日期/权重
        # 生成当天的指标-因子暴露数据
        no_cash_weight = make_no_cash_weight(w)  # 将权重图归化为全仓权重
        dF = UTILS.get_exposure(day, no_cash_weight, [baseIndexName])

        pmf = dF['thisPM']  # 本投资组合的因子暴露值 {key=factorname,value=v}
        for f, v in pmf.items():
            if not pmFactor.__contains__(f):
                pmFactor[f] = {}
            pmFactor[f][day] = v

        pmf = dF[baseIndexName]  # 业绩基准指数的因子暴露值{key=factorname,value=v}
        for f, v in pmf.items():
            if not baseFactor.__contains__(f):
                baseFactor[f] = {}
            baseFactor[f][day] = v
    return pmFactor, baseFactor


def cal_factor_value(pmFactor, factorName):
    """
    计算一个投资组合(业绩基准)在所有投资日期上的平均暴露/标准差/极值比例
    :param pmFactor: 全部因子在所有日期上的暴露字典{key=factorname,value={key=day,value=v}}
    :param factorName: 本次需要计算的因子名称
    :return avg,std,expct
    """

    fdict = pmFactor[factorName]
    vlist = list(fdict.values())
    vlist_np = np.array(vlist)
    avg = float(np.mean(vlist_np))  # 平均值
    std = float(np.std(vlist_np))  # 标准差
    exp = 0
    for v in vlist:
        if v > (avg + std) or v < (avg - std):
            exp += 1
    exp = exp / len(vlist)  # 极端值的占比
    return avg, std, exp


def barra(weights, baseIndexName):
    """
    barra分析
    :param weights: 投资组合的每日权重
    :param baseIndexName: 作为业绩比较基准的指数名称
    """

    # baseIndexName 是下面字典中的一个(中文名称)
    # base_index = {'000001.SH': '上证指数', '399001.SZ': '深证综指', '000905.SH': '中证500',
    #               '932000.SH': '中证2000', '899050.BJ': '北证50', '399005.SZ': '中小板指',
    #               '399006.SZ': '创业板指', '930903.CSI': '中证全指', '000300.SH': '沪深300'}
    factor = {'size': '市值', 'beta': 'beta', 'momentum': '动量',
              'non_linear_size': '非线性市值', 'bp': '账面市值比', 'earnings_yield': '盈利能力',
              'growth': '成长', 'leverage': '杠杆', 'liquidity': '流动性', 'volatility': '残差波动率'}

    # 调用工具包中的方法,生成投资组合与业绩基准指标在各个因子上的暴露(每天)
    # 返回数据类型为二级字典 {key=factor,value={key=day,value=v}}
    pmFactor, baseFactor = make_barra_factor(weights, baseIndexName)

    # 对每个因子循环处理
    for fn in list(factor.keys()):
        # 计算因子暴露值的均值/标准差/极值占比(暴露超过±1标准差的天数比例)
        f_avg, f_std, f_expct = cal_factor_value(pmFactor, fn)
        b_avg, b_std, b_expct = cal_factor_value(baseFactor, fn)
        print(f"*****[{factor[fn]}]因子分析*****")
        print(f"科目,{baseIndexName},本投资组合")
        print(f"平均暴露,{b_avg:.4f},{f_avg:.4f}")
        print(f"标准差,{b_std:.4f},{f_std:.4f}")
        print(f"极值占比,{b_expct * 100:.2f}%,{f_expct * 100:.2f}%")
        print()

    # 组合收益 = 无风险收益 + ∑(因子暴露 × 因子收益) + 特异性收益

    # 方法B：精确计算每日因子贡献
    # 对每个交易日t
    # 因子贡献_kₜ = 组合对因子k的暴露_Exposureₖₜ × 因子k的日收益_fₖₜ
    # 特异性收益ₜ = 组合实际收益ₜ - ∑因子贡献_kₜ
    #
    # # 累计收益分解
    # 累计因子贡献_k = ∑(1 + 因子贡献_kₜ)
    # 的连乘积 - 1
    # 累计特异性收益 = ∑(1 + 特异性收益ₜ)
    # 的连乘积 - 1


if __name__ == '__main__':
    args = get_args()


    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights = UTILS.read_weigh_from_path(args.path, args.match, args.match_year)
    if len(weights) < 5:
        print(f"权重数据太少{len(weights)},不足以进行区间分析,exit")
        exit(0)
    print(f"{CM.timestr()}读取权重数据完成,天数{len(weights)},耗时{time() - ts:.2f}秒")
    # print(weights.keys())
    # 读取指数数据

    market_info = CM.load_index("market_info.json")
    market_info = reverse_dict_valueIsInt(market_info)

    # STEP2: 读取理论交易价格
    print(f"{CM.timestr()}读取所有交易日价格信息...")
    ts = time()
    all_stock_info = UTILS.read_all_stock_info(weights)
    # priceInfo = read_price(args.targetPrice, weights)
    priceInfo = make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # STEP2: 计算每日收益,这里会计算每日绝对收益(扣除成本)
    dayslist = list(weights.keys())
    daysLen = len(dayslist)

    win_dict = {}  # 每天盈利股票
    lost_dict = {}  # 每天带来亏损的股票
    model_value = {}  # 每天模型的相对价值(税后)--税后是指扣除了交易成本,下同
    market_value_zz = {}  # 大盘指数相对价值 zz1000
    market_value_gz = {}  # 大盘指数相对价值 gz2000
    market_value_bj = {}  # 北证50指数相对价值
    market_value_qa = {}  # 全A指数相对价值
    tdays = []
    for i in range(1, daysLen - 1):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i - 1]  # 前一交易日
        wA = weights[dA]  # 前一交易日持仓
        dB = dayslist[i]  # 今日
        tdays.append(dB)
        wB = weights[dB]  # 今日持仓
        # 计算从前一交易日到今日的相对盈亏, 具体到每一次股票(以前一日所有持仓股票的视角)
        # win, lost, v = cal_trade_win_lost(dA, wA, dB, wB, priceInfo, args.tradeCost)
        # win_dict[dB] = win
        # lost_dict[dB] = lost
        # model_value[dB] = v


    # 指标3: 投资标的分析
    img = function03(args, weights, all_stock_info, market_info)

