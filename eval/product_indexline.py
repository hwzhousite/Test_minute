"""
尝试建立更多维度的评估指标
为了方便团队协作共享, 我们规定当前脚本的基本输入是一个模型回测时一段时间内连续N个交易日的持仓文件
文件为json格式
"""
import argparse
import copy
import datetime
import json
import os
from time import time
import utils as UTILS
import common_utils as CM
import utils_date as DT
import utils_barra as BA
import product_check_barra_style as BAT
import utils_image as IMG_TOOL


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
    parser.add_argument('--match_year', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--indexName', type=str, default='qa', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='eval', help='标识,输出某些文档时以此命名')
    parser.add_argument('--debug', type=int, default=0, help='是否输出')
    parser.add_argument('--pdf', type=int, default=0, help='是否输出PDF')
    parser.add_argument('--line', type=str, default='', help='要绘制的曲线类型')

    args = parser.parse_args()

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
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path)

    return args


def reverse_dict_valueIsInt(ad):
    reversed_dic = {}
    for k, v in ad.items():
        v = int(round(v))
        reversed_dic[v] = k
    return reversed_dic


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
            todayPrice = priceA.get(codeA, 0.)
            nextPrice = priceB.get(codeA, 0.)
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

    di = datetime.datetime.strptime(daylist[index], "%Y%m%d")
    wi = int(di.strftime('%V'))

    dip1 = datetime.datetime.strptime(daylist[index + 1], "%Y%m%d")
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


def make_cal_value(valueList):
    """
    将一个每日相对价值序列转换为一个累计价值序列
    """
    nv = []
    v = 1.
    for x in valueList:
        v = v * x
        nv.append(v)
    return nv


def draw_picture_01(args):
    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    # weights = UTILS.read_weigh_from_path(args.path, '', args.match_year)
    weights = UTILS.read_weight_from_index(indexName=args.indexName, match_year=args.match_year)

    if len(weights) < 5:
        print(f"权重数据太少{len(weights)},不足以进行区间分析,exit")
        exit(0)
    print(f"{CM.timestr()}读取权重数据完成,天数{len(weights)},耗时{time() - ts:.2f}秒")
    # print(weights.keys())
    # 读取指数数据

    zzindex = CM.load_index(f"{args.indexName}.json")
    assert zzindex is not None, f"读取指数数据失败{args.indexName}"

    # STEP2: 读取理论交易价格
    print(f"{CM.timestr()}读取所有交易日价格信息...")
    ts = time()
    all_stock_info = UTILS.read_all_stock_info(weights)
    priceInfo = make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # STEP2: 计算每日收益,这里会计算每日绝对收益(扣除成本)
    dayslist = list(weights.keys())
    daysLen = len(dayslist)

    # model_value = [1]  # 每天模型的相对价值(税后)--税后是指扣除了交易成本,下同
    # market_value = [1]  # 大盘指数相对价值 zz1000
    # tdays = [dayslist[0]]
    # for i in range(1, daysLen - 1):
    #     # 计算前后两个交易日期之间的相对收益
    #     dA = dayslist[i - 1]  # 前一交易日
    #     wA = weights[dA]  # 前一交易日持仓
    #     dB = dayslist[i]  # 今日
    #     tdays.append(dB)
    #     wB = weights[dB]  # 今日持仓
    #     # 计算从前一交易日到今日的相对盈亏, 具体到每一次股票(以前一日所有持仓股票的视角)
    #     win, lost, v = cal_trade_win_lost(dA, wA, dB, wB, priceInfo, args.tradeCost)
    #     model_value.append(v)
    #     market_value.append(zzindex.get(dB, 1) / zzindex.get(dA, 1))
    # tdays.append(dayslist[-1])
    # model_value.append(1)
    # market_value.append(1)

    model_value = []  # 每天模型的相对价值(税后)--税后是指扣除了交易成本,下同
    market_value = []  # 大盘指数相对价值 zz1000
    tdays = []
    for i in range(daysLen):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日日期
        wA = weights[dA]  # 今日持仓
        if i < daysLen-1:
            dB = dayslist[i+1]  # 明日日期
            wB = weights[dB] # 明日持仓
        else:
            # 已经是最后一天了
            dB = DT.get_real_next_trade_day(dA, True)
            wB = {'CASH':1.0}
        tdays.append(dA)
        # 计算从前一交易日到今日的相对盈亏, 具体到每一次股票(以前一日所有持仓股票的视角)
        win, lost, v = cal_trade_win_lost(dA, wA, dB, wB, priceInfo, args.tradeCost)
        model_value.append(v)
        market_value.append(zzindex.get(dB, 1) / zzindex.get(dA, 1))

    model_dv = make_cal_value(model_value)
    market_dv = make_cal_value(market_value)

    if args.debug>0:
        T = zip(tdays,model_value,model_dv)
        print("日期,当日收益,累计收益")
        for x,y,z in T:
            print(f"{x},{y:.4f},{z:.4f}")

    dataInfo = {
        'title': [f'模型净值{model_dv[-1]:.4f}', f"大盘{args.indexName}价值{market_dv[-1]:.4f}"],
        'days': [tdays, tdays],
        'value': [model_dv, market_dv],
        'split': False,
    }
    IMG_TOOL.draw_line_picture(args, f'模型净值与{args.indexName}净值', dataInfo)
    return tdays, market_value


def cut_date_area(dayList, valueList, match_year):
    if match_year == '' or match_year is None:
        return dayList, valueList
    yl = str(match_year).split(',')
    miny = min(yl)
    maxy = max(yl)
    if len(miny)==4:
        start = f"{miny}0101"
        end = f"{maxy}1231"
    elif len(miny)==6:
        start = f"{miny}01"
        end = f"{maxy}31"
    elif len(miny)==8:
        start = miny
        end = maxy
    else:
        raise ValueError(f"参数错误{match_year}")
    DL = []
    VL = []
    for i in range(len(dayList)):
        if start <= dayList[i] <= end:
            DL.append(dayList[i])
            VL.append(valueList[i])
    return DL, VL


def sort_data(title, all_ValList, t):
    """
    对所有因子数据进行排序,如果是暴露就按累计值的绝对值来排序,如果是收益就按最后一天的收益来排序
    """

    DL = len(title)
    sd = {}
    for i in range(DL):
        if t == 'pm_exp':
            expv = [abs(x) for x in all_ValList[i]]
            v = sum(expv)
            sd[i] = v
        else:
            valf = all_ValList[i]
            v = valf[-1]
            sd[i] = v
    sorted_w = sorted(sd.items(), key=lambda x: x[1], reverse=True)

    new_title = []
    new_value = []
    for i in range(len(sorted_w)):
        si = sorted_w[i][0]
        vi = sorted_w[i][1]
        xtitle = title[si]
        if t == 'pm_value':
            xtitle = f"{xtitle} {vi:.2f}"
        new_title.append(xtitle)
        new_value.append(all_ValList[si])
    return new_title, new_value


def output_info(expinfo, t, dateArea):
    d = {}
    for k, v in expinfo.items():
        if t == 'exp':
            vv = sum(v) / len(v)
            d[k] = abs(vv)
        else:
            vv = v[-1]
            d[k] = vv
    sorted_w = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print(f"{dateArea}-因子-{t}排名")
    for fi in range(len(sorted_w)):
        fc = sorted_w[fi][0]  # 因子名称
        if t == 'exp':
            vs = expinfo[fc]
            vs = sum(vs) / len(vs)
        else:
            vs = d[fc]
        print(f"#{fi + 1},{fc},{t},{vs:.4f}")
    print()


def draw_picture_02(args, factor, marketInfo=None):
    """
    绘制alpha曲线
    """

    # 读取barra数据
    ba_dict = BAT.read_index_badic(args)
    if factor == 'alpha':
        # 读取所有日期的收益(0附近的)
        dayList, valueList = BAT.make_day_value_list(ba_dict, 'pm_alpha')
        dayList, valueList = cut_date_area(dayList, valueList, args.match_year)
        calV = BAT.make_cal_value_list(valueList)
        dataInfo = {
            'title': ["日度alpha收益", f"累计alpha收益-{calV[-1]:.4f}"],
            'days': [dayList, dayList],
            'value': [valueList, calV],
            'split': True
        }
        IMG_TOOL.draw_line_picture(args, 'alpha', dataInfo)
        print(f"{args.mark}-alpha:{calV[-1]:.4f}")
    elif factor == 'ALL' or factor in BA.style_factor1:
        # 对所有因子进行暴露和收益分析,将暴露和累计收益绘制到同一幅图上
        if factor == 'ALL':
            f = copy.deepcopy(BA.style_factor1)
        else:
            f = [factor]

        expinfo = {}
        valinfo = {}
        dateArea =''
        for fc in f:
            # 准备数据--exp
            dayList, valueList_exp = BAT.make_day_value_list(ba_dict, 'pm_exp', fc)
            dayList_exp, valueList_exp = cut_date_area(dayList, valueList_exp, args.match_year)
            avg_exp = sum(valueList_exp) / len(valueList_exp)
            # 准备数据--value
            dayList, valueList_val = BAT.make_day_value_list(ba_dict, 'pm_value', fc)
            dayList_val, valueList_val = cut_date_area(dayList, valueList_val, args.match_year)
            dateArea = f"{args.mark}-{dayList_val[0]}~{dayList_val[-1]}"

            assert(len(dayList_exp)==len(dayList_exp)),f"数据长度异常"

            # 因子收益每天都是非常小的数,这里转换为累计价值
            valueList_val = BAT.make_cal_value_list(valueList_val)
            dataInfo = {
                'title': [f"{fc}日度暴露-avg:{avg_exp:.2f}", f"{fc}累计收益:{valueList_val[-1]:.2f}"],
                'days': [dayList_exp, dayList_val],
                'value': [valueList_exp, valueList_val],
                'split': True,
            }
            expinfo[fc] = valueList_exp
            valinfo[fc] = valueList_val
            if len(valueList_exp)!=len(valueList_val):
                print("eeeee-------------1")
                continue
            if len(valueList_exp)!=len(dayList_exp) or len(valueList_exp)!=len(dayList_val):
                print("eeeee-------------2")
                continue
            IMG_TOOL.draw_line_picture(args, f'{fc}因子的日度暴露和累计收益', dataInfo, marketInfor=marketInfo)
        output_info(valinfo, 'value', dateArea)
        output_info(expinfo, 'exp', dateArea)

    else:
        raise ValueError(f"参数错误{factor}")


if __name__ == '__main__':
    args = get_args()

    # 绘制alpha曲线,日均和累计
    draw_picture_02(args, 'alpha')

    # 绘制净值曲线,将净值与大盘绘制到同一张曲线图上
    tdays, market_value = draw_picture_01(args)
    market_value = make_cal_value(market_value)

    # 绘制所有因子暴露和收益曲线
    marketInfo = {'pos': 0, 'name': f'{args.indexName}:{market_value[-1]:.2f}', 'value': market_value}
    draw_picture_02(args, 'ALL', marketInfo=marketInfo)
