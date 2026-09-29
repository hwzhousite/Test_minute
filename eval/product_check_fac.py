import argparse
import datetime
import json
import os
import utils as UTILS
import common_utils as CM
import utils_date as DT
import pandas as pd


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
    parser.add_argument('--indexName', type=str, default='zzqa', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='eval', help='标识,输出某些文档时以此命名')
    parser.add_argument('--indexfile', type=str, default='', help='区间索引文件csv路径')
    parser.add_argument('--SI', type=int, default=2, help='起始日期在哪一列')
    parser.add_argument('--EI', type=int, default=3, help='结束日期在哪一列')
    parser.add_argument('--lastDate', type=str, default='20260518', help='最大日期')

    with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
        DT.g_calendar = json.load(f)

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
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path)

    return args


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


def make_av_ev(model_value, market_value):
    aV = {}
    exV = {}
    dayList = list(model_value.keys())
    avv = 1.
    evv = 1.
    for d in dayList:
        mv = model_value[d]
        indexv = market_value[d]

        avv = avv * mv  # 模型的绝对净值, 就累乘上当天的模型相对收益
        evv = evv * indexv  # 大盘净值
        aV[d] = avv
        exV[d] = evv
    return aV, exV


def get_model_return(json_path, start_time, end_time, indexName):
    my = f"{start_time},{end_time}"
    weights = UTILS.read_weigh_from_path(json_path, match_year=my)
    if weights is None or len(weights) <= 0:
        raise ValueError(f"{json_path}没有找到数据--{my}")

    all_stock_info = UTILS.read_all_stock_info(weights)
    priceInfo = make_price_info_from_totalinfo('close_hfq', all_stock_info)
    dayslist = list(weights.keys())
    dayslist = sorted(dayslist)
    daysLen = len(dayslist)

    zzindex = CM.load_index(f"{indexName}.json")
    assert zzindex is not None, f"读取指数数据失败{indexName}"

    # STEP2: 计算每日收益,这里会计算每日绝对收益(扣除成本)
    # 计算模型每日相对价值与指数每日相对价值,所有值在1附近
    base_model_value = {}  # 每天模型的相对价值(税后)--税后是指扣除了交易成本,下同
    base_market_value = {}  # 大盘指数相对价值 zz1000
    tdays = []
    for i in range(daysLen):
        # 计算前后两个交易日期之间的相对收益
        dA = dayslist[i]  # 今日日期
        wA = weights[dA]  # 今日持仓
        if i < daysLen - 1:
            dB = dayslist[i + 1]  # 明日日期
            wB = weights[dB]  # 明日持仓
        else:
            # 已经是最后一天了
            dB = DT.get_real_next_trade_day(dA, True)
            wB = {'CASH': 1.0}
        tdays.append(dA)
        # 计算从前一交易日到今日的相对盈亏, 具体到每一次股票(以前一日所有持仓股票的视角)
        win, lost, v = cal_trade_win_lost(dA, wA, dB, wB, priceInfo, 0.001)
        base_model_value[dA] = v
        base_market_value[dA] = zzindex.get(dB, 1) / zzindex.get(dA, 1)

    # 根据1附近的模型收益和指数基准收益来计算绝对的模型净值序列和相对净值序列
    baseValue, baseEx = make_av_ev(base_model_value, base_market_value)
    # 返回模型最终点的绝对净值和相对净值
    return baseValue[tdays[-1]], baseEx[tdays[-1]]


def read_csv_with_fallback(filepath, encodings=None, **kwargs):
    """
    尝试多种编码读取 CSV 文件，返回第一个成功的 DataFrame。

    参数
    ----------
    filepath : str
        文件路径。
    encodings : list, optional
        要尝试的编码列表，默认为 ['utf-8', 'gb2312', 'gbk', 'gb18030', 'latin1', 'iso-8859-1']。
    **kwargs : 其他参数
        直接传递给 pd.read_csv 的参数（如 sep, header, index_col 等）。

    返回
    -------
    pandas.DataFrame
        成功读取的数据框。

    抛出
    -------
    UnicodeDecodeError
        如果所有编码都失败，抛出最后一个编码的异常。
    """
    if encodings is None:
        encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'latin1', 'iso-8859-1']

    # last_exception = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            print(f"成功使用编码: {enc}")
            return df
        except (UnicodeDecodeError, LookupError) as e:
            # LookupError 可能由于编码名称不支持（但这里都是通用编码）
            last_exception = e
            continue
    raise ValueError(f"无法读取索引文件{filepath}")


def main(args):
    df = read_csv_with_fallback(args.indexfile)
    facInfo = list(df.values.tolist())
    # ['#1', '计算机收益', 20180118, 20180327, 2.29]...]
    print(
        f"异常因子,起始时间,结束时间,偏离度,模型终点价值,指数{args.indexName},算术超额,续期起始,续期结束,模型价值,指数价值,算术超额")
    SI = args.SI
    EI = args.EI
    for x in facInfo:
        # x是一个子列表, 5个字段, 2字段为因子名称, 后面两个是时间
        # x[1] 是一个yyyy-mm-dd格式的日期字符串,现需要转换成yyyymmdd
        startTime = str(x[SI])  # datetime.datetime.strptime(x[2], "%Y-%m-%d").strftime("%Y%m%d")
        endTime = str(x[EI])  # datetime.datetime.strptime(x[3], "%Y-%m-%d").strftime("%Y%m%d")
        if len(startTime) == 10:
            # 日期字符串长度为10,格式要转换一下
            startTime = datetime.datetime.strptime(startTime, "%Y-%m-%d").strftime("%Y%m%d")
            endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d").strftime("%Y%m%d")
        tdd = DT.count_trade_days(startTime, endTime)
        nextStart = DT.get_real_next_trade_day(endTime, 1)
        nextEnd = DT.get_next_trade_day(nextStart, tdd - 1)
        mv, ev = get_model_return(args.path, startTime, endTime, args.indexName)
        mv2 = ev2 = 0.
        if nextEnd <= args.lastDate:
            mv2, ev2 = get_model_return(args.path, nextStart, nextEnd, args.indexName)
        print(
            f"{x[1]},{x[2]},{x[3]},{x[4]:.2f},{mv:.4f},{ev:.4f},{(mv - ev) * 100:.2f}%,{nextStart},{nextEnd},{mv2:.4f},{ev2:.4f},{(mv2 - ev2) * 100:.2f}%")


"""
功能: 从某一个csv文件中读取信息,主要是获得一些时间区间(起始日期和结束日期)
CSV文件路径由参数指定, 由于csv格式不定, 所以重要的两个字段(日期)所在的列号要从参数中传入
然后从参数path指定的路径下读取所有的权重文件, 计算在这些小区间内模型的收益/同期的指数收益
以及后续连续相同天数内模型的收益情况
"""

if __name__ == '__main__':
    args = get_args()
    assert os.path.exists(args.indexfile), f"指定的索引文件{args.indexfile}不存在"
    main(args)
