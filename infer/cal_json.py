"""
根据json权重文件,生成统计报表, 完全不进行任何推理
python ./cal_json.py --json=/data/lhm_share/product_202603/day_0215MA/json/ --cfd=20180101 --cld=20260601 --indexName=qa
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import utils_date as DT
import utils as UTILS
import common_utils as CM
import cal_value as CalTool
import sys

sys.path.append("../")
import eval.utils_barra as BA
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--json', type=str, default='', help='权重数据json文件所在的路径')
    parser.add_argument('--path', type=str, default='', help='权重数据json文件所在的路径')
    parser.add_argument('--match', type=str, default='', help='权重数据json文件中必须匹配的子串')
    parser.add_argument('--mark', type=str, default='EVAL', help='mark -flag')
    parser.add_argument('--GPU', type=int, default=0, help='[0,1,2..]表示用哪一块GPU, 默认为0, -1表示不使用GPU')
    parser.add_argument('--MODELTYPE', type=str, default='A', help='[A,B,SUB01,SUPER]')
    parser.add_argument('--TD', type=int, default=1, help='多少天执行一次交易,默认为1,即每天都推理交交易')
    parser.add_argument('--skipLoss', type=float, default=1.0,
                        help='跳过开盘跌幅大于skipLoss的股票, 默认为1.0(即100%,不跳过)')
    parser.add_argument('--cfd', type=str, default='', help='交易日期起始点')
    parser.add_argument('--cld', type=str, default='', help='交易日期终止点')
    parser.add_argument('--DS', type=str, default='wd311', help='数据集名称[day104,day106]')
    parser.add_argument('--price_target', type=str, default='close_hfq', help='交易使用的价格字段,默认为空')
    # 复权处理: 对price_target要乘上复权因子后再使用, 如果本来就已经是复权价格了,则不可以再设置fuquan参数为1
    parser.add_argument('--fuquan', type=int, default=0,
                        help='是否要对价格进行复权操作,默认为0,设置为1则表示对price_target要做复权处理')
    parser.add_argument('--ssmodel_path', type=str, default='',
                        help='评分子模型')
    parser.add_argument('--model', type=str, default='', help='直接指定单一模型')
    parser.add_argument('--period', type=str, default='week', help='按周期分组')
    parser.add_argument('--G', type=int, default=300, help='股票小组的大小')
    parser.add_argument('--t', type=float, default=3, help='温度系数(只对mask_mf=8时有效，若虚的某一个专用模型)')

    parser.add_argument('--minw', type=float, default=0.0005, help='最小权重, 默认为0.001')
    parser.add_argument('--max_sw', type=float, default=0.05, help='单只股票的最大权重,默认为1即为不设限')
    # 如果设置了maxtop1为某个小于1的值,比如说0.05, 则在做权重平滑处理时,所有大于0.05的权重都会被削去多于0.05的部分,转移到现金上  ADD 20260304
    parser.add_argument('--amount5', type=float, default=1000,
                        help='过于5天的日均成交额下限,低于此下限的股票不进入推理池')  # ADD 20260304
    parser.add_argument('--time_step', type=int, default=80, help='额外向历史方向读取数据多少天(用于过滤)')
    parser.add_argument('--ext_day', type=int, default=0, help='额外向历史方向读取数据多少天(用于过滤)')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--withBJ', type=int, default=1, help='是否包含北交所股票,默认为1-包含')
    parser.add_argument('--BJMAX', type=int, default=0, help='北交所股票总权重之上限,默认为0-不设限')
    parser.add_argument('--min_total_mv', type=int, default=0, help='限制进入推理池的股票的最小市值,默认为0表示不限制')
    parser.add_argument('--max_total_mv', type=int, default=0, help='限制进入推理池的股票的最大市值,默认为0表示不限制')
    parser.add_argument('--mask_mf', type=int, default=0, help='特殊的数据处理模式[0,4,5,6,7,8]')
    parser.add_argument('--shuffle', type=int, default=1, help='排序参数')
    parser.add_argument('--split_year', type=int, default=1, help='是否分年,默认为1')
    parser.add_argument('--indexName', type=str, default='qa', help='确保{CM.G_ROOT_PATH}/indexData/下有同名的json文件')
    parser.add_argument('--stockpool', type=str, default='', help='指定推理股票池')
    parser.add_argument('--opv', type=int, default=0, help='')
    parser.add_argument('--opv_f', type=float, default=0.01, help='')
    parser.add_argument('--optimizer', type=int, default=0, help='')
    parser.add_argument('--op_fn', type=int, default=10, help='优化器要优化的因子个数')
    parser.add_argument('--op_to', type=float, default=10., help='维持原始权重的程度参数,值越大表示越靠近原始值')
    parser.add_argument('--sp', type=int, default=0, help='DEBUG用, 展示计算价值过程')
    parser.add_argument('--SUM', type=int, default=1, help='固定参数')

    args = parser.parse_args()
    if args.cld == '':
        args.cld = DT.timestr(1)
    args.device = 'cpu'
    if torch.cuda.is_available() and 0 <= args.GPU < torch.cuda.device_count():
        args.device = f'cuda:{args.GPU}'
    args.withBJ = args.withBJ == 1

    args.setZeroIndex = []
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
        args.g_csv_path = f"{CM.G_ROOT_PATH}/yy_data/qd/train/wd_311f/0723/pads"  # 每日增量更新
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8-sig') as f:
            DT.g_calendar = json.load(f)
        args.keep_fields = [i for i in range(310)]  # 全部字段序列,一共310个
    else:
        raise ValueError(f"数据集参数错误{args.DS}")

    if args.indexName == '':
        args.indexName = 'qa'
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path, args.indexName, set_pad_open_zero=False)

    if args.json !='':
        args.path = args.json
    elif args.path !='':
        args.json = args.path
    else:
        raise ValueError("要指定json权重文件所在路径")

    if args.cfd!='' and args.cld!='' and args.cfd>args.cld:
        args.cfd, args.cld = args.cld, args.cfd
    args.jsonInfo = check_path(args)
    dayList = sorted(list(args.jsonInfo.keys()))
    if len(dayList) == 0:
        raise ValueError(f"读取权重数据文件失败,请检查{args.json}")
    print(f"{args.json}路径下发现权重文件共{len(dayList)},{dayList[0]}~{dayList[-1]}")
    args.cfd = dayList[0]
    args.cld = dayList[-1]
    return args


def check_path(args):
    if not os.path.exists(args.json):
        return []
    jsonInfo = {}
    for fn in os.listdir(args.json):
        if not fn.endswith(".json"):
            continue
        if args.match != '' and (not args.match in fn):
            continue
        dt = fn[:8]
        if not DT.is_valid_date(dt):
            continue
        if args.cfd != '' and dt < args.cfd:
            continue
        if args.cld != '' and dt > args.cld:
            continue
        if jsonInfo.__contains__(dt):
            raise ValueError(f"路径{args.json}下存在重复日期的权重文件,请检查")
        fullName = os.path.join(args.json, fn)
        jsonInfo[dt] = fullName
    return jsonInfo


def add3months(date_str):
    date_str = str(date_str)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    month += 3
    if month > 12:
        year += 1
        month -= 12
    new_date = f"{year:04d}{month:02d}{day:02d}"
    return new_date


def getBJ(next_w: dict):
    """
    统计一个权重列表中北交所股票数量占比和权重占比
    """
    if next_w is None:
        return 0., 0.
    bjc = 0
    bjw = 0.
    for k, v in next_w.items():
        if "BJ" in k:
            bjc += 1
            bjw += v
    return bjc / len(next_w), bjw


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


def get_stock_average_MV(next_w: dict, tradeDay):
    # 获取当天投资的所有股票的平均流通市值(亿)
    klist = list(next_w.keys())
    if 'CASH' in klist:
        klist.remove('CASH')
    mv = UTILS.read_group_price(klist, 'mv', tradeDay)
    if mv is None or len(mv) == 0:
        return 0.

    # 改为加权流通市值
    total_w = 0.
    weighted_mv = 0.
    for stock_code, stock_mv in mv.items():
        smooth_mv = smooth_stock_mv(stock_mv)  # 对股票流通市值做平滑分档处理
        total_w += next_w[stock_code]  # 投资股票的权重求和
        smv = smooth_mv * next_w[stock_code]  # 该投资标的的市值*其权重
        weighted_mv += smv
    weighted_mv = weighted_mv * (1.0 / total_w)  # 将投资标的的加权市值进行归一化(即假定为满仓股票时对应市值)
    return round(weighted_mv, 2)


def getTop1StockWeight(next_w: dict):
    if next_w is None:
        return 0
    klist = list(next_w.keys())
    if len(klist) < 2:
        return 0
    return next_w[klist[1]]


def output_log(logfile, infoStr):
    print(infoStr)
    print(infoStr, file=logfile)


def period_result_exist(args, begin_date, end_date):
    """
    判断当前这个交易区间内的调仓文件是否全都已存在
    """
    tradeDayList = DT.trade_date_list(begin_date, end_date)
    tradeDayList = DT.get_rid_of_none_trading_days(tradeDayList)
    if len(tradeDayList) == 0:
        return True
    for tradeDay in tradeDayList:
        targetFile = UTILS.make_output_json_name(args, tradeDay)
        targetFile = os.path.join('./output', targetFile)
        if not os.path.exists(targetFile):
            return False
    return True


def period_test(args, begin_date, end_date, prev_weight=None, tradeCost=0.001):
    tradeDayList = DT.trade_date_list(begin_date, end_date)
    tradeDayList = DT.get_rid_of_none_trading_days(tradeDayList)
    if len(tradeDayList) == 0:
        print(f"交易日期起始点{begin_date}和终止点{end_date}之间没有交易日")
        return None

    # STEP1: 数据缓存
    firstPWisNone = prev_weight is None

    # STEP2: 对这个日期列表中的每一个交易日进行处理
    print(f"{DT.timestr()}模拟交易...")

    # STEP3: 计算每一日交易的累计收益
    print(f"{DT.timestr()}计算每日收益以及累计收益...")
    trade_days = []
    trade_value = []
    stock_count = []
    wins_count = []
    tr = []  # 换手率
    top1 = []  # 权重最大的股票的权重值
    total_value = 1.
    ts = time.time()
    if firstPWisNone:
        prev_w = {}
    else:
        prev_w = prev_weight
    bj_c = []
    bj_w = []
    cash_weight = []
    mv = []  # 每天投资的所有股票的流通市值的平均值
    all_weight = []
    for tradeDay in tradeDayList:
        targetFile = args.jsonInfo[tradeDay]
        if not os.path.exists(targetFile):
            raise ValueError(f"日期{tradeDay}的结果文件{targetFile}不存在")
        # tt = time.time()
        value, sc, sw, next_w = CalTool.cal_value_by_pmfile(targetFile, output=(args.sp==1), price_target=args.price_target,fuquan=args.fuquan==1)
        if value is None:
            print(f"日期{tradeDay}的结果文件{targetFile}计算收益失败, 跳过...")
        top1.append(getTop1StockWeight(next_w))
        mv.append(get_stock_average_MV(next_w, tradeDay))
        # print(f"----{tradeDay}---mv{mv[-1]}")
        cash_weight.append(next_w['CASH'])
        bjc, bjw = getBJ(next_w)
        bj_c.append(bjc)
        bj_w.append(bjw)
        trade_days.append(tradeDay)
        stepTr = UTILS.cal_tr(prev_w, next_w)
        tr.append(stepTr)  # 换手率(单向)
        trade_value.append(value * (1. - 2 * stepTr * tradeCost))  # 双向扣除交易成本
        stock_count.append(sc)
        wins_count.append(sw)
        all_weight.append(next_w)

        prev_w = next_w
        # print(f"日期{tradeDay}的收益为{value:.4f}, 耗时{time.time() - tt:.2f}秒")
    print(f"*****{DT.timestr()}计算成功{len(trade_days)}个交易日,总耗时{time.time() - ts:.2f}秒,结果如下:")
    bj_c = sum(bj_c) / len(bj_c)
    bj_w = sum(bj_w) / len(bj_w)
    scash_weight = sum(cash_weight) / len(cash_weight)
    mvcount = (len(mv) - sum(x < 0.001 for x in mv))  # 极小的数(其实就是0)不参与分母计数
    smv = sum(mv) / (mvcount if mvcount > 0 else 1)

    # STEP4: 输出结果
    print(
        f"交易日期,现金,股票数,日内胜出,日内胜率,当日收益,当日大盘{args.indexName},当日收益率,换手率,累计胜率(绝对),累计胜率(相对),累计价值,累计收益率")
    ### 当日收益: 其实是假定当天以目标价买入且次日以目标价进行评估的收益
    ### 相应的,作为对照的当日大盘: 其实是计算了次日收盘对比于今日收盘的收益
    total_days = 0
    win_days = 0
    win_to_market_days = 0
    min_value = 1.
    max_value = 1.
    vlist = []
    mlist = []
    for i in range(len(trade_days)):
        total_days += 1
        if trade_value[i] > 1:
            win_days += 1
        td = trade_days[i]
        # next_td = UTILS.get_real_next_trade_day(td)
        market_value = UTILS.cal_market_value(args, [td, td], indexName=args.indexName)
        mlist.append(market_value)
        if trade_value[i] > market_value:
            win_to_market_days += 1
        total_value *= trade_value[i]
        if total_value < min_value:
            min_value = total_value
        if total_value > max_value:
            max_value = total_value
        vlist.append(total_value)
        if stock_count[i] <= 0:
            rw = 0.
        else:
            rw = wins_count[i] / stock_count[i] * 100
        print(
            f"{trade_days[i]},{cash_weight[i]:.2f},{stock_count[i]},{wins_count[i]},{rw:.1f}%,{trade_value[i]:.4f},{market_value:.4f},{(trade_value[i] - 1) * 100:.2f}%,{tr[i]:.2f},{win_days / total_days * 100:.2f}%,{win_to_market_days / total_days * 100:.2f}%,{total_value:.4f},{(total_value - 1) * 100:.2f}%")
    ww = sum(stock_count)
    if ww == 0:
        ww = 1.
    print(f"平均,{sum(stock_count) / len(stock_count):.1f},"
          f"{sum(wins_count) / len(wins_count):.1f},"
          f"{sum(wins_count) / ww * 100:.1f}%,"
          f"{sum(trade_value) / len(trade_value):.4f},,,{sum(tr) / len(tr):.2f}")
    print(f"{args.md}")

    mdd = UTILS.max_drawdown(vlist)
    market_value = UTILS.cal_market_value(args, tradeDayList, indexName=args.indexName)
    ir = CM.get_ir(trade_value, mlist, period='day')  # 计算IR-年化信息比率
    total_ee = (total_value - market_value) / market_value
    # 输出总体信息
    print(
        f"BTF首日,末日,交易天数,绝对胜率,相对胜率,最大,最小,最大回撤,终点收益,大盘{args.indexName},超额,IR,BJS,BJW,CASH,CMV")
    print(
        f"BTI{tradeDayList[0]},{tradeDayList[-1]},{len(tradeDayList)},{win_days / total_days * 100:.2f}%,{win_to_market_days / total_days * 100:.2f}%,{max_value:.4f},{min_value:.4f},{mdd * 100:.2f}%,"
        f"{total_value:.4f},{market_value:.4f},{total_ee * 100:.3f}%,{ir:.4f},{bj_c:.2f},{bj_w:.2f},{scash_weight:.2f},{smv:.2f}")

    # 计算平均换手率, 忽略掉首日从全现金到第一个交易日之间的换仓
    # print(f"DEBUG--{firstPWisNone}--{tr}")
    if firstPWisNone and len(tr) > 1:
        tr = tr[1:]
    tr = sum(tr) / len(tr)
    # print(f"DEBUG--{tr}")

    # 返回结果: 交易日期起始点,交易日期终止点,交易天数,累计收益,大盘收益,超额, IR
    ret_info = {
        "start_date": tradeDayList[0],
        "end_date": tradeDayList[-1],
        "trade_days": len(tradeDayList),
        "total_value": total_value,
        "market_value": market_value,
        "win": total_value > market_value,
        "total_ee": total_ee,
        "ir": ir,
        "mdd": mdd,
        "stock_count": sum(stock_count) / len(stock_count),
        "tr": tr,
        "top1": sum(top1) / len(top1),
        "last_w": prev_w,
        "bj_c": bj_c,
        "bj_w": bj_w,
        "cash_weight": scash_weight,
        "mv": smv,
        "all_weight": all_weight,
        "trade_day_list": tradeDayList,
        "Day_market": mlist,
        "Day_model": trade_value,
    }
    return ret_info


def make_year_list(startDate, endDate):
    year_list = []
    ys = int(startDate[:4])
    ye = int(endDate[:4])
    if ys == ye:
        return [[startDate, endDate]]
    for y in range(ys, ye + 1):
        if y == ys:
            year_list.append([startDate, f"{ys}1231"])
        elif y == ye:
            year_list.append([f"{ye}0101", endDate])
        else:
            year_list.append([f"{y}0101", f"{y}1231"])
    return year_list


def year_test(args, startDate, endDate):
    """
    在指定的一年上做回测
    """

    args.keepWeightDays = args.TD
    # 将本年度的日期按照参数args.period所指定的类型再度切分成若干个子区间
    seg = DT.make_sub_segment_date(startDate, endDate, args.period)
    if len(seg) == 0:
        print(f"日期{startDate}到{endDate}之间按{args.period}分组结果为空")
        return None

    # 一次性将本年度的全部数据都读入缓存,包括向历史方向回退args.time_step个交易日

    if args.split_year > 0:
        # 切分年份, 则一次性将全年需要的数据读入缓存
        buffer_day0 = DT.get_real_next_trade_day(startDate, False)
        buffer_day0 = DT.get_next_trade_day(buffer_day0, -args.time_step - args.ext_day)
        buffer_day1 = DT.get_real_next_trade_day(endDate)
        UTILS.G_ALL_STOCK_DATA.init_buffer_data(args.DS,buffer_day0, buffer_day1)

    pd = []
    prev_w = None
    for s in seg:
        if args.split_year == 0:
            # 参数要求不切分年份, 则每次单独读取一个小区间的数据(因为总区间可能在时间跨度上非常长, 如果一次性全部读入,有可能
            # 会导致内存消耗过大/数据不全等问题
            buffer_day0 = DT.get_real_next_trade_day(s[0], False)
            buffer_day0 = DT.get_next_trade_day(buffer_day0, -args.time_step - args.ext_day)
            buffer_day1 = DT.get_real_next_trade_day(s[1])
            UTILS.G_ALL_STOCK_DATA.init_buffer_data(args.DS,buffer_day0, buffer_day1)

        r = period_test(args, s[0], s[1], prev_weight=prev_w, tradeCost=args.tradeCost)
        if r is not None:
            pd.append(r)
            prev_w = r["last_w"]
    if len(pd) == 0:
        print(f"日期{startDate}到{endDate}之间按{args.period}分组测试结果为空")
        return None

    # 以追加方式打开日志文件
    logfile = open("./log.csv", "a+", encoding="utf-8-sig")
    output_log(logfile, f"model:{args.md}---at {DT.timestr()}")
    output_log(logfile, f"日期{startDate}到{endDate}之间按{args.period}分组测试结果如下:")
    output_log(logfile,
               f"交易起点,交易终点,交易天数,区间终点收益,区间大盘{args.indexName},超额,日均持股,日均T1,日均TR,赢大盘")
    vlist = []
    mlist = []
    tvlist = [1.]
    total_da = ''
    total_db = ''
    gv = 1.
    ee = []
    victory = 0
    hold_stock = []
    top1 = []
    tr = []
    bj_c = []
    bj_w = []
    cash_weight = []
    mv = []
    all_weight = []

    Day_market = []
    Day_model = []
    all_trade_day_list = []

    for p in pd:
        ws = '是' if p['win'] else '否'
        output_log(logfile,
                   f"{p['start_date']},{p['end_date']},{p['trade_days']},{p['total_value']:.4f},{p['market_value']:.4f},{p['total_ee'] * 100:.2f}%,{p['stock_count']:.1f},{p['top1']:.2f},{p['tr']:.2f},{ws}")
        hold_stock.append(p["stock_count"])
        top1.append(p["top1"])
        tr.append(p["tr"])
        if total_da == '':
            total_da = p["start_date"]
        total_db = p["end_date"]
        total_da = min(total_da, p["start_date"])
        vlist.append(p["total_value"])
        mlist.append(p["market_value"])
        if p["win"]:
            victory += 1
        gv = float(tvlist[len(tvlist) - 1]) * p["total_value"]
        tvlist.append(gv)
        ee.append(p["total_ee"])
        bj_c.append(p["bj_c"])
        bj_w.append(p["bj_w"])
        cash_weight.append(p["cash_weight"])
        mv.append(p["mv"])
        all_trade_day_list.extend(p['trade_day_list'])
        all_weight.extend(p['all_weight'])
        Day_market.extend(p['Day_market'])
        Day_model.extend(p['Day_model'])

    # 新增加功能20260628 在每个年份的回测跑完之后,根据权重图数据,调用barra计算每日的因子暴露和因子收益
    alpha_year = alpha_IR = size_Exp = 0.
    year_ba = None
    if args.SUM == 1:
        print(f"{DT.timestr()}生成barra数据...")
        year_ba = make_barra_year(args, all_trade_day_list, all_weight)
        print(f"{DT.timestr()}生成barra数据...完成")
        alphaList = [x[-1] for x in list(year_ba.values())]  # alpha收益list(日度)
        alpha_year = sum(alphaList) / len(alphaList) * 252  # 年度alpha收益
        alpha_IR = CM.get_net_ir(alphaList, period='day')
        sizeI = BA.get_factor_index('size', 'exp')
        sizeList = [x[sizeI] for x in list(year_ba.values())]  # size暴露list(日度)
        size_Exp = sum(sizeList) / len(sizeList)

    bj_c = sum(bj_c) / len(bj_c)
    bj_w = sum(bj_w) / len(bj_w)
    scash_weight = sum(cash_weight) / len(cash_weight)

    mvcount = (len(mv) - sum(x < 0.001 for x in mv))  # 极小的数(其实就是0)不参与分母计数
    smv = sum(mv) / (mvcount if mvcount > 0 else 1)

    # smv = sum(mv) / len(mv)

    Pstd = np.std(np.array(vlist), ddof=1)
    Mstd = np.std(np.array(mlist), ddof=1)
    Estd = np.std(np.array(ee), ddof=1)
    Emae = CM.cal_ex_mae(ee)

    SR = CM.get_SR(vlist, period=args.period)  # SR
    ir = CM.get_ir(vlist, mlist, period=args.period)  # 计算IR-年化信息比率
    beta = CM.get_beta(vlist, mlist)  # 计算beta系数
    # LAP = CM.get_bigloss_and_return(vlist, mlist)
    offensive = CM.get_offensive_power(vlist, mlist)
    defensive = CM.get_defensive_power(vlist, mlist)
    downrisk = CM.get_down_risk(vlist, period=args.period, no_risk_year_p=0.02)
    mdd = UTILS.max_drawdown(tvlist)

    Calmar_Rate = CM.cal_calmar_rate(vlist, mdd=mdd)  # 计算卡马比率
    SOR_Rate = CM.cal_sor_rate(vlist)

    abs_win_rate, win_loss_rate = CM.cal_win_loss_rate(vlist)  # 绝对胜率与盈亏比
    # avg_hold_days = CM.cal_avg_hold_days(all_weight)

    ex_value_list = [1.]
    ex_value = 1.
    for i in range(len(vlist)):
        x = vlist[i] / mlist[i]
        ex_value = ex_value * x
        ex_value_list.append(ex_value)
    emdd = UTILS.max_drawdown(ex_value_list)

    gmv = UTILS.cal_market_value(args, [total_da, total_db], indexName=args.indexName)
    mee = sum(ee) / len(ee)
    hold_stock_m = sum(hold_stock) / len(hold_stock)
    top1_m = sum(top1) / len(top1)
    tr_m = sum(tr) / len(tr)
    v2 = victory / len(pd) * 100.
    output_log(logfile, f"model:{args.md}---at {DT.timestr()}")
    output_log(logfile, f"总测试区间:[{startDate}~{endDate}]-子区间类型:{args.period}-子区间个数:{len(pd)}")
    title = f"区间/{args.period},累计价值,大盘{args.indexName},总超额,平均超额,MAD,MAD-te,Pstd,Mstd,Estd,Emae,SR,IR年化,MDD,EMDD,下行风险,SOR,CALMAR,BETA,进攻能力,防守能力,日均持股,CASH,CMV-亿,日均T1,日均TR,BJ,相对胜率,绝对胜率,盈亏比,Alpha年化,AlphaIR,SizeExp"
    output_log(logfile, title)
    info = f"{total_da}~{total_db},{gv:.4f},{gmv:.4f},{(gv - gmv) / gmv * 100:.2f}%,{mee * 100:.2f}%"
    info = info + f",{CM.get_day_diff(Day_market, Day_model)}"
    info = info + f",{Pstd:.4f},{Mstd:.4f},{Estd:.4f},{Emae:.4f}"
    # info = info + f",{LAP[0]:.1f},{LAP[1]:.1f},{LAP[2]:.1f},{LAP[3]:.1f}"
    info = info + f",{SR:.4f},{ir:.4f},{mdd * 100:.2f}%,{emdd * 100:.2f}%,{downrisk:.4f}"
    info = info + f",{SOR_Rate:.4f},{Calmar_Rate:.4f}"
    info = info + f",{beta:.4f},{offensive:.4f},{defensive:.4f},{hold_stock_m:.1f}"  # ,{avg_hold_days:.1f}"
    info = info + f",{scash_weight:.2f},{smv:.2f},{top1_m:.4f},{tr_m:.2f},{bj_c:.2f}/{bj_w:.2f},{v2:.2f}%"
    info = info + f",{abs_win_rate * 100:.2f}%,{win_loss_rate:.2f}"
    info = info + f",{alpha_year * 100:.2f}%,{alpha_IR:.2f},{size_Exp:.2f}"
    output_log(logfile, info + "\n")

    logfile.close()

    ret = {
        "title": title,
        "info": info,
        "startDate": total_da,  # 首个交易日
        "endDate": total_db,  # 末交易日
        "vlist": vlist,  # 子区间模型收益(相对)
        "mlist": mlist,  # 子区间大盘收益(相对)
        "ee": ee,  # 子区间超额收益
        "hold_stock": hold_stock,  # 日均持股
        "top1": top1,  # 日均top1
        "tr": tr,  # 日均换手
        "victory": victory,  # 胜出数量
        "cash_weight": cash_weight,
        "mv": mv,
        "trade_day_list": all_trade_day_list,
        "all_weight": all_weight,
        "Day_market": Day_market,
        "Day_model": Day_model,
        "year_ba": year_ba,
    }
    return ret


def read_zz1000_exp():
    """
    从共享存储数据中读取zz1000的barra因子数据
    """
    bafile = '/data/indexData/zz1000_barra.json'
    # 读取barra数据字典
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    with open(bafile, 'r', encoding='utf-8-sig') as file:
        ba_dict = json.load(file)
    if ba_dict is None:
        print(f"读取数据{bafile}失败")
    xb = {}
    style_factor = ['size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price', 'leverage', 'growth',
                    'earnings_yield', 'beta', 'residual_volatility']
    for kday, kba in ba_dict.items():
        pm_exp = kba['pm_exp']
        vl = []
        for fn in style_factor:
            vl.append(float(pm_exp[fn]))
        xb[kday] = vl
    print(f"读取zz1000指数的barra-10风格因子暴露数据完成,总天数{len(xb)}")
    return xb

def make_barra_year(args, year_day_list, year_weight):
    assert (len(year_day_list) == len(year_weight)), f"交易日期长度{len(year_day_list)},权重图长度{len(year_weight)}"

    ba_dic = None
    bf = os.path.join(args.path, "pd_barra.json")
    if os.path.exists(bf):
        ba_dic = UTILS.read_pmfile(bf)
    else:
        print(f"{bf}----文件不存在,需要调用BA接口实时获取")

    year_ba = {}
    for day, weight in zip(year_day_list, year_weight):
        if ba_dic is not None and ba_dic.__contains__(day):
            # pm_exp = ba_dic[day]['pm_exp']
            # pm_value = ba_dic[day]['pm_value']
            # pm_alpha = ba_dic[day]['pm_alpha']
            vl = ba_dic[day]
        else:
            NDay = DT.get_real_next_trade_day(tradeDay=day, next_day=True)
            qa_exp, qa_value, qa_alpha, pm_exp, pm_value, pm_alpha = BA.get_barra(NDay, weight, keep_cash=True)
            vl = []
            vl.extend(list(pm_exp.values()))  # 42个因子暴露
            vl.extend(list(pm_value.values()))  # 42个因子收益
            vl.append(pm_alpha)  # alpha收益
        year_ba[day] = vl  # 总长度85列
    return year_ba

def make_summery_info(args, all_trade_day_list, Day_model, year_ba):
    # all_trade_day_list日期列表, Day_model是每日收益
    TD = len(all_trade_day_list)
    TM = len(Day_model)
    print(f"总交易日:{TD},模型日度相对收益列表长度:{TM}")
    assert TD == TM, "---请检查数据!!!---"
    fn = os.path.join(args.path,'pd_list.json')
    if os.path.exists(fn):
        print(f"{fn}--模型每日收益数据文件已存在,跳过!!!")
    else:
        nv = {}
        for day, val in zip(all_trade_day_list, Day_model):
            nv[day] = val
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(nv, f, indent=1)
        print(f"{fn}--模型每日收益数据文件已创建完成!!!")

    # 构造一个完成的barra字典
    fn = os.path.join(args.path,"pd_barra.json")
    if os.path.exists(fn):
        print(f"{fn}--每日因子暴露和收益数据文件已存在,跳过!!!")
    else:
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(year_ba, f, indent=1)
        print(f"{fn}--每日因子暴露和收益数据文件已创建完成!!!")

if __name__ == '__main__':
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU}"

    if not DT.is_valid_date(args.cfd):
        raise ValueError(f"交易日期起始点{args.cfd}格式不正确, 请使用YYYYMMDD格式")
    if not DT.is_valid_date(args.cld):
        raise ValueError(f"交易日期终止点{args.cld}格式不正确, 请使用YYYYMMDD格式")

    # print(f"----当前的回测运行参数:{args}")

    # STEP1: 加载模型
    args.md = args.mark

    # STEP3: 对日期进行分组(如需要)
    if args.split_year > 0:
        year_seg = make_year_list(args.cfd, args.cld)  # 将测试期按年划分
    else:
        year_seg = [[args.cfd, args.cld]]

    all_vlist = []
    all_mlist = []
    all_ee = []
    all_startDate = ''
    all_endDate = ''
    all_hold_stocks = []
    all_top1 = []
    all_tr = []
    all_cash_weigh = []
    all_mv = []
    victory = 0
    yc = 0
    title = ''
    info = []
    all_weight = []

    Day_market = []
    Day_model = []
    all_trade_day_list = []
    year_ba = {}

    for year in year_seg:
        st = time.time()
        ret = year_test(args, year[0], year[1])
        print(f"process {year},耗时:{time.time() - st:.2f}秒")
        if ret is None:
            continue
        if title == '':
            title = ret['title']
        info.append(ret['info'])
        all_vlist.extend(ret['vlist'])
        all_mlist.extend(ret['mlist'])
        all_ee.extend(ret['ee'])
        if all_startDate == '':
            all_startDate = ret['startDate']
        all_endDate = ret['endDate']
        all_hold_stocks.extend(ret['hold_stock'])
        all_top1.extend(ret['top1'])
        all_tr.extend(ret['tr'])
        all_cash_weigh.extend(ret['cash_weight'])
        all_mv.extend(ret['mv'])
        all_trade_day_list.extend(ret['trade_day_list'])
        all_weight.extend(ret['all_weight'])
        Day_market.extend(ret['Day_market'])
        Day_model.extend(ret['Day_model'])
        victory += ret['victory']
        year_ba.update(ret['year_ba'])
        yc += 1
    if args.SUM == 1:

        make_summery_info(args, all_trade_day_list, Day_model, year_ba)

        # 总的有效测试年度数大于1时,出一个汇总报告
        logfile = open("./log.csv", "a+", encoding="utf-8-sig")
        output_log(logfile,
                   f"*****汇总结果:{args.md}***交易价格{args.price_target}**成本{args.tradeCost}**{args.DS}**{DT.timestr()}")
        output_log(logfile, title)  # title
        for txt in info:  # 分年测试结果
            output_log(logfile, txt)
        output_log(logfile, f"日期{args.cfd}到{args.cld}之间按{args.period}分组测试汇总结果-子区间数{len(all_vlist)}")

        Pstd = np.std(np.array(all_vlist), ddof=1)
        Mstd = np.std(np.array(all_mlist), ddof=1)
        Estd = np.std(np.array(all_ee), ddof=1)
        Emae = CM.cal_ex_mae(all_ee)

        SR = CM.get_SR(all_vlist, period=args.period)  # SR
        ir = CM.get_ir(all_vlist, all_mlist, period=args.period)  # 计算IR-年化信息比率
        LAP = CM.get_bigloss_and_return(all_vlist, all_mlist)
        beta = CM.get_beta(all_vlist, all_mlist)  # 计算beta系数
        offensive = CM.get_offensive_power(all_vlist, all_mlist)
        defensive = CM.get_defensive_power(all_vlist, all_mlist)

        downrisk = CM.get_down_risk(all_vlist, period=args.period, no_risk_year_p=0.02)
        abs_value_list = [1.]
        abs_value = 1.
        for x in all_vlist:
            abs_value = abs_value * x
            abs_value_list.append(abs_value)

        ex_value_list = [1.]
        ex_value = 1.
        for i in range(len(all_vlist)):
            x = all_vlist[i] / all_mlist[i]
            ex_value = ex_value * x
            ex_value_list.append(ex_value)
        mdd = UTILS.max_drawdown(abs_value_list)

        Calmar_Rate = CM.cal_calmar_rate(all_vlist, mdd=mdd)  # 计算卡马比率
        SOR_Rate = CM.cal_sor_rate(all_vlist)

        emdd = UTILS.max_drawdown(ex_value_list)
        gmv = UTILS.cal_market_value(args, [all_startDate, all_endDate], force_read_file=True, indexName=args.indexName)
        mee = sum(all_ee) / len(all_ee)
        hold_stock = sum(all_hold_stocks) / len(all_hold_stocks)
        top1 = sum(all_top1) / len(all_top1)
        tr = sum(all_tr) / len(all_tr)
        scash_weight = sum(all_cash_weigh) / len(all_cash_weigh)

        abs_win_rate, win_loss_rate = CM.cal_win_loss_rate(all_vlist)  # 绝对胜率与盈亏比
        # avg_hold_days = CM.cal_avg_hold_days(all_weight)

        # smv = sum(all_mv) / len(all_mv)
        mvcount = (len(all_mv) - sum(x < 0.001 for x in all_mv))  # 极小的数(其实就是0)不参与分母计数
        smv = sum(all_mv) / (mvcount if mvcount > 0 else 1)

        # 新增加功能20260628 在每个年份的回测跑完之后,根据权重图数据,调用barra计算每日的因子暴露和因子收益
        alphaList = [x[-1] for x in list(year_ba.values())]  # alpha收益list(日度)
        alpha_year = sum(alphaList) / len(alphaList) * 252  # 年度alpha收益
        alpha_IR = CM.get_net_ir(alphaList, period='day')
        sizeI = BA.get_factor_index('size', 'exp')
        sizeList = [x[sizeI] for x in list(year_ba.values())]  # size暴露list(日度)
        size_Exp = sum(sizeList) / len(sizeList)

        v2 = victory / len(all_vlist) * 100.
        gv = 1.
        for v in all_vlist:
            gv *= v
        output_log(logfile, title)
        info = f"{all_startDate}~{all_endDate},{gv:.4f},{gmv:.4f},{(gv - gmv) / gmv * 100:.2f}%,{mee * 100:.2f}%"
        info = info + f",{CM.get_day_diff(Day_market, Day_model)}"
        info = info + f",{Pstd:.4f},{Mstd:.4f},{Estd:.4f},{Emae:.4f}"
        # info = info + f",{LAP[0]:.1f},{LAP[1]:.1f},{LAP[2]:.1f},{LAP[3]:.1f}"
        info = info + f",{SR:.4f},{ir:.4f},{mdd * 100:.2f}%,{emdd * 100:.2f}%,{downrisk:.4f}"
        info = info + f",{SOR_Rate:.4f},{Calmar_Rate:.4f}"
        info = info + f",{beta:.4f},{offensive:.4f},{defensive:.4f},{hold_stock:.1f}"  # ,{avg_hold_days:.1f}"
        info = info + f",{scash_weight:.2f},{smv:.2f},{top1:.4f},{tr:.2f},,{v2:.2f}%"
        info = info + f",{abs_win_rate * 100:.2f}%,{win_loss_rate:.2f}"
        info = info + f",{alpha_year * 100:.2f}%,{alpha_IR:.2f},{size_Exp:.2f}"
        output_log(logfile, info + "\n")
        logfile.close()
    print(f"{DT.timestr()}***测试结束***")
