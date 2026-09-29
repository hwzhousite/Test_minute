"""
产品分析---根据过去的投资权重、指定一个业绩基准指数
分析以下指标：
1. 日度/周度/月度的净收益(平均值)/标准差
   日度/周度/月度的超额(平均值)/标准差
   日度/周度/月度--年化SR/IR
2. 历史最大绝对回撤及其修复时长
   历史最大相对回撤及其修复时长
   周度/月度的胜率(相对)
3. alpha收益(日度/周度/月度)-均值和标准差
4. 按暴露值排名前3(日均)的风格因子---及其收益
   按收益值排名前3(日均)的风格因子---及其暴露

"""
import argparse
import json
import math
import os
from time import time
import numpy as np
import utils as UTILS
import common_utils as CM
import product_eval as EVTOOL
import utils_date as DT
import utils_barra as BA


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
    parser.add_argument('--indexName', type=str, default='qa', help='业绩比较基准指数')

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
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path, args.indexName)

    args.indexFile = os.path.join("/data/indexData/", f"{args.indexName}.json")
    print(f"业绩比较基准为<{args.indexName}>,数据文件<{args.indexFile}>")
    assert os.path.exists(args.indexFile), "指数数据文件不存在"

    return args


def cal_info(model_value, base_value, period):
    """
    :param model_value 模型的日度收益(1附近净值,已扣除成本)
    :param base_value 同期的业绩基准收益(1附近)
    :param period str=[day,week,month]
    返回字典 {'abs_mean':v, 'abs_std':v, 'abs_var':v, 'ex_mean':v, 'ex_std':v, 'ex_var':v, 'win_a':v, 'win_r':v, 'sr':v ,'ir':v}
    """

    minfo = {}
    pdcount = {'day': 252, 'week': 52, 'month': 12}
    total_period = pdcount[period]
    if period != 'day':
        # 如果当前要求输出的不是日度数据,则需要先将数据统一转换为按指定周期的数据序列(是1附近的净值)
        # 月度的话, 有可能首末区间长度不足月,导致数据统计误差!!!
        m_d_v, wDayList = EVTOOL.make_seris_value(model_value, vtype='total', period=period)  # 返回的m_d_v已经是个list了
        b_d_v, _ = EVTOOL.make_seris_value(base_value, vtype='total', period=period)  # 这里b_d_v也是个list
    else:
        # 日度数据,需要从字典中提取出来
        m_d_v = list(model_value.values())
        b_d_v = list(base_value.values())

    # print(f"{period}-SRC1:",m_d_v)
    # print(f"{period}-SRC2:",b_d_v)

    # 数据分析
    DL = len(m_d_v)  # 总天数/周数/月数

    m_abs = [x - 1 for x in m_d_v]  # 将每日模型收益转换为0附近的数值(即收益率)
    minfo['abs_mean'] = float(np.mean(m_abs))  # 每日绝对收益的均值
    minfo['abs_std'] = float(np.std(m_abs))  # std
    minfo['abs_var'] = float(np.var(m_abs))  # var
    minfo['sr'] = minfo['abs_mean'] / minfo['abs_std'] * math.sqrt(total_period)

    m_ex = [x - y for x, y in zip(m_d_v, b_d_v)]  # 计算每日的算术超额(即模型收益-基准收益)
    minfo['ex_mean'] = float(np.mean(m_ex))  # 每日绝对收益的均值
    minfo['ex_std'] = float(np.std(m_ex))
    minfo['ex_var'] = float(np.var(m_ex))
    minfo['ir'] = minfo['ex_mean'] / minfo['ex_std'] * math.sqrt(total_period)

    minfo['win_a'] = sum(x >= 1. for x in m_d_v) / DL  # 绝对胜率(扣除成本后不亏本,净值大于等于1)
    minfo['win_r'] = sum(x >= 0.0000001 for x in m_ex) / DL  # 相对胜率(即有正超额)

    infoStr = "收益均值,收益标准差,绝对胜率,年化SR,超额均值,超额标准差,相对胜率,年化IR\n"
    infoStr = infoStr + f"{minfo['abs_mean']:.6f},{minfo['abs_std']:.6f},{minfo['win_a'] * 100:.2f}%,{minfo['sr']:.4f}"
    infoStr = infoStr + f",{minfo['ex_mean']:.6f},{minfo['ex_std']:.6f},{minfo['win_r'] * 100:.2f}%,{minfo['ir']:.4f}"

    return minfo, infoStr


def max_drawdown(v):
    if len(v) == 0:
        return 0.0  # 空序列返回0

    peak = v[0]  # 初始化历史峰值
    max_dd = 0.0  # 初始化最大回撤
    peak_i = 0
    mp_i = 0
    max_dd_i = 0

    for i in range(1, len(v)):
        if v[i] > peak:
            peak = v[i]  # 更新峰值
            peak_i = i
        else:
            drawdown = (peak - v[i]) / peak
            if drawdown > max_dd:
                max_dd = drawdown  # 更新最大回撤
                max_dd_i = i
                mp_i = peak_i
    return max_dd, mp_i, max_dd_i


def repair_info(m_d_v, wDayList, pi, mi):
    """
    m_d_v 净值列表
    wDayList 日期列表
    pi MDD所对应的峰值下标
    mi MDD所对应的谷值下标
    这里要寻找修复点,返回相关信息
    """
    DL = len(m_d_v)
    pv = m_d_v[pi]
    gv = m_d_v[mi]
    for i in range(mi + 1, DL):
        if m_d_v[i] >= pv:
            # 价值得以修复
            return f"从{wDayList[mi]}:{gv:.4f}经过{i - mi}个交易日到{wDayList[i]}:{m_d_v[i]:.4f}成功修复."
    # 如果循环内部未达到修复点,则表明当前序列的最后时间点仍然在修复中
    return f"从{wDayList[mi]}:{gv:.4f}经过{DL - mi - 1}个交易日到{wDayList[-1]}:{m_d_v[-1]:.4f}仍未修复."


def cal_wd_info(model_value, base_value):
    """
    :param model_value 模型的日度收益(1附近净值,已扣除成本)
    :param base_value 同期的业绩基准收益(1附近)
    返回字典 {'abs_mdd':v, 'abs_mdd_days':v, 'abs_repair_days':v,'rel_mdd':v, 'rel_mdd_days':v, 'rel_repair_days':v,}
    """

    minfo = {}

    # 计算出累计实际净值列表
    m_d_v, wDayList = EVTOOL.make_day_total_net_value(model_value)  # 返回的m_d_v已经是个list了
    abs_mdd, pi, mi = max_drawdown(m_d_v)
    minfo['abs_mdd'] = abs_mdd
    minfo['abs_mdd_info'] = f"从{wDayList[pi]}:{m_d_v[pi]:.4f}经过{mi - pi}个交易日到{wDayList[mi]}:{m_d_v[mi]:.4f}"
    minfo['abs_repair_info'] = repair_info(m_d_v, wDayList, pi, mi)

    # 计算模型相对净值列表(1附近的净值)
    m_d_v = list(model_value.values())
    # 计算业绩基准的相对净值列表(1附近的净值)
    b_d_v, _ = EVTOOL.make_seris_value(base_value, vtype='total', period='day')
    # 计算模型相对于业绩基准的相对净值(1附近)
    m_to_b = [x / y for x, y in zip(m_d_v, b_d_v)]
    # 序列改造为累计值
    v = m_to_b[0]
    m_to_b_net = [v]
    for i in range(1, len(m_to_b)):
        v = v * m_to_b[i]
        m_to_b_net.append(v)
    # 计算相对回撤
    rel_mdd, pi, mi = max_drawdown(m_to_b_net)
    minfo['rel_mdd'] = rel_mdd
    minfo[
        'rel_mdd_info'] = f"从{wDayList[pi]}:{m_to_b_net[pi]:.4f}经过{mi - pi}个交易日到{wDayList[mi]}:{m_to_b_net[mi]:.4f}"
    minfo['rel_repair_info'] = repair_info(m_to_b_net, wDayList, pi, mi)

    infoStr = f"最大绝对回撤{minfo['abs_mdd'] * 100:.2f}%\n"
    infoStr += f"信息:{minfo['abs_mdd_info']}\n"
    infoStr += f"修复:{minfo['abs_repair_info']}\n"

    infoStr += f"最大相对回撤{minfo['rel_mdd'] * 100:.2f}%\n"
    infoStr += f"信息:{minfo['rel_mdd_info']}\n"
    infoStr += f"修复:{minfo['rel_repair_info']}"

    return minfo, infoStr


def cal_alpha_info(ba_dict):
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    minfo = {}
    alpha_list = []
    alpha_dict = {}
    days = list(ba_dict.keys())
    for d in days:
        alpha_list.append(ba_dict[d]['pm_alpha'])
        alpha_dict[d] = ba_dict[d]['pm_alpha'] + 1.

    # 计算日度均值和标准差
    minfo['day_mean'] = float(np.mean(alpha_list))
    minfo['day_std'] = float(np.std(alpha_list))

    # 计算周度的alpha变化率(0附近）
    alpha_week, wDayList = EVTOOL.make_seris_value(alpha_dict, vtype='change', period='week')  # 返回的m_d_v已经是个list了
    minfo['week_mean'] = float(np.mean(alpha_week))
    minfo['week_std'] = float(np.std(alpha_week))

    # 计算月度的alpha变化率(0附近）
    alpha_month, wDayList = EVTOOL.make_seris_value(alpha_dict, vtype='change', period='month')  # 返回的m_d_v已经是个list了
    minfo['month_mean'] = float(np.mean(alpha_month))
    minfo['month_std'] = float(np.std(alpha_month))

    infoStr = "日均,日度STD,周均,周度STD,月均,月度STD\n"
    infoStr += f"{minfo['day_mean']:.6f},{minfo['day_std']:.6f}"
    infoStr += f",{minfo['week_mean']:.6f},{minfo['week_std']:.6f}"
    infoStr += f",{minfo['month_mean']:.6f},{minfo['month_std']:.6f}"

    return minfo, infoStr


def get_avg_value(ba_dict, fn, keyname):
    value = []
    for d, daydic in ba_dict.items():
        pm_value = daydic[keyname]
        v = pm_value[fn]
        value.append(v)
    return sum(value) / len(value)


def cal_most_factor_info(ba_dict, topK, keyname, other_name, factorType='style'):
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    # 找到日均暴露最大的3个因子,输出其信息

    days = list(ba_dict.keys())
    # 所有的因子列表
    factor_list = []
    if factorType == 'style':
        factor_list.extend(BA.style_factor1)
    elif factorType == 'industry':
        factor_list.extend(BA.industry_factor)
    else:
        factor_list.extend(BA.style_factor1)
        factor_list.extend(BA.industry_factor)
    f_exp = {}
    f_exp_avg = {}
    f_exp_avg_real={}
    for f in factor_list:
        f_exp[f] = []
        f_exp_avg[f] = 0.
        f_exp_avg_real[f] = 0.
    for d in days:
        pm_exp = ba_dict[d][keyname]
        for f, expv in pm_exp.items():
            if f not in factor_list:
                continue
            f_exp[f].append(expv)
    for f, lv in f_exp.items():
        if keyname == 'pm_exp':
            # 主键是暴露的话,就按绝对值来排序,我们看的是极端情况
            f_exp_avg[f] = abs(sum(lv) / len(lv))
        else:
            f_exp_avg[f] = sum(lv) / len(lv)
        f_exp_avg_real[f]= sum(lv) / len(lv)
    # 降序排列
    sorted_w = sorted(f_exp_avg.items(), key=lambda x: x[1], reverse=True)
    topKinfo = []
    for i in range(min(len(sorted_w), topK)):
        fn = sorted_w[i][0]
        fn_exp = f_exp_avg_real[fn]
        # 计算该因子的平均收益
        fn_value = get_avg_value(ba_dict, fn, other_name)
        topKinfo.append([fn, fn_exp, fn_value])

    if keyname == 'pm_exp':
        na, nb = '日均暴露', '日均收益'
    else:
        na, nb = '日均收益', '日均暴露'
    infoStr = f"排名,因子名称,{na},{nb}"
    for i in range(len(topKinfo)):
        kf = topKinfo[i]
        infoStr += f"\n#{i + 1},{kf[0]},{kf[1]:.8f},{kf[2]:.8f}"

    return topKinfo, infoStr


def analysis_product(args, path, match_year):
    """
    :param args 运行参数
    :param path 权重图所在的路径(默认该路径下也存在barra分析数据)
    :param match_year 前缀匹配字符串
    """

    # 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据,过滤匹配'{match_year}'...")
    ts = time()
    weights = UTILS.read_weigh_from_path(path=path, match_year=match_year)
    DayCount = len(weights)
    dayList = list(weights.keys())
    print(f"{CM.timestr()}读取权重数据完成,天数{DayCount},耗时{time() - ts:.2f}秒")

    # 读取指数数据
    indexBase = CM.load_index(args.indexFile)

    # 读取所有股票信息
    print(f"{CM.timestr()}读取所有交易日股票信息...")
    ts = time()
    # 一级索引为股票代码,二级索引为该股票出现过的日期,最后value是一个list
    all_stock_info = UTILS.read_all_stock_info(weights)
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = EVTOOL.make_price_info_from_totalinfo(args.targetPrice, all_stock_info)
    print(f"{CM.timestr()}读取所有交易日价格信息完成,天数{len(priceInfo)},耗时{time() - ts:.2f}秒")

    # 创建输出文件CSV
    # outname = os.path.join(args.out_path, f'{args.mark}模型评估表.csv')
    # f = open(outname,'w',encoding='utf-8-sig')

    # 计算每日收益,这里会计算每日绝对收益(扣除成本)!!!
    # 下面的方法返回:
    # 1.每天模型的相对前一天的价值(税后)--税后是指扣除了交易成本
    # 2.每天盈利股票
    # 3.每天带来亏损的股票
    model_value, _, _ = EVTOOL.cal_trade_series_value(args.tradeCost, weights, priceInfo)  # 计算模型收益
    base_value = EVTOOL.cal_seris_value(indexBase, dayList)  # 计算同期的基准收益

    # 计算下面的一些指标
    """
    日度 / 周度 / 月度的净收益(平均值) / 标准差  /绝对胜率 /相对胜率
    日度 / 周度 / 月度的超额(平均值) / 标准差
    日度 / 周度 / 月度 - -SR / IR
    """
    # 返回字典 {'mean':v, 'std':v, 'var':v, 'win_a':v, 'win_r':v, 'sr':v ,'ir':v}
    m_day, infoStr = cal_info(model_value, base_value, period='day')
    print(f"{args.mark}-{dayList[0]}-{dayList[-1]}--index[{args.indexName}]")
    print(f"日度统计信息\n{infoStr}\n")

    minfo_week, infoStr = cal_info(model_value, base_value, period='week')
    print(f"周度统计信息\n{infoStr}\n")

    minfo_month, infoStr = cal_info(model_value, base_value, period='month')
    print(f"月度统计信息\n{infoStr}\n")

    # 计算历史最大回撤及修复时长(绝对和相对)
    minfo_wd, infoStr = cal_wd_info(model_value, base_value)
    print(f'最大回撤信息\n{infoStr}\n')

    # 因子相关信息
    # 检查权重目录下是否已经存在了barra分析数据
    bafile = os.path.join(args.path, "barra.json")
    if not os.path.exists(bafile):
        raise ValueError(f"{bafile}文件不存在")
    # 读取barra数据字典
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    with open(bafile, 'r', encoding='utf-8-sig') as file:
        ba_dict = json.load(file)
    minfo_alpha, infoStr = cal_alpha_info(ba_dict)
    print(f'ALPHA收益\n{infoStr}\n')

    topK = 5
    fact = 'style'
    top_factor_mexp, infoStr = cal_most_factor_info(ba_dict, topK=topK, keyname='pm_exp', other_name='pm_value',
                                                    factorType=fact)
    print(f'TOP{topK}-{fact}因子暴露\n{infoStr}\n')
    top_factor_value, infoStr = cal_most_factor_info(ba_dict, topK=topK, keyname='pm_value', other_name='pm_exp',
                                                     factorType=fact)
    print(f'TOP{topK}-{fact}因子收益\n{infoStr}\n')

    fact = 'industry'
    top_factor_mexp, infoStr = cal_most_factor_info(ba_dict, topK=topK, keyname='pm_exp', other_name='pm_value',
                                                    factorType=fact)
    print(f'TOP{topK}-{fact}因子暴露\n{infoStr}\n')
    top_factor_value, infoStr = cal_most_factor_info(ba_dict, topK=topK, keyname='pm_value', other_name='pm_exp',
                                                     factorType=fact)
    print(f'TOP{topK}-{fact}因子收益\n{infoStr}\n')


if __name__ == '__main__':
    args = get_args()
    analysis_product(args, args.path, args.match_year)
