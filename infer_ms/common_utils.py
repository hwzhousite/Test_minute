import argparse
import copy
import json
import math
import os
import platform
import re
import time
import numpy as np
import torch
import torch.nn.functional as F

# 全局唯一的一个配置信息,表示数据所在的根路径, 如果是挂载的NFS共享数据,则设置为 /data
# 如果是使用本地磁盘文件,则设置为 /swdata
G_ROOT_PATH = '/data'

model_base_args = ['DS', 'M', 'ad_info', 'blocks', 'channels', 'ci', 'close_price_index', 'cq_index', 'cash_layer',
                   'cross_stock_attn', 'd_model', 'data_norm', 'dnv', 'dropout', 'firstChannelIsID', 'sp_layer',
                   'fusion_manner', 'id_emb', 'global_stock_space', 'layers', 'maxw', 'moe', 'patch_size',
                   'patch_stride',
                   'pos_manner', 'pre_known_future', 'priceRelatedField', 'price_channels', 'price_layer',
                   'price_target', 'router_factor', 'self_group_size', 'stock_factor', 'time_step', 'use_prev_w']


def cal_ex_mae(ee):
    exsum = 0
    if len(ee) <= 0:
        return 0
    for e in ee:
        exsum += abs(e)
    return exsum / len(ee)


def get_SR(ret: list, period='week'):
    # ret 是模型每周收益的列表,它的值是在1附近
    if len(ret) <= 1:
        return 0.
    ret = np.array(ret) - 1.  # 减去1变为0附近的值,代表收益率
    if period == 'week':
        pl = 52
    elif period == '2week':
        pl = 26
    elif period == 'month':
        pl = 12
    else:
        pl = 252
    # 计算SR, 连续若干周的每周收益率的均值 / 每周收益率的标准差 * sqrt(52)
    SR = 1.0 * np.mean(ret) * np.sqrt(pl) / np.std(ret)
    return SR


def cal_avg_hold_days(all_weight):
    """
    给出持仓记录列表,统计平均持股天数
    """

    # 构造一个缓冲数据字典,表示当前所持有的股票,key=stock_code,value=截止目前累计持有的天数
    keep_stock = {}
    # 构造一个列表,用于记录每次持股的天数
    all_keep_days = []

    DL = len(all_weight)  # 总天数,按时序顺序排列的
    for i in range(DL):
        wD = all_weight[i]  # 当天的持股数据字典,key=stock_code,value=权重

        # 遍历当前所持股数据字典,对于今日已不再持有的股票,则将其移出
        has_keep_code = list(keep_stock.keys())
        for code in has_keep_code:
            if not wD.__contains__(code):
                # 股票code在今日持仓wD中已经不存在
                keep_days = keep_stock.pop(code)  # 移出当前持仓字典
                all_keep_days.append(keep_days)  # 持仓天数加入统计表

        for code, cw in wD.items():
            if code == 'CASH' or code == 'ETF':
                continue
            if cw <= 0.00001:
                continue
            if keep_stock.__contains__(code):
                # 当前已经持有这只股票,则对其持股天数+1
                keep_stock[code] += 1
            else:
                # 新买入一只原来本不持有的股票
                keep_stock[code] = 1
    # 循环结束后,可能keep_stock字典中仍然还有剩余持仓,但我们不知道未来会什么时候卖出,
    # 所以忽略这些股票,不做统计
    if len(all_keep_days) == 0:
        return 0
    return sum(all_keep_days) / len(all_keep_days)


def cal_win_loss_rate(vlist):
    """
    给出按周/月的模型绝对收益列表,计算绝对胜率和盈亏比
    """

    win_value = 0.
    win_count = 0
    loss_value = 0.
    loss_count = 0
    for x in vlist:
        if x > 1.0001:
            win_value += (x - 1)
            win_count += 1
        else:
            loss_value += (1 - x)
            loss_count += 1

    # 绝对胜率,即终点收益大于1的次数的占比
    win_rate = win_count / len(vlist)
    if win_count == 0:
        win_count = 1
    if loss_count == 0:
        loss_count = 1
    # 盈亏比: 所有盈利的周内平均每周盈利多少个点  /  所有亏损的周内平均每周亏损多少个点
    win_loss_rate = (win_value / win_count) / ((loss_value / loss_count) + 1e-8)
    return win_rate, win_loss_rate


def get_day_diff(Day_market, Day_model, threshold=0.005):
    """
    计算模型收益与业绩基准(大盘指数)收益之间的偏离度
    :param Day_market 大盘指数每日相对收益(1附近)
    :param Day_model  模型每日相对收益(1附近)--扣除交易成本后的
    :param threshold  合规阈值,默认为0.005
    :return [日均偏离度(算术),最大偏离,达标率,跟踪误差,风控阈值]
    """

    assert len(Day_market) == len(
        Day_model), f"基准数据长度{len(Day_market)}与模型收益数据长度{len(Day_model)}不相等..."
    # 计算每日模型收益与大盘指数收益之间的差异(绝对值)
    dif = [abs(x - y) for x, y in zip(Day_model, Day_market)]
    dif_wl = [x - y for x, y in zip(Day_model, Day_market)]

    DMAE = sum(dif) / (len(dif))  # 每日偏离的均值
    DMAX = max(dif)  # 最大值

    OKC = sum(x <= threshold for x in dif)  # 统计合规的天数
    OKC = OKC / len(dif)  # 达标率

    TERR = float(np.std(dif_wl, ddof=1)) * math.sqrt(252)  # 跟踪误差(均值/标准差)

    RISK = (threshold - DMAE) / threshold
    # [DMAE, DMAX, OKC, TERR, RISK]  # [日均偏离度(算术),最大偏离,达标率,跟踪误差,风控阈值]
    # info = f"{DMAE:.4f},{DMAX:.4f},{OKC * 100:.2f}%,{TERR * 100:.2f}%,{RISK * 100:.2f}%"
    info = f"{DMAE:.4f},{TERR * 100:.2f}%"
    return info


def cal_calmar_rate(value, mdd, risk_free_rate_annual=0.01):
    """
    计算卡马比率
    value 是每周的相对收益,值为1附近. 1.05表示这周盈利5%, 0.98表示这周亏损2%
    """

    # 计算总收益
    v = 1.
    for wv in value:
        v = v * wv
    N = len(value)  # 总的基数(周数)
    # 计算年化收益率
    annualized_return = ((v ** (52 / N)) - 1.) * 100.  # 百分数
    # 最大回撤也转换为百分数
    mdd = mdd * 100

    calmar = (annualized_return - risk_free_rate_annual) / (mdd + 1e-8)
    return calmar


def cal_sor_rate(value, risk_free_rate_annual=0.01, MAR_type='risk_free'):
    """
    计算索提诺比率

    参数：
    value: 每周收益率倍数列表，如[1.05, 0.92, 1.03, ...]
    risk_free_rate_annual: 年化无风险利率，默认为1%
    MAR_type: 最小可接受收益率类型，可选 'risk_free'（无风险利率）或 'zero'（0%）

    返回：
    sortino_ratio
    """

    # 1. 转换周收益率
    weekly_returns = [v - 1 for v in value]  # 小数形式
    N = len(weekly_returns)

    # 2. 计算周度无风险利率（精确复利转换）
    risk_free_rate_weekly = (1 + risk_free_rate_annual) ** (1 / 52) - 1

    # 3. 确定最小可接受收益率
    if MAR_type == 'risk_free':
        MAR = risk_free_rate_weekly
    elif MAR_type == 'zero':
        MAR = 0.0
    else:
        raise ValueError("MAR_type must be 'risk_free' or 'zero'")

    # 4. 计算下行偏差
    downside_deviations = []
    downside_periods = 0  # 记录下行期数

    for r in weekly_returns:
        if r < MAR:
            deviation = MAR - r  # 这是正数
            downside_deviations.append(deviation)
            downside_periods += 1
        else:
            downside_deviations.append(0)

    # 5. 计算下行标准差（周度）
    downside_variance = sum([d ** 2 for d in downside_deviations]) / N
    downside_std_weekly = math.sqrt(downside_variance)

    # 6. 年化下行标准差
    downside_std_annual = downside_std_weekly * math.sqrt(52)

    # 7. 计算年化收益率
    cumulative_value = 1.0
    for r in weekly_returns:
        cumulative_value *= (1 + r)

    total_return = cumulative_value - 1
    annualized_return = (1 + total_return) ** (52 / N) - 1

    # 8. 计算索提诺比率
    if downside_std_annual == 0:
        sortino_ratio = float('inf')
    else:
        sortino_ratio = (annualized_return - risk_free_rate_annual) / downside_std_annual

    return sortino_ratio


def get_down_risk(ret: list, period='month', no_risk_year_p=0.02):
    """
    计算下行风险
    """

    # 首先将无风险年化收益转换为与当前子区间相同的周期内收益
    if period == 'week':
        no_risk_year_p = no_risk_year_p / 52
    elif period == '2week':
        no_risk_year_p = no_risk_year_p / 26
    elif period == 'month':
        no_risk_year_p = no_risk_year_p / 12
    elif period == 'day':
        no_risk_year_p = no_risk_year_p / 252
    else:
        raise ValueError(f"ERROR:{period}")
    v = 0.
    if len(ret) <= 1:
        return 0.
    for x in ret:
        yx = x - 1  # 子区间收益率
        if yx < no_risk_year_p:
            # 子区间收益低于同期无风险收益,则统计在内
            v = v + (yx ** 2)
    v = math.sqrt(v / (len(ret) - 1))
    return v


def cal_down_cap(ret: list):
    """
    计算下行捕获率
    """
    up_v = 1.
    up_c = 0
    for x in ret:
        if x < 1.0:
            up_v *= x
            up_c += 1
    if up_c == 0:
        # 整个序列中都没有一个下行的
        return 1, False
    # 存在上行区间
    uc = up_v ** (1 / up_c) - 1
    return uc, True


def get_defensive_power(ret: list, dapan: list):
    """
    计算进攻能力
    :param ret 模型在连续若干个子区间内的相对收益[]
    :param dapan 大盘在连续若干个子区间内相对收益[]
    """

    m_uc, m_uc_valid = cal_down_cap(ret)
    dp_uc, dp_uc_valid = cal_down_cap(dapan)
    if m_uc_valid and dp_uc_valid:
        return m_uc / dp_uc
    else:
        return 0.


def cal_up_cap(ret: list):
    """
    计算上行捕获率
    """
    up_v = 1.
    up_c = 0
    for x in ret:
        if x >= 1.0:
            up_v *= x
            up_c += 1
    if up_c == 0:
        # 整个序列中都没有一个有正收益的
        return 1, False
    # 存在上行区间
    uc = up_v ** (1 / up_c) - 1
    return uc, True


def get_offensive_power(ret: list, dapan: list):
    """
    计算进攻能力
    :param ret 模型在连续若干个子区间内的相对收益[]
    :param dapan 大盘在连续若干个子区间内相对收益[]
    """

    m_uc, m_uc_valid = cal_up_cap(ret)
    dp_uc, dp_uc_valid = cal_up_cap(dapan)
    if m_uc_valid and dp_uc_valid:
        if abs(dp_uc) <= 1e-6:
            return 0.
        return m_uc / dp_uc
    else:
        return 0.


def get_beta(ret: list, dapan: list):
    """
    计算beta系数
    :param ret 模型在连续若干个子区间内的相对收益[]
    :param dapan 大盘在连续若干个子区间内相对收益[]
    """

    if len(ret) <= 1 or len(dapan) <= 1:
        return 0.

    # 计算收益率序列（从相对收益变化转换为收益率）
    fund_returns = np.array(ret) - 1
    market_returns = np.array(dapan) - 1

    # 计算贝塔系数
    covariance = np.cov(fund_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns, ddof=1)  # 使用ddof=1进行无偏估计
    beta = covariance / market_variance

    # print(f"贝塔系数: {beta:.4f}")
    return round(beta, 4)


def cal_max_continue_loss(vlist, baseValue):
    """
    给定一个数据列表,求出其中最大的连续小于baseValue的子区间长度
    """
    LM = 0
    DL = len(vlist)
    index = 0
    current_count = 0
    while index < DL:
        if vlist[index] >= baseValue:
            # 遇到一个大于base的数据,则原来的计数器要归0
            current_count = 0
        else:
            # 遇到一个小于base的数据,则继续计数
            current_count += 1
            if current_count > LM:
                LM = current_count
        index += 1
    return LM


def make_abs_value_list(vlist):
    """
    给定一个相对收益列表,生成绝对净值列表
    """
    abs_value = [1.]
    v = 1.
    for x in vlist:
        v = v * x
        abs_value.append(v)
    return abs_value


def find_first_bigger_value_in_list(vlist, i, CV):
    """
    给定一个序列,从第i个位置开始,查找第一个大于等于CV的元素,返回其位置
    """
    DL = len(vlist)
    for f in range(i, DL):
        if vlist[f] >= CV:
            return f
    return -1


def find_smallerest_value_in_list(vlist, i, CV):
    """
    给定一个序列,从第i个位置开始,查找小于CV的最小值的位置, 一旦再次遇到超过CV的项,则停止搜索
    """
    DL = len(vlist)
    smi = -1
    smv = None
    for f in range(i, DL):
        if vlist[f] < CV:
            if smv is None:
                smv = vlist[f]
                smi = f
            else:
                if vlist[f] < smv:
                    smv = vlist[f]
                    smi = f
        else:
            # 再次遇到超过CV的值
            break
    return smi


def point_is_peak(vlist, i):
    DL = len(vlist)
    if i == 0:
        if DL > 1 and vlist[i + 1] < vlist[i]:
            # 第1个节点,且总序列长度大于1,且第2个节点小于第1个节点,则1号节点是一个波峰
            return True
        else:
            return False
    if i >= DL - 1:
        return False
    if vlist[i - 1] <= vlist[i] > vlist[i + 1]:
        return True
    return False


def cal_average_repair(vlist):
    """
    给定一个净值序列,求出平均修复长度
    """
    # print("VLU:",vlist)
    replist = []
    DL = len(vlist)
    i = 0
    while i < DL - 1:
        # 首先判断i点是否为一个小波峰, 如果不是,则忽略
        if not point_is_peak(vlist, i):
            i += 1
            continue

        # i点是一个小波峰
        CV = vlist[i]  # 当前价值
        j = find_smallerest_value_in_list(vlist, i + 1, CV)  # 查找后面比CV小的最小数值
        if j == -1:
            # 序列中再无小于CV的值
            i += 1
            continue
        # 找到了i点之后的波谷位置j
        # 需要从j+1开始查找第一个大于等于CV的值,视为修复点
        z = find_first_bigger_value_in_list(vlist, j + 1, CV)
        if z == -1:
            # 未找到修复点
            i += 1
            continue
        dist = z - i  # 修复距离
        replist.append(dist)
        # print(f"#{i}-{vlist[i]}------#{z}-{vlist[z]}-------rep:{z-i}")
        i = z + 1
    if len(replist) == 0:
        return -1
    return sum(replist) / len(replist)


def make_ex_abs_value_list(model_list, base_list):
    """
    给定模型相对收益序列和业绩基准收益序列,计算相对价值序列
    """
    ex_value_list = [1.]
    ex_value = 1.
    for i in range(len(model_list)):
        x = model_list[i] / base_list[i]
        ex_value = ex_value * x
        ex_value_list.append(ex_value)
    return ex_value_list


def get_bigloss_and_return(model_value_list: list, base_value_list: list):
    # 计算最大连续亏损周期数
    L1 = cal_max_continue_loss(model_value_list, baseValue=1.)
    exList = [x - y for x, y in zip(model_value_list, base_value_list)]
    L2 = cal_max_continue_loss(exList, baseValue=0.)

    # 计算平均绝对回撤修复周期数
    absModelValue = make_abs_value_list(model_value_list)  # 生成绝对净值列表
    RP1 = cal_average_repair(absModelValue)  # 绝对回撤的平均修复周期

    relValue = make_ex_abs_value_list(model_value_list, base_value_list)
    RP2 = cal_average_repair(relValue)  # 相对回撤的平均修复周期

    return [L1, L2, RP1, RP2]


def get_net_ir(ret: list, period='day'):
    # 年化信息比率 (假定一年有252个交易日/52周/12月)
    # 输入 ret: 收益率序列 [B,episodes]  也可以是单个收益率序列 [episodes], 这个数值是在0附近的,表示当期的收益率
    #        dapan: 基准收益率序列 [B,episodes] 也可以是单个基准收益率序列 [episodes]
    #        period: 年化信息比率的计算周期, 默认为日度 (day), 可选周 (week), 月 (month), 2周 (2week)

    ret = torch.tensor(ret)  # [B,episodes]
    # 判断张量的维度是否只有1维
    if len(ret.shape) == 1:
        ret = ret.unsqueeze(0)
    B, E = ret.shape
    if E <= 1:
        return 0.

    excess_returns = ret  # 从get_ir代码改造过来的,ex就是ret

    # 3. 计算平均超额收益率
    mean_excess_return = torch.mean(excess_returns, dim=1)

    # 4. 计算跟踪误差 (超额收益的标准差)
    # 使用 ddof=1 计算样本标准差 (N-1 分母)
    tracking_error = torch.std(excess_returns, dim=1, unbiased=E > 1)

    # 5. 计算日度信息比率
    daily_ir = mean_excess_return / tracking_error

    # 6. 年化信息比率 (假设252个交易日)
    total_period = 252
    if period == 'day':
        total_period = 252
    elif period == 'week':
        total_period = 52
    elif period == 'month':
        total_period = 12
    elif period == '2week':
        total_period = 26
    annualized_ir = daily_ir * math.sqrt(total_period)

    # 打印结果
    # print(f"超额收益率: {excess_returns.tolist()}")
    # print(f"平均超额收益率: {mean_excess_return.item():.4f}")
    # print(f"跟踪误差: {tracking_error.item():.4f}")
    # print(f"日度信息比率: {daily_ir.item():.4f}")
    # print(f"年化信息比率: {annualized_ir.item():.4f}")

    # 取平均值返回
    return round(annualized_ir.mean().item(), 4)


def get_ir(ret: list, dapan: list, period='day'):
    # 年化信息比率 (假定一年有252个交易日/52周/12月)
    # 输入 ret: 收益率序列 [B,episodes]  也可以是单个收益率序列 [episodes]
    #        dapan: 基准收益率序列 [B,episodes] 也可以是单个基准收益率序列 [episodes]
    #        period: 年化信息比率的计算周期, 默认为日度 (day), 可选周 (week), 月 (month), 2周 (2week)

    ret = torch.tensor(ret)  # [B,episodes]
    # 判断张量的维度是否只有1维
    if len(ret.shape) == 1:
        ret = ret.unsqueeze(0)
    B, E = ret.shape
    if E <= 1:
        return 0.
    dapan = torch.tensor(dapan)  # [B,episodes]
    # 判断张量的维度是否只有1维
    if len(dapan.shape) == 1:
        dapan = dapan.unsqueeze(0)

    portfolio_returns = ret - 1.0
    benchmark_returns = dapan - 1.0

    # 2. 计算超额收益率 (主动收益)
    excess_returns = portfolio_returns - benchmark_returns

    # 3. 计算平均超额收益率
    mean_excess_return = torch.mean(excess_returns, dim=1)

    # 4. 计算跟踪误差 (超额收益的标准差)
    # 使用 ddof=1 计算样本标准差 (N-1 分母)
    tracking_error = torch.std(excess_returns, dim=1, unbiased=E > 1)

    # 5. 计算日度信息比率
    daily_ir = mean_excess_return / tracking_error

    # 6. 年化信息比率 (假设252个交易日)
    total_period = 252
    if period == 'day':
        total_period = 252
    elif period == 'week':
        total_period = 52
    elif period == 'month':
        total_period = 12
    elif period == '2week':
        total_period = 26
    annualized_ir = daily_ir * math.sqrt(total_period)

    # 打印结果
    # print(f"超额收益率: {excess_returns.tolist()}")
    # print(f"平均超额收益率: {mean_excess_return.item():.4f}")
    # print(f"跟踪误差: {tracking_error.item():.4f}")
    # print(f"日度信息比率: {daily_ir.item():.4f}")
    # print(f"年化信息比率: {annualized_ir.item():.4f}")

    # 取平均值返回
    return round(annualized_ir.mean().item(), 4)


def make_price_relative_v2(prices, price_channels: list, t1_open):
    """
    生成价格变化向量, 将所有价格类字段都除以T1的开盘价
    :param prices: [B, M, L, C]  M个资产的价格序列
    :param price_channels: 价格序列的通道数
    :param t1_open: 最后的开盘价[B,M]
    :return: [B,M, L, C]  M个资产的价格变化序列
    """

    B, M, L, C = prices.shape

    # 扩展维度
    t1_open = t1_open.view(B, M, 1).clone()

    # 计算价格占比
    for pos in price_channels:
        prices[:, :, :, pos] = prices[:, :, :, pos] / t1_open  # [B, M, L, C]
    return prices


def get_percent_list(a):
    r = []
    s = max(sum(a), 1)
    for i in range(len(a)):
        pi = round(a[i] / s, 2)
        r.append(pi)
    return r


def make_pp_label(num_class_value, dayT2_prices, base_prices):
    """
    生成预测价格的标签
    :param num_class_value: 分类数量列表list [-0.0261, -0.0076, 0.0, 0.0075, 0.0146, 0.0235, 0.0359, 0.0561, 0.1048]
    :param dayT2_prices: 价格序列 [B,M,P] M个资产T2日的价格
    :param base_prices: 基准价格序列 [B,M,1] M个资产的基准价格
    :return: 预测价格的标签 [B,M,P] M个资产连续P天的预测价格的标签
    """

    # B, M = dayT2_prices.shape
    device = dayT2_prices.device

    # 计算价格变化率(即相对于基准价格的涨跌幅)
    price_change = (dayT2_prices - base_prices) / (base_prices + 1e-6)  # [B, M]

    num_class = [-float('inf')] + num_class_value
    num_class = torch.tensor(num_class).to(device)  # [C+1]

    # 使用 torch.bucketize 来快速计算每个元素所在的区间编号
    x_label = torch.bucketize(price_change, num_class, right=False) - 1  # [B, M, P]

    # 扩展一个维度,1天
    # x_label = x_label.unsqueeze(2)  # [B, M, 1]

    return x_label


def cal_if_updown_stop(close_t0, tradeprice_t1, stock_type_t1, tradeDay):
    """
    判断是否已经涨停或跌停
    :param close_t0: 第0天的收盘价 [B,M]
    :param tradeprice_t1: 第1天的交易价格[B,M]
    :param stock_type_t1: 第1天的股票类型[B,M]
    :param tradeDay: 交易日期str-YYYYMMDD
    :return: is_up_stop, is_down_stop
    """

    # 计算t1日交易时刻是否已经在涨停（跌停）状态, 判断的依据是t1时刻的价格与t0收盘价的涨跌幅
    rate_trade = (tradeprice_t1 - close_t0 + 0.01) / close_t0  # [B,M]

    # 生成一个用于比较的矩阵[B,M]
    rate_limit_t1 = torch.zeros_like(rate_trade).to(rate_trade.device)  # [B,M]
    rate_limit_t1[:, :] = 0.1  # 主板股票的涨跌幅限制 10p (也是默认值)
    chuangyeban = 0.2
    if tradeDay < '20200824':
        chuangyeban = 0.1
    rate_limit_t1[stock_type_t1 == 2] = chuangyeban  # 创业板股票的涨跌幅限制 20p
    rate_limit_t1[stock_type_t1 == 3] = 0.3  # 北交所股票的涨跌幅限制 30p
    rate_limit_t1[stock_type_t1 == 4] = 0.2  # 科创板股票的涨跌幅限制 20p
    rate_limit_t1[stock_type_t1 == 5] = 0.05  # ST股票的涨跌幅限制 5p

    # t1交易时刻价格相对t0时刻收盘价的涨跌幅超过涨跌幅限制的股票, 则认为是涨停
    is_up_stop = rate_trade >= rate_limit_t1  # [B,M]

    # 统计在T1日开盘即跌停但模型预测要减仓事件发生的次数
    rate_limit_t1 = -1. * rate_limit_t1  # 取相反数变成下跌幅限制
    rate_trade = (tradeprice_t1 - close_t0 - 0.01) / close_t0  # [B,M]
    is_down_stop = rate_trade <= rate_limit_t1  # [B,M]

    return is_up_stop, is_down_stop


def trunc_float(x, d):
    f = 10 ** d
    x = math.floor(x * f) / f
    return x


def smooth_bjmax(stockCode, wi, BJMAX):
    """
    单独为BJ所股票进行平滑,即:设置所有BJ所股票的总权重占比不超过指定的阀值
    :param stockCode: 股票代码列表 list[M]
    :param wi: 权重 [1+M]
    :param BJMAX: 北交所股票的总占比最大阀值
    """

    BJT = BJMAX / 100.  # 总权重上限,从整数变成百分数

    # 首先计算所有BJ所股票的总权重
    CC = wi.shape[0]
    bjtotal = 0.
    nonebjtotal = 0.
    for i in range(1, CC):
        sc = stockCode[i - 1]
        if '.BJ' in sc:
            bjtotal += float(wi[i].item())
        else:
            nonebjtotal += float(wi[i].item())
    if bjtotal <= BJT:
        # 总权重本来就不超限, 直接返回
        return wi
    # 否则, 发现BJ所股票总额已超限,需要扣除
    disCountTotal = bjtotal - BJT  # 需要被扣减掉的权重部分

    # 保留的BJ所股票,每只都等比例压缩掉
    pr_bj = BJT / bjtotal  # BJ票的压缩系数
    # 所有非BJ所股票也需要等比例放大
    pr_none_bj = 1.  # 非BJ票的放大系数默认为1
    if nonebjtotal > 0:
        # 只有原本就存在非BJ票的时候,才有这个系数
        pr_none_bj = (nonebjtotal + disCountTotal) / nonebjtotal

    tw = 0.
    for i in range(1, CC):
        sc = stockCode[i - 1]
        if '.BJ' in sc:
            # BJ票,压缩
            wi[i] = wi[i] * pr_bj
        else:
            # 非BJ票,放大
            wi[i] = wi[i] * pr_none_bj
        tw += trunc_float(float(wi[i].item()), 6)
    wi[0] = 1. - tw
    return wi


def smooth_weight_BJMAX(stockCode, w, BJMAX=15):
    """
    单独为BJ所股票进行平滑,即:设置所有BJ所股票的总权重占比不超过指定的阀值
    :param stockCode: 股票代码列表 list[M]
    :param w: 权重 [1, 1+M]
    :param BJMAX: 北交所股票的总占比最大阀值
    """

    B, CC = w.shape
    ret_w = []
    for b in range(B):
        # 对单个样本进行处理
        wi = w[b]
        adw = smooth_bjmax(stockCode, wi, BJMAX)
        ret_w.append(adw)
    ret_w = torch.stack(ret_w, dim=0).to(w.device)  # [B, 1+M]
    return ret_w


def smooth_weight(w, smooth_weight=1e-3, single_max_weight=1.0, method='STOCK', movecash=0):
    """
    将权重中过小的值全部清0
    :param w: 权重 [B, 1+M]
    :param smooth_weight: 最小权重cd
    :param single_max_weight: 最只股票的最大权重(超过它的部分会被直接砍掉转入现金)
    :param method: 平滑方法['STOCK', 'CASH']
    :param movecash: 0:砍仓转移到现金  1/2/3...:转移到股票
    """

    B, CC = w.shape
    M = CC - 1  # 股票数量
    for i in range(B):
        sw = w[i, 1:]  # 股票权重
        # 找到sw中所有权重小于smooth_weight的索引
        small_index = torch.where(sw < smooth_weight)[0]
        smallCount = len(small_index)
        if smallCount == 0:
            continue
        # 计算这些待清零股票的权重的总和
        small_sum = sw[small_index].sum().item()
        if method == 'STOCK' and smallCount < M:
            # 平滑方法: 将清零的权重分摊到其他股票,且至少存在一只原本就持仓>于smooth_weight的股票
            # 计算非清零权重的总和
            non_zero_sum = sw.sum().item() - small_sum
            # 计算非清零股票应该等比例扩张的倍数系数
            ratio = sw.sum().item() / non_zero_sum  # smallCount<M确保了这里不会除0
            # 将所有待清零股票的权重全部清零
            sw[small_index] = 0.
            # 非清零股票等比例扩张
            sw *= ratio
        else:
            # 平滑方法: 将清零的权重分摊到现金
            # 将所有待清零股票的权重全部清零
            sw[small_index] = 0.
            # 现金权重增加
            w[i, 0] += small_sum

    # ADD 20260304 如果有设置单只股票最大上限,则在此处理
    if single_max_weight < 1.0:
        for i in range(B):
            killSW = 0.
            for j in range(M):
                if w[i, j + 1].item() > single_max_weight:
                    # 股票j的权重超出了最大上限
                    ks = float(w[i, j + 1].item()) - single_max_weight
                    killSW += ks  # 将被砍掉的部分记录下来
                    w[i, j + 1] = single_max_weight  # 股票被砍到上限
            if killSW > 1e-5:
                # 如果有砍下来的权重, 下面看如何分配这些权重
                if movecash == 0 or movecash == 2:
                    # 被砍掉的部分全部转移至现金项
                    w[i, 0] = w[i, 0] + killSW
                elif movecash == 1:
                    # 砍仓追加到其他股票,但仍需确保单只股票权重不超给定的阈值,即single_max_weight
                    csw = float(torch.sum(w[i, 1:]).item())  # 当下所有股票权重之和
                    fc = (csw + killSW) / csw  # 每只股票放大系数
                    for sm in range(M):
                        # 循环处理所有股票, 将每只股票的权重等比例放大, 但仍然不允放超过指定的最大权重
                        w[i, sm + 1] = min(float(w[i, sm + 1].item()) * fc, single_max_weight)
                    w[i, 0] = 1. - float(torch.sum(w[i, 1:]).item())  # 重算现金, 以免因精度问题导致和不为1

            # ADD 20260702 如果movechsh设置为2, 则表示需要尽量将所有权重都放到股票上去
            # 但仍需要确保单只股票权重不超过single_max_weight
            if movecash == 2 and w[i, 0] > 0.01:
                total_cash = float(w[i, 0].item())  # 当前可用于重新分配的现金量
                # 查询所有有权重的股票
                dic_info = {}
                for sm in range(M):
                    if w[i, sm + 1] > 1e-4:
                        dic_info[sm] = float(w[i, sm + 1].item())
                if len(dic_info) > 0:
                    # 存在有权重的股票,则按权重由大到小排序
                    sorted_w = sorted(dic_info.items(), key=lambda x: x[1], reverse=True)
                    # 按原本由大到小的顺序(代表着概率/看好程度),依次调升权重
                    for sm, sv in sorted_w:
                        # 循环处理每只股票,将其上调至最高的可能值
                        if float(w[i, sm + 1].item()) < single_max_weight:
                            uc = single_max_weight - float(w[i, sm + 1].item())
                            if uc <= total_cash:
                                w[i, sm + 1] = w[i, sm + 1] + uc
                                total_cash -= uc
                            else:
                                uc = total_cash
                                w[i, sm + 1] = w[i, sm + 1] + uc
                                total_cash = 0.
                        # 直至可用于调增的权重被耗尽
                        if total_cash <= 1e-4:
                            break
                w[i, 0] = 1. - float(torch.sum(w[i, 1:]).item())  # 重算现金, 以免因精度问题导致和不为1

    return w


def deal_illeagle_weight(pre_w, next_w, is_up_stop, is_down_stop):
    """
    处理非法的权重变化
    :param pre_w: 前一时刻的权重 [B,1+M]
    :param next_w: 后一时刻的权重 [B,1+M]
    :param is_up_stop: 在假定的交易时刻是否已经在涨停状态 [B,M]
    :param is_down_stop: 在假定的交易时刻是否已经在跌停状态 [B,M]
    :return: 处理后的权重 [1+M]
    """

    B, M = is_up_stop.shape
    ret_w = []
    for b in range(B):
        # 对单个样本进行处理
        w1 = pre_w[b].tolist()  # [1+M]
        w2 = next_w[b].tolist()  # [1+M]
        up_stop = is_up_stop[b].tolist()  # [M]
        if is_down_stop is not None:
            down_stop = is_down_stop[b].tolist()  # [M]
        else:
            down_stop = None
        adjust_w = deal_one_sample_weight(w1, w2, up_stop, down_stop)  # [1+M]
        ret_w.append(adjust_w)
    ret_w = torch.stack(ret_w, dim=0).to(pre_w.device)  # [B, 1+M]
    return ret_w


def deal_one_sample_weight(w1, w2, up_stop, down_stop):
    """
    处理单个样本的权重变化
    :param w1: 前一时刻的权重 [1+M]
    :param w2: 后一时刻的权重 [1+M]
    :param up_stop: 在假定的交易时刻是否已经在涨停状态 [M]
    :param down_stop: 在假定的交易时刻是否已经在跌停状态 [M]
    :return: 处理后的权重 [1+M]
    """

    # STEP1: 处理涨停情况
    M = len(w1) - 1
    adjw = copy.deepcopy(w2)
    if sum(up_stop) > 0:
        # 存在涨停, 则将涨停且加仓的权重恢复到原来的值
        restore_w = 0.
        for i in range(M):
            if up_stop[i] and w2[i + 1] > w1[i + 1]:
                # 该股票在涨停且加仓, 则将权重恢复到原来的值
                adjw[i + 1] = w1[i + 1]
                restore_w = restore_w + (w2[i + 1] - w1[i + 1])
        # 将所有误加的权重都给到现金帐户
        adjw[0] += restore_w

    # STEP2: 处理跌停情况
    if down_stop is not None and sum(down_stop) > 0:
        # 存在跌停, 则将跌停且减仓的权重恢复到原来的值
        restore_w = 0.
        have_weight_index = []  # 记录有权重且非跌停的股票的索引: 备用
        for i in range(M):
            if down_stop[i]:
                # 该股票处于跌停状态,则需要将其权重恢复到原来的值
                if w2[i + 1] > w1[i + 1]:
                    # 该股票在跌停且加仓, 则将权重恢复到原来的值(这个在实操中是允许的,但为了避开连续多个跌停的风险,我们暂时不允许加仓)
                    adjw[i + 1] = w1[i + 1]
                    # 这部分错误增加的权重,暂时给到现金项上去
                    delta = w2[i + 1] - w1[i + 1]
                    adjw[0] += delta
                elif w2[i + 1] < w1[i + 1]:
                    # 该股票在跌停且减仓, 则将权重恢复到原来的值(这个在实操中是绝对不允许的,因为跌停时无法卖出)
                    adjw[i + 1] = w1[i + 1]
                    delta = w1[i + 1] - w2[i + 1]
                    restore_w += delta
            else:
                # 该股票处于正常状态
                if w2[i + 1] > 0:
                    have_weight_index.append(i + 1)

        # 所有误减的权重首先都从现金帐户中取出
        # 但还需要判断,是否有足够的现金来补偿误减的权重
        if restore_w > 0:
            # 发生了误减行为
            if restore_w <= adjw[0]:
                # 现金帐户足以补偿误减的权重
                adjw[0] -= restore_w
            else:
                # 现金帐户不足以补偿误减的权重
                # 优先将现金全部分配完
                restore_w -= adjw[0]
                adjw[0] = 0.
                # 然后将剩余的权重按比例从其他股票中扣减
                # 先求出所有能扣减的权重的总和
                if len(have_weight_index) > 0:
                    crw = [adjw[i] for i in have_weight_index]
                    can_reduce_w = sum(crw)
                    for i in have_weight_index:
                        adjw[i] *= ((can_reduce_w - restore_w) / can_reduce_w)
                    # adjw[have_weight_index] *= ((can_reduce_w - restore_w) / can_reduce_w)
                else:
                    # 理论上程序不应该执行到此!!!
                    pass
    return torch.tensor(adjw)


def load_json_args(json_file):
    """
    加载json文件中的参数
    :param json_file: json文件路径
    :return: 参数字典
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        base_args = json.load(file)
        print(f"从文件{json_file}中加载模型主要参数")
        return base_args


def load_base_args(load_model_file):
    """
    尝试从文件中加载模型的基础参数
    :param load_model_file: 模型文件路径
    :return: 基础参数字典
    """

    if not load_model_file.endswith('.pth'):
        return None
    load_model_file = load_model_file.replace('.pth', '')
    if os.path.exists(load_model_file + '.basepm'):
        return load_json_args(load_model_file + '.basepm')

    # 判断basename是否以 '_epochXX' 结尾，如果是，则尝试去掉'_epochXX'
    pattern = r'_epoch\d+$'
    load_model_file = re.sub(pattern, '', load_model_file)

    if os.path.exists(load_model_file + '.basepm'):
        return load_json_args(load_model_file + '.basepm')

    return None


def load_args_from_file(model_file):
    assert os.path.exists(model_file), f"模型文件{model_file}不存在"
    base_args = load_base_args(model_file)
    assert base_args is not None, f"模型文件{model_file}关联的参数文件不存在"
    args = argparse.Namespace()
    for base_arg_name in base_args.keys():
        setattr(args, base_arg_name, base_args[base_arg_name])
    return args


def distribution_distance(w1, label, method='MSE'):
    """
    计算两个分布之间的距离
    :param w1: 第一个分布 [batch_size, num_classes]
    :param label: 第二个分布 [batch_size, num_classes]
    :param method: 距离计算方法
    :return: 距离 [batch_size]
    """

    if method == 'MSE':
        return F.mse_loss(w1, label, reduction='mean')
    elif method == 'MAE':
        loss = torch.sum(torch.abs(w1 - label), dim=1)
        return loss.mean()
    elif method == 'KL':
        loss = F.kl_div(w1.log(), label, reduction='batchmean')
        return loss
    else:
        # 自定义算法, 只考虑label中大于0的部分
        mask = label > 0
        loss = F.mse_loss(w1[mask], label[mask], reduction='mean')
        return loss


def distribution_loss(w1, w2, method='kl'):
    """
    计算两个概率分布之间的损失
    Args:
        w1: 第一个概率分布 [batch_size, num_classes]
        w2: 第二个概率分布 [batch_size, num_classes]
        method: 使用的损失函数类型
    """
    # 确保输入是概率分布（和为1）
    # if not torch.allclose(w1.sum(dim=1), torch.ones_like(w1.sum(dim=1))):
    #     print("Warning: w1 is not a probability distribution]\n", w1, w1.sum(dim=1))
    # if not torch.allclose(w2.sum(dim=1), torch.ones_like(w2.sum(dim=1))):
    #     print("Warning: w2 is not a probability distribution]\n", w2, w2.sum(dim=1))

    assert torch.allclose(w1.sum(dim=1), torch.ones_like(w1.sum(dim=1)))
    assert torch.allclose(w2.sum(dim=1), torch.ones_like(w2.sum(dim=1)))

    eps = 1e-8  # 数值稳定性

    if method == 'kl':
        # KL散度
        loss = torch.sum(w1 * torch.log(w1 / (w2 + eps) + eps), dim=1)

    elif method == 'symmetric_kl':
        # 对称KL散度
        kl1 = torch.sum(w1 * torch.log(w1 / (w2 + eps) + eps), dim=1)
        kl2 = torch.sum(w2 * torch.log(w2 / (w1 + eps) + eps), dim=1)
        loss = (kl1 + kl2) / 2

    elif method == 'js':
        # JS散度
        m = (w1 + w2) / 2
        loss1 = torch.sum(w1 * torch.log(w1 / (m + eps) + eps), dim=1)
        loss2 = torch.sum(w2 * torch.log(w2 / (m + eps) + eps), dim=1)
        loss = (loss1 + loss2) / 2

    elif method == 'l1':
        # L1距离
        loss = torch.sum(torch.abs(w1 - w2), dim=1)

    else:
        # elif method == 'l2':
        # L2距离
        loss = torch.sum((w1 - w2) ** 2, dim=1)

    return loss


def mark_tmp(OP, mark):
    if platform.system() == 'Windows':
        return False
    markfile = f"PFLAG_{mark}.t"
    if OP == 'clear':
        try:
            if os.path.exists(markfile):
                os.remove(markfile)
        except FileNotFoundError:
            pass
        return True
    elif OP == 'create':
        with open(markfile, 'w') as f:
            f.write(str(time.time()))
        return True
    else:  # check
        if os.path.exists(markfile):
            return True
        else:
            return False


def cal_big_stock_count(w, smooth_weight=1e-3):
    """
    计算权重中占比大于threshold的股票数量均值
    :param w: 权重 [batch_size, 1+M]
    :param smooth_weight: 平滑权重阈值
    :return: 大盘股票的数量
    """

    w = w.clone()
    B, S = w.shape
    # 去掉现金项,只计算股票权重
    w = w[:, 1:]
    big = w >= smooth_weight  # [B, M]
    t0 = big.sum().item() / B
    big = w >= 0.02  # [B, M]
    t2 = big.sum().item() / B
    big = w >= 0.05  # [B, M]
    t5 = big.sum().item() / B
    big = w >= 0.10  # [B, M]
    t10 = big.sum().item() / B
    big = w >= 0.20  # [B, M]
    t20 = big.sum().item() / B

    # 求所有股票中权重最大的前2名
    top2_index = torch.topk(w, k=2, dim=1)[1]  # [B, 2]

    # 求所有样本中权重最大的股票的平均权重
    max_weight = torch.gather(w, 1, top2_index[:, 0].unsqueeze(1)).squeeze(1)  # [B]
    st1 = max_weight.mean().item()  # 平均最大权重

    # 求所有样本中权重最大的2只股票权重和的平均值 (这里的权重和是指前2名的权重和)
    max_weight2 = torch.gather(w, 1, top2_index[:, 1].unsqueeze(1)).squeeze(1)  # [B]
    st2 = (max_weight + max_weight2).mean().item()  # 平均最大权重和

    return t0, t2, t5, t10, t20, st1, st2


def cal_weight_entropy(w):
    """
    计算权重的熵
    :param w: 权重 [batch_size, num_classes]
    :return: 权重的熵均值,现金的平均值
    """

    c = w[:, 0]

    # 去掉现金项,只计算股票权重
    w = w[:, 1:].clone()

    # 添加一个非常小的常数，以防止log(0)
    epsilon = 1e-10
    w = torch.clamp(w, min=epsilon)

    # 计算每个样本的熵
    entropy = -torch.sum(w * torch.log(w), dim=1)

    # 计算所有样本的平均熵
    mean_entropy = entropy.mean().item()
    return mean_entropy, c.mean().item()


def cal_portfolio_value(w0, w1, price_t0, price_t1):
    """
    计算两个权重之间的价值相对价值(即以price_t0为成本价,持有w0, 到price_t1时,资产价值几何?)
    :param w0: 第一个权重 [B, 1+M]
    :param w1: 第二个权重 [B, 1+M]
    :param price_t0: t0的价格 [B, M]
    :param price_t1: t1的价格 [B, M]
    :return: 价值 [B], 权重的变化量 [B]
    """

    B = w0.shape[0]

    cash_price_change = torch.ones(B, 1).to(w0.device)  # [B, 1]

    # 计算相对价格 [B,M]
    relative_price = price_t1 / price_t0  # [B,M]  相对价格
    relative_price = torch.cat([cash_price_change, relative_price], dim=1)  # [B,1+M]
    # NOTE: 本函数中对价值的计算与下一时刻权重无关，因此不用考虑权重的变化量
    next_value = torch.mul(w0, relative_price)
    next_value = next_value.sum(dim=1)  # [B]

    # w1意思是: 准备要在t1时刻之后,以price_t1为成本价,将目标仓位调整成为w1
    # 这里,顺便计算一下权重的变化量
    weight_change = torch.abs(w1 - w0)  # [B,asset_num]
    weight_change = weight_change[:, 1:]  # 去掉现金，只保留股票，确保形状为[B,M]
    weight_change = weight_change.sum(dim=1)  # [B]

    return next_value, weight_change


def noisy_data(data, start_idx, noiseChannel, rf=0.03, method='random'):
    B, M, L, C = data.shape
    deviceName = data.device

    if method == 'random':
        # 随机噪声
        x = torch.rand(B, M, L, noiseChannel).to(deviceName) * 2 * rf + (1 - rf)
    else:
        # 高斯噪音 'normal'
        x = torch.normal(mean=0, std=1., size=(B, M, L, noiseChannel)).to(deviceName)
        x = (x - x.min()) / (x.max() - x.min()) * (2 * rf) + (1 - rf)

    data[:, :, :, start_idx:start_idx + noiseChannel] *= x

    return data


def transform_relative_price_value(n):
    # if n <= 0.8:
    #     return 0.5
    # elif n >= 1.2:
    #     return 2.
    # if 0.99 <= n <= 1.01:
    #     return n
    # elif n < 1:
    #     return 1 - (1 - 0.5) * ((1 - n) / 0.2) ** 2
    # else:
    #     return 1 + (2 - 1) * ((n - 1) / 0.2) ** 2
    if 0.995 <= n <= 1.005:
        return n
    return 2 * n - 1


def make_fit_showbatch(args, tb):
    if args is not None and args.fast_running_its > 0:
        return 1
    num = tb / 100
    if num <= 20:
        return 20
    elif num <= 50:
        return 50
    else:
        return 100


def cal_loss(loss_func, pred, label, softlabel=0):
    """
    计算损失
    :param loss_func: 损失函数
    :param pred: 预测值, 注意这里的形状必须是: [Sample,Class]
    :param label: 标签值, 注意这里的形状必须是: [Sample]
    :param softlabel: 是否使用软标签
    :return: 损失值
    """

    if softlabel > 0:
        # 使用软标签
        S, C = pred.shape  # S为样本个数，C为类别数
        soft_labels = torch.arange(C).unsqueeze(0).expand(S, -1).to(label.device)
        soft_labels = -torch.abs(soft_labels - label) * 2.0
        soft_labels = F.softmax(soft_labels, dim=-1)
        return loss_func(pred, soft_labels)
    else:
        return loss_func(pred, label)


def save_base_args(args):
    # 保存模型参数到文件
    if args.mark == '':
        return  # 无需保存模型参数
    save_path = os.path.join(args.save_path, f"{args.mark}.basepm")  # 保存到mark同名文件中
    if os.path.exists(save_path):
        return  # 文件已存在，不需要保存
    # 生成基础参数字典
    base_args = {}
    ns_dict = vars(args)
    # 遍历字典并打印所有属性和它们的值
    for key, value in ns_dict.items():
        if isinstance(value, (int, float, str, bool, list)):
            base_args[key] = value
        # base_args[arg] = getattr(args, arg)
    # 将字典对象写入文件
    with open(save_path, 'w', encoding="utf-8") as file:
        json.dump(base_args, file)


def find_str_in_list(input_list, target_str):
    """
    找到列表中字符串的索引
    :param input_list: 输入列表
    :param target_str: 目标字符串
    :return: 字符串的索引
    """
    for i, item in enumerate(input_list):
        if item == target_str:
            return i
    print("---{} 不在列表中---".format(target_str))
    return -1


def find_strlist_in_list(input_list, target_list):
    """
    找到列表中字符串的索引列表
    :param input_list: 输入列表
    :param target_list: 目标字符串列表
    :return: 字符串的索引列表
    """

    retlist = []
    for target_str in target_list:
        retlist.append(find_str_in_list(input_list, target_str))
    return retlist


def listnum_tostr(a, n=4, l=7):
    r = ''
    for x in a:
        x = round(x, n)
        r = r + str(x).ljust(l - 1, '0') + ' '
    return r


def liststr(a):
    r = '['
    for x in a:
        r = r + str(x).ljust(9)
    r = r.rstrip() + ']'
    return r
