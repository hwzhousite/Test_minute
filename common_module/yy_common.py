import argparse
import copy
import datetime
import json
import math
import os
import platform
import random
import re
import time
import numpy as np
import pytz
import torch
import torch.nn.functional as F


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


def cal_if_updown_stop(close_t0, tradeprice_t1, stock_type_t1, tradeDay=None):
    """
    判断是否已经涨停或跌停
    :param close_t0: 第0天的收盘价 [B,M]
    :param tradeprice_t1: 第1天的交易价格[B,M]
    :param stock_type_t1: 第1天的股票类型[B,M]
    :param tradeDay: 第1天(即交易日)的日期  [B, M], 为None时不区分创业板20200824前后的涨跌幅限制
    :return: is_up_stop, is_down_stop
    """

    # 计算t1日交易时刻是否已经在涨停（跌停）状态, 判断的依据是t1时刻的价格与t0收盘价的涨跌幅
    rate_trade = (tradeprice_t1 - close_t0 + 0.01) / close_t0  # [B,M]

    # 生成一个用于比较的矩阵[B,M]
    rate_limit_t1 = torch.zeros_like(rate_trade).to(rate_trade.device)  # [B,M]
    rate_limit_t1[:, :] = 0.1  # 主板股票的涨跌幅限制 10p (也是默认值)
    rate_limit_t1[stock_type_t1 == 2] = 0.2  # 创业板股票的涨跌幅限制 20p
    if tradeDay is not None:
        isSp_chuang = tradeDay < 20200824
        rate_limit_t1[stock_type_t1 == 2 & isSp_chuang] = 0.1  # 创业板股票的涨跌幅限制在20200824之前为 10p
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


def smooth_weight(w, smooth_weight=1e-3, method='STOCK'):
    """
    将权重中过小的值全部清0
    :param w: 权重 [B, 1+M]
    :param smooth_weight: 最小权重cd
    :param method: 平滑方法['STOCK', 'CASH']
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
        w1 = pre_w[b]  # [1+M]
        w2 = next_w[b]  # [1+M]
        up_stop = is_up_stop[b]  # [M]
        down_stop = is_down_stop[b]  # [M]
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
    M = w1.shape[0] - 1
    adjw = w2.clone()
    if torch.sum(up_stop) > 0:
        # 存在涨停, 则将涨停且加仓的权重恢复到原来的值
        restore_w = 0.
        for i in range(M):
            if up_stop[i] and w2[i + 1] > w1[i + 1]:
                # 该股票在涨停且加仓, 则将权重恢复到原来的值
                adjw[i + 1] = w1[i + 1].item()
                restore_w = restore_w + (w2[i + 1].item() - w1[i + 1].item())
        # 将所有误加的权重都给到现金帐户
        adjw[0] += restore_w

    # STEP2: 处理跌停情况
    if torch.sum(down_stop) > 0:
        # 存在跌停, 则将跌停且减仓的权重恢复到原来的值
        restore_w = 0.
        have_weight_index = []  # 记录有权重且非跌停的股票的索引: 备用
        for i in range(M):
            if down_stop[i]:
                # 该股票处于跌停状态,则需要将其权重恢复到原来的值
                if w2[i + 1] > w1[i + 1]:
                    # 该股票在跌停且加仓, 则将权重恢复到原来的值(这个在实操中是允许的,但为了避开连续多个跌停的风险,我们暂时不允许加仓)
                    adjw[i + 1] = w1[i + 1].item()
                    # 这部分错误增加的权重,暂时给到现金项上去
                    delta = w2[i + 1].item() - w1[i + 1].item()
                    adjw[0] += delta
                elif w2[i + 1] < w1[i + 1]:
                    # 该股票在跌停且减仓, 则将权重恢复到原来的值(这个在实操中是绝对不允许的,因为跌停时无法卖出)
                    adjw[i + 1] = w1[i + 1].item()
                    delta = w1[i + 1].item() - w2[i + 1].item()
                    restore_w += delta
            else:
                # 该股票处于正常状态
                if w2[i + 1] > 0:
                    have_weight_index.append(i + 1)

        # 所有误减的权重首先都从现金帐户中取出
        # 但还需要判断,是否有足够的现金来补偿误减的权重
        if restore_w > 0:
            # 发生了误减行为
            if restore_w <= adjw[0].item():
                # 现金帐户足以补偿误减的权重
                adjw[0] -= restore_w
            else:
                # 现金帐户不足以补偿误减的权重
                # 优先将现金全部分配完
                restore_w -= adjw[0].item()
                adjw[0] = 0.
                # 然后将剩余的权重按比例从其他股票中扣减
                # 先求出所有能扣减的权重的总和
                if len(have_weight_index) > 0:
                    can_reduce_w = adjw[have_weight_index].sum()
                    adjw[have_weight_index] *= ((can_reduce_w - restore_w) / can_reduce_w)
                else:
                    # 理论上程序不应该执行到此!!!
                    pass
    return adjw


def make_next_portfolio_by_pp(ep_portfolio, pred, is_t1_up_stop, is_t1_down_stop, exp_value_list, maxstock,
                              method='ALL', vpc=1, wf=1000.):
    """
    根据价格预测的结果来手工生成下一个时刻的投资组合权重
    :param ep_portfolio: 第i-1时刻的投资组合权重 [B, 1+M]
    :param pred: 第i时刻的价格预测结果 [B,M,P,C]
    :param is_t1_up_stop: 第1天是否已经在涨停状态 [B,M]
    :param is_t1_down_stop: 第1天是否已经在跌停状态 [B,M]
    :param exp_value_list: 各档位预期收益列表
    :param maxstock: 最多持仓股票数
    :param vpc: 最有价值区间(倒数)
    :param wf: 温度系数
    :param method: 手工生成下一个时刻的投资组合权重的方法
    :return: 下一个时刻的投资组合权重 [B, 1+M]
    """

    B, M, P, C = pred.shape
    next_w = []
    exp_value = []

    pct = 0.
    if method == 'ALL':
        pass
    elif method.startswith('PCT_'):
        pct = int(method.split('_')[-1])
        pct /= 100.
    else:
        raise ValueError(f"不支持的手工生成下一个时刻的投资组合权重的方法:{method}")

    for b in range(B):
        # 对单个样本进行处理
        w = ep_portfolio[b, 1:]  # [1+M]
        p = pred[b]
        t1up_stop = is_t1_up_stop[b]
        t1down_stop = is_t1_down_stop[b]

        if method == 'ALL':
            nw, nv = make_next_w_by_pp(w, p, t1up_stop, t1down_stop, exp_value_list, maxstock, vpc=vpc, wf=wf)  # [1+M]
        else:
            nw, nv = make_next_w_by_pct(w, p, t1up_stop, t1down_stop, exp_value_list, maxstock, pct, wf=wf)  # [1+M]
        next_w.append(nw)
        exp_value.append(nv)
    next_w = torch.stack(next_w, dim=0)  # [B, 1+M]
    if method == 'ALL':
        exp_value = torch.stack(exp_value, dim=0)  # [B, M]
    else:
        exp_value = torch.tensor(exp_value)
    return next_w.to(ep_portfolio.device), exp_value.to(ep_portfolio.device)


def cal_topK_positive_percent(pp, topK=0, vpc=1):
    """
    根据当前的预测分布,计算一个topK的平均正收益率
    :param pp: [M,C] 预测的概率分布
    :param topK: 计算topK的平均正收益率
    :param vpc: 最有价值区间(倒数)
    :return: positive_percent: 预测的期望收益
    """

    M, C = pp.shape  # 这里C是分类数
    if topK == 0:
        topK = int(max(M * 0.001, 5))

    # 取出尾部的self.args.very_positive_class个概率数值
    very_pos = pp[:, -vpc:]
    # 将这些概率值进行求和
    very_pos = very_pos.sum(dim=1)  # [M]

    # 从中选出topK个概率最大的
    topK_value, topK_index = torch.topk(very_pos, k=topK, dim=0)

    # 求topK_value的均值
    positive_percent = topK_value.mean().item()

    return (positive_percent + 9) / 10


def calc_expected_return(pred_class, expect_class_value):
    """
    计算预测的期望收益率
    :param pred_class: [M, C] 预测的分类结果,这里的C是分类数
    :param expect_class_value: [C] 各个分类的期望收益率
    :return: expected_return: [M] 预测的期望收益
    """

    M, C = pred_class.shape
    class_value = torch.tensor(expect_class_value).to(pred_class.device)  # [C]
    class_value = class_value.unsqueeze(0).expand(M, -1)  # [M, C]
    expected_return = torch.mul(pred_class, class_value).sum(dim=1)  # [M]
    return expected_return


def make_next_w_by_pp(w, p, t1up_stop, t1down_stop, exp_value_list, maxstock, vpc=1, wf=1000.):
    """
    根据价格预测的结果来手工生成下一个时刻的投资组合权重
    :param w: 第i-1时刻的投资组合权重 [M]
    :param p: 第i时刻的价格预测结果 [M,P,C]
    :param t1up_stop: 第1天是否已经在涨停状态 [M]
    :param t1down_stop: 第1天是否已经在跌停状态 [M]
    :param exp_value_list: 各个分类的期望收益率
    :param maxstock: 最多持仓股票数
    :param vpc: 最有价值区间(倒数)
    :param wf: 温度系数
    :return: 下一个时刻的投资组合权重 [1+M], 预期收益 [M]
    """

    M, P, C = p.shape  # 这里P=1
    # 把p这个维度去掉,变成[M,C]
    p = p.squeeze(1)
    # 计算t1开盘时刻是否已经在涨停（跌停）状态, 判断的依据是t1时刻开盘价与t0收盘价的涨跌幅
    # 生成一个全0的权重向量[M] 只用于表示股票项
    nw = torch.zeros(M).to(w.device)  # [M]
    # 将已经处于跌停状态的股票权重直接复制过来, 这些股票的权重将直接保留(因为无法卖出)
    nw[t1down_stop] = w[t1down_stop]

    nv = [0. for _ in range(M)]  # [M] 预期收益
    # nv[t1down_stop] = 100.  # 跌停股票的预期收益标记为100

    # 计算所有股票的期望价值, 期望价值 = sum(预测区间所在的概率 * 区间的涨跌幅)
    # 首先要将预测结果转换为概率分布
    pp = F.softmax(p, dim=-1)  # [M,C]

    # 计算一个topK的平均正收益率
    if vpc == 1:
        pass
    topK_positive_pct = 1.  # cal_topK_positive_percent(pp, vpc=vpc)
    # 先验值: 我们用于投资的所有资金应该占全部资金的百分比,不应该超过 topK_positive_pct

    # 计算所有股票的预期收益
    expected_return = calc_expected_return(pp, exp_value_list)  # [M]
    topS = maxstock  # 最多持仓股票数
    # 取出预期收益前topS个股票
    topS_expv, topS_index = torch.topk(expected_return, k=topS, dim=0)  # 选出topS个股票的预期收益
    # 对这些股票进行一次过滤排查
    keep_weight = 0.
    need_buy_index = []  # 需要买入的股票的索引
    need_buy_expv = []  # 需要买入的股票预期价值
    sh = 0.0005  # 设置一个阈值, 低于这个阈值的股票不买入
    for i in range(topS):
        si = topS_index[i]
        ev = topS_expv[i]
        if ev < sh:
            # 从高到低排序到这只股票时, 预期收益已经小于sh,
            # 则从这只股票开始,后面的股票都不值得入手了,退出循环
            break

        if w[si] > 0.:
            # 如果该股票昨日已经持仓
            if t1down_stop[si]:
                # 如果该股票今早开盘已经跌停, 则前面已经统一一次性处理过了,这里必须跳过
                nv[si] = 100 + ev  # 跌停股票的预期收益标记为100+预期收益
                continue
            # 其他情况: 该股票都是有正收益的,且期望值大于sh,则认为该股票值得继续持有
            keep_weight += w[si]
            nw[si] = w[si]
            nv[si] = 1000 + ev  # 前日已持仓,标记为1000+预期收益
        else:
            # 如果该股票没有持仓, 则应该尝试买入,除非今早开盘就涨停了(此时买入不到)
            if not t1up_stop[si]:
                # 未涨停,则标记为待买入的股票
                need_buy_index.append(si)
                need_buy_expv.append(ev)
                nv[si] = ev  # 标记为待买入的股票的预期收益

    # 计算可用权重
    used_w = nw.sum().item()  # 已使用的权重
    # 将list 转为 tensor
    nv = torch.tensor(nv).to(w.device)  # [M] 预期收益

    # 所有可用的资金
    ideal_cash = 1 - topK_positive_pct  # 理论应该保留的现金数
    current_cash = 1 - used_w  # 当前现金占比
    if current_cash <= ideal_cash:
        # 如果当前实际现金已小于理论现金,则表明不再需要增加投资了,直接按现有持仓继续保持即可
        weight = torch.tensor([current_cash]).to(w.device)  # 现金权重为1
        weight = torch.cat([weight, nw], dim=0)  # 合并现金权重和股票权重 [1+M]
        return weight, nv

    available_cash = current_cash - ideal_cash  # 剩余可用资金
    # 需要将剩余资金分配到需要买入的股票上
    if len(need_buy_index) <= 0:
        # 如果没有需要买入的股票,则按现有持仓继续保持
        weight = torch.tensor([current_cash]).to(w.device)  # 现金权重为1
        weight = torch.cat([weight, nw], dim=0)  # 合并现金权重和股票权重 [1+M]
        return weight, nv

    # 计算需要买入的股票的权重
    need_buy_expv = torch.tensor(need_buy_expv).to(w.device)  # [S]
    if wf > 0.01:
        # 如果wf大于0.01,则将它作为温度系数使用
        need_buy_expw = F.softmax(need_buy_expv * wf, dim=-1) * available_cash  # [S]
    else:
        # 否则, 就平均分配
        need_buy_expw = available_cash / len(need_buy_index) * torch.ones(len(need_buy_index)).to(w.device)  # [S]
    for i in range(len(need_buy_index)):
        si = need_buy_index[i]
        nw[si] = need_buy_expw[i]

    weight = torch.cat([torch.tensor([ideal_cash]).to(w.device), nw], dim=0)  # 合并现金权重和股票权重 [1+M]
    return weight, nv


def make_next_w_by_pct(w, p, t1up_stop, t1down_stop, exp_value_list, maxstock, pct, wf=1000.):
    """
    根据价格预测的结果来手工生成下一个时刻的投资组合权重
    :param w: 第i-1时刻的投资组合权重 [M]
    :param p: 第i时刻的价格预测结果 [M,P,C]
    :param t1up_stop: 第1天是否已经在涨停状态 [M]
    :param t1down_stop: 第1天是否已经在跌停状态 [M]
    :param exp_value_list: 各个分类的期望收益率
    :param maxstock: 最多持仓股票数
    :param pct: 指定换仓百分比(即持仓总数的后部pct部分换成新的股票)
    :param wf: 温度系数
    """

    M, P, C = p.shape  # 这里P=1
    # 把p这个维度去掉,变成[M,C]
    p = p.squeeze(1)
    # 计算t1开盘时刻是否已经在涨停（跌停）状态, 判断的依据是t1时刻开盘价与t0收盘价的涨跌幅
    # 生成一个全0的权重向量[M] 只用于表示股票项
    nw = w.clone()  # 新的权重向量[M], 先假定与昨天的权重完全相同

    nv = [0. for _ in range(M)]  # [M] 预期收益

    # 计算所有股票的期望价值, 期望价值 = sum(预测区间所在的概率 * 区间的涨跌幅)
    # 首先要将预测结果转换为概率分布
    pp = F.softmax(p, dim=-1)  # [M,C]

    # 计算所有股票的预期收益
    expected_return = calc_expected_return(pp, exp_value_list)  # [M]
    # 对预期收益进行排序(这里用一个经验值来控制排序的范围, K=M//10)
    K = max(M // 10, maxstock)
    topK_expv, topK_index = torch.topk(expected_return, k=K, dim=0)  # 选出topK个股票的预期收益

    # STEP1: 找到所有已经持仓,且可以卖出的股票的索引
    total_hold_stock_count = 0  # 持仓股票的总数
    total_hold_stock_weight = 0.  # 持仓股票的总权重
    can_be_sold = {}  # 可以被卖出的股票的字典, 格式为{股票索引: 股票预期价值}
    for i in range(M):
        if nw[i] > 0:
            total_hold_stock_count += 1
            total_hold_stock_weight += nw[i].item()
            if t1up_stop[i] and expected_return[i] > 0.0005:
                # 股票i处于持仓状态,且今日开盘即涨停
                # 那么,只要其预测的收益为正,我们就继续持有它
                continue
            if not t1down_stop[i]:
                # 股票i处于持仓状态,且未跌停, 则可以卖出
                can_be_sold[i] = expected_return[i].item()

    # STEP2: 先执行卖出操作
    real_sold_index = []  # 真正卖出的股票的索引
    if len(can_be_sold) == 0:
        # 没有可以卖出的股票
        if total_hold_stock_count >= maxstock or total_hold_stock_weight >= 0.95:
            # 当前持仓数量或者权重已经达标,则直接返回原有投资组合, 不需要再补充股票了
            cash = torch.tensor([1.0 - total_hold_stock_weight]).to(w.device)  # 现金权重
            nw = torch.cat([cash, nw], dim=0)  # 合并现金权重和股票权重 [1+M]
            return nw, nv
    else:
        # 否则,有可卖出股票, 先执行卖出操作
        # 对can_be_sold字典进行排序,按照预期价值从小到大排序
        sorted_can_be_sold = sorted(can_be_sold.items(), key=lambda x: x[1])
        plan_to_sell_count = int(len(sorted_can_be_sold) * pct)  # 计划卖出股票的数量,即sort_can_be_sold的前pct
        if plan_to_sell_count == 0:
            # 可以执行卖出的股票数量为0
            if total_hold_stock_count >= maxstock or total_hold_stock_weight >= 0.95:
                # 当前持仓数量或者权重已经达标,则直接返回原有投资组合, 不需要再补充股票了
                cash = torch.tensor([1.0 - total_hold_stock_weight]).to(w.device)  # 现金权重
                nw = torch.cat([cash, nw], dim=0)  # 合并现金权重和股票权重 [1+M]
                return nw, nv
        else:
            # 到这里, 真正去执行卖出操作了,为了简化整个算法,暂时不考虑在稍后是否有比这只股票预期更高的股票可供买入
            for si, sv in sorted_can_be_sold[:plan_to_sell_count]:
                # 将排名靠前的这些股票卖出(因为预期收益已经按照从小到大排序了)
                total_hold_stock_count -= 1
                total_hold_stock_weight -= nw[si].item()
                nw[si] = 0.
                real_sold_index.append(si)

    # STEP3: 最后执行买入操作
    plan_to_buy_index = []  # 计划买入的股票的索引
    plan_to_buy_expv = []  # 计划买入的股票预期价值
    topK_index_list = topK_index.tolist()
    for i in range(len(topK_index_list)):
        # 遍历topK个股票, 找到预期收益最高的,且未持仓的且并不是刚刚卖掉的股票,用于补仓
        si = topK_index_list[i]
        evi = topK_expv[i].item()
        if evi < 0.001:
            # 预期收益太小, 则不再继续买入
            break
        if nw[si] > 0:
            # 本股票是昨日已在手且确定今日需要继续持仓的,跳过
            continue
        if si in real_sold_index:
            # 本股票是昨天已在手,且今日刚刚卖掉的股票,跳过
            continue

        # 到这里, 找到了一个可以买入的股票
        plan_to_buy_index.append(si)
        plan_to_buy_expv.append(evi)
        if total_hold_stock_count + len(plan_to_buy_index) > maxstock:
            # 已经达到最大持仓数, 则不再继续买入
            break

    # STEP4: 真正执行买入操作
    plb_count = len(plan_to_buy_index)
    if plb_count == 0:
        # 没有需要买入的股票
        pass
    elif plb_count == 1:
        # 只有一个需要买入的股票, 则直接买入
        # 将所有剩余权重全部分配给这只股票
        si = plan_to_buy_index[0]
        nw[si] = 1.0 - total_hold_stock_weight
        total_hold_stock_count += 1
        total_hold_stock_weight = 1.0
    else:
        # 到这里, 至少有两个需要买入的股票
        # 将预期收益转换为tensor
        buy_expv_tensor = torch.tensor(plan_to_buy_expv).to(w.device)
        # 对预期收益张量应用softmax, 得到概率分布, 并乘以剩余可用权重

        if wf > 0.01:
            # 如果wf大于0.01,则将它作为温度系数使用
            buy_pp = F.softmax(buy_expv_tensor * wf, dim=0) * (1.0 - total_hold_stock_weight)
        else:
            # 否则, 就平均分配
            available_cash = 1.0 - total_hold_stock_weight
            buy_pp = available_cash / plb_count * torch.ones(plb_count).to(w.device)  # [S]
        for i in range(plb_count):
            # 遍历需要买入的股票, 按照概率分布进行买入
            si = plan_to_buy_index[i]
            nw[si] = buy_pp[i]
        total_hold_stock_count += plb_count
        total_hold_stock_weight = 1.0

    cash = torch.tensor([1.0 - total_hold_stock_weight]).to(w.device)  # 现金权重
    nw = torch.cat([cash, nw], dim=0)  # 合并现金权重和股票权重 [1+M]
    return nw, nv


def load_json_args(json_file):
    """
    加载json文件中的参数
    :param json_file: json文件路径
    :return: 参数字典
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        base_args = json.load(file)
        print(f"{timestr()}从文件{json_file}中加载模型主要参数")
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


def cal_big_stock_count(w, smooth_weight=1e-4):
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


def make_date_from_tensor(x, yidx, midx, didx):
    # 生成日期, 这个方法只适用于老版本数据如: day54
    year = round(x[yidx].item() * 40 + 1990)
    month = round(x[midx].item() * 12 + 1)
    day = round(x[didx].item() * 31 + 1)
    return year * 10000 + month * 100 + day


def check_data_sism(si_sample, future_sample, sm_sample):
    """
    检查SI和SM数据是否匹配
    :param si_sample: SI数据 [B,M,L,C]
    :param future_sample: 未来数据 [B,M,P,C]
    :param sm_sample: SM数据 [B,G,L,C]
    """

    # 检查点1： SI数据的L天是连续的，随机取出一条来打印出L个日期
    B, M, L, C = si_sample.shape
    cb = random.randint(0, B - 1)

    print(f"-----SI数据日期B[{cb}]-----")
    for stock in range(M):
        sd = si_sample[cb, stock, :, :]  # [L,C]
        for l in range(0, L - 1, 10):
            stock_no = int(sd[l, 0])  # 股票的编码
            year = int(sd[l, 5] * 40 + 1990)
            month = int(sd[l, 6] * 12 + 1)
            day = int(sd[l, 7] * 31 + 1)
            print(f"股票{stock_no}的第{l + 1}天日期：{year}年{month}月{day}日")

    # 检查点2： 未来数据的P
    if future_sample is not None:
        B, M, P, C = future_sample.shape
        print(f"-----未来数据日期B[{cb}]-----")
        for stock in range(M):
            sd = future_sample[cb, stock, :, :]  # [L,C]
            for l in range(P):
                stock_no = int(sd[l, 0])  # 股票的编码
                year = int(sd[l, 5] * 40 + 1990)
                month = int(sd[l, 6] * 12 + 1)
                day = int(sd[l, 7] * 31 + 1)
                print(f"股票{stock_no}的第{l + 1}天日期：{year}年{month}月{day}日")

    # 检查点3：     SM数据的L天是连续的，随机取出一条来打印出L个日期
    B, G, L, C = sm_sample.shape
    print(f"-----SM数据日期B[{cb}]-----")
    for stock in range(0, G - 1, 10):
        sd = sm_sample[cb, stock, :, :]  # [L,C]
        for l in range(0, L - 1, 10):
            stock_no = int(sd[l, 0])  # 股票的编码
            year = int(sd[l, 5] * 40 + 1990)
            month = int(sd[l, 6] * 12 + 1)
            day = int(sd[l, 7] * 31 + 1)
            print(f"股票{stock_no}的第{l + 1}天日期：{year}年{month}月{day}日")


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
    if num <= 10:
        return 10
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
    # if os.path.exists(save_path):
    #     return  # 文件已存在，不需要保存
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


# from sklearn.metrics import log_loss
def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix in each of the state dict's keys."""
    return {key.replace('module.', ''): value for key, value in state_dict.items()}


def get_pm_metrics_2(ret: list, dapan: list, period='day'):
    # 1. 计算日收益率

    ret = torch.tensor(ret)  # [B,episodes]
    # 判断张量的维度是否只有1维
    if len(ret.shape) == 1:
        ret = ret.unsqueeze(0)
    B, E = ret.shape
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
    tracking_error = torch.std(excess_returns, dim=1, unbiased=True)

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
    annualized_ir = daily_ir * math.sqrt(total_period / E)

    # 打印结果
    # print(f"超额收益率: {excess_returns.tolist()}")
    # print(f"平均超额收益率: {mean_excess_return.item():.4f}")
    # print(f"跟踪误差: {tracking_error.item():.4f}")
    # print(f"日度信息比率: {daily_ir.item():.4f}")
    # print(f"年化信息比率: {annualized_ir.item():.4f}")

    # 取平均值返回
    return [mean_excess_return.mean().item(), daily_ir.mean().item(), annualized_ir.mean().item()]


def get_pm_metrics(ret: list):
    """
    input:
        ret:两日相对回报列表,[B,episodes]
    output:
        pm_metres: ARR, VOL, DD, MDD, SR, CR, SOR，[B,7]
    """
    ret = np.array(ret) - 1.0  # 装换为numpy数组

    '''ARR'''
    ARR = (np.cumprod(ret + 1.0, axis=1)[:, -1] - 1.0) / ret.shape[1] * 252

    '''VOL'''
    VOL = np.std(ret, axis=1)

    '''DD'''
    DD = np.zeros((ret.shape[0],))
    for i in range(ret.shape[0]):
        # 选择当前批次中的负收益
        negative_ret = ret[i, np.where(ret[i, :] < 0)]
        # 如果存在负收益，则计算标准差；否则，结果为0
        if negative_ret.size > 0:
            DD[i] = np.std(negative_ret)
        else:
            DD[i] = 0.0

    '''MDD'''
    iter_ret = np.cumprod(ret + 1.0, axis=1)
    peak = iter_ret[:, 0]  # 初始化每个批次的峰值为第一个值
    MDD = np.zeros((ret.shape[0],))  # 初始化最大回撤数组，形状为(b,)

    for i in range(ret.shape[1]):  # 遍历每个时间点
        value = iter_ret[:, i]
        peak = np.maximum(peak, value)  # 更新峰值
        dd = (peak - value) / peak
        MDD = np.maximum(MDD, dd)  # 更新最大回撤

    '''SR'''
    SR = 1.0 * np.mean(ret, axis=1) * np.sqrt(252) / np.std(ret, axis=1)

    '''CR'''
    CR = np.mean(ret, axis=1) * 252 / (MDD + 1e-8)

    '''SOR'''
    SOR = 1.0 * np.mean(ret, axis=1) * 252 / (DD + 1e-8)

    pm_metrics = np.vstack((ARR, VOL, DD, MDD, SR, CR, SOR)).mean(1).tolist()

    return pm_metrics


def timestr(d=0):
    # 设置时区为北京时间
    tz = pytz.timezone('Asia/Shanghai')
    # 获取当前时间并转换为北京时间
    nowtime = datetime.datetime.now(tz)

    if d == 0:
        return nowtime.strftime("[%Y-%m-%d %H:%M:%S]")
    elif d == 1:
        return nowtime.strftime("%Y%m%d")
    else:
        return nowtime.strftime("[%Y-%m-%d %H:%M:%S]")


def date_str(today, fmt, d=0):
    if today is None:
        today = time.strftime(fmt, time.localtime())
    if d != 0:
        dd = datetime.datetime.strptime(today, fmt)
        dd = dd + datetime.timedelta(days=d)
        today = dd.strftime(fmt)
    date_obj = datetime.datetime.strptime(today, fmt)
    return today, date_obj.weekday()


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


def count_trade_days(begin_date, end_date, datefmt='%Y%m%d'):
    """
    计算两个日期之间的交易日数
    :param begin_date: 开始日期
    :param end_date: 结束日期
    :param datefmt: 日期格式
    :return: 交易日数
    """
    ds = date_list(str(begin_date), str(end_date), datefmt)
    count = 0
    for d in ds:
        if is_valid_trade_date(d):
            count += 1
    return count


def trade_date_list(begin_date, end_date, datefmt='%Y%m%d'):
    # 生成从begin_date到end_date的交易日日期列表
    # begin_date YYYY-MM-YY
    # end_date YYYY-MM-YY
    retlist = [begin_date]
    if begin_date == end_date:
        return retlist
    stepdays = 1
    if end_date < begin_date:
        stepdays = -1
    bgd = datetime.datetime.strptime(begin_date, datefmt)
    while True:
        edd = bgd + datetime.timedelta(days=stepdays)
        eds = edd.date().strftime(datefmt)
        retlist.append(eds)
        if eds == end_date:
            break
        bgd = edd
    trade_list = []
    for date in retlist:
        if is_valid_trade_date(date):
            trade_list.append(date)
    return trade_list


def date_list(begin_date, end_date, datefmt):
    # begin_date YYYY-MM-YY
    # end_date YYYY-MM-YY
    retlist = [begin_date]
    if begin_date == end_date:
        return retlist
    stepdays = 1
    if end_date < begin_date:
        stepdays = -1
    bgd = datetime.datetime.strptime(begin_date, datefmt)
    while True:
        edd = bgd + datetime.timedelta(days=stepdays)
        eds = edd.date().strftime(datefmt)
        retlist.append(eds)
        if eds == end_date:
            break
        bgd = edd
    return retlist


def is_valid_date(date_str, fmtstr='%Y%m%d'):
    """
    判断日期字符串是否是有效的日期
    :param date_str:
    :param fmtstr:
    :return:
    """
    try:
        datetime.datetime.strptime(date_str, fmtstr)
        return True
    except ValueError:
        return False


def make_pre_date(oneday, lookbackwindow):
    """
    指定一个输入日期oneday, 计算其前lookbackwindow个交易日的日期
    """

    endday, _ = date_str(oneday, '%Y%m%d', -lookbackwindow * 2)
    datelist = date_list(oneday, endday, '%Y%m%d')
    tradeDays = 0
    for date in datelist:
        if is_valid_trade_date(date):
            tradeDays += 1
            if tradeDays == lookbackwindow:
                return date
    return endday


def parse_date(date):
    """
    解析日期字符串,返回年/月/日/星期几
    :param date: 日期字符串
    :return: 年/月/日/星期几
    """
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    weekday = datetime.datetime(year, month, day).weekday()
    return year, month, day, weekday


def get_next_trade_day(tradeDay, d):
    """
    输入交易日期tradeDay, 计算其后d个交易日的日期
    """

    if d == 0:
        return tradeDay
    nextday = str(tradeDay)
    count = 0
    step = 1 if d > 0 else -1
    while True:
        nextday, _ = date_str(nextday, '%Y%m%d', step)
        if is_valid_trade_date(nextday):
            count += 1
            if count == abs(d):
                return nextday


def make_date_area_list(check_area, lookbackwindow, futurewindow, mode='M'):
    """
    根据输入的check_area生成对应的日期区间列表
    :param check_area: 输入的日期区间, 格式为[YYYYMMDD,YYYYMMDD]
    :param lookbackwindow: 回溯窗口大小
    :param futurewindow: 未来窗口大小
    :param mode: 日期区间模式, 'M'代表月
    """

    ret_list = []
    start_year = int(check_area[0][:4])
    start_month = int(check_area[0][4:6])
    end_year = int(check_area[1][:4])
    end_month = int(check_area[1][4:6])
    assert mode == 'M', '暂时只支持月模式'
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                break
            start_date = f"{year}{month:02d}01"
            cfd = make_pre_date(start_date, lookbackwindow + 1)
            cld = get_next_trade_day(start_date, futurewindow)
            ret_list.append([start_date, cfd, cld])

    return ret_list


def is_valid_trade_date(date_str, fmtstr='%Y%m%d'):
    """
    判断日期字符串是否是有效的交易日期
    :param date_str:
    :param fmtstr:
    :return:
    """
    if not is_valid_date(date_str, fmtstr):
        return False
    weekday = week_day(date_str, fmtstr)
    if weekday == 5 or weekday == 6:
        return False
    return True


def week_day(date_str, fmtstr='%Y%m%d'):
    """
    输入日期字符串和格式，返回星期几（0-6，0代表星期一）
    :param date_str:  输入日期str
    :param fmtstr:  日期格式
    :return: int
    """
    date_obj = datetime.datetime.strptime(date_str, fmtstr)
    return date_obj.weekday()


def append_train_info(fn, info):
    f = open(fn, "a+", encoding="utf-8-sig")
    print(f"{timestr()},{info.replace(',', '_')}", file=f)
    f.close()


def summary_train_info(args, verbose=True):
    if verbose:
        print('本次训练全部初始参数:')
    ns_dict = vars(args)
    model_info = ''
    # 遍历字典并打印所有属性和它们的值
    for key, value in ns_dict.items():
        if isinstance(value, (int, float, str, bool, list)):
            if str(value) == '':
                continue
            if verbose:
                print(f'{key}: {value}')
            model_info = model_info + f" {key}:{value}"
    return model_info


def make_change_rate_map_for_pretrain(rmin, rmax, class_num):
    """
    当任务为价格落位区间分类任务时, 需要对g_change_rate_map进行初始化
    """
    boundaries = np.linspace(rmin, rmax, class_num - 1)
    change_rate_map = []
    for x in boundaries:
        change_rate_map.append(round(x, 4))
    return change_rate_map


def make_device_list(use_gpu_str):
    if use_gpu_str == 'A':
        # 使用所有的GPU
        devlist = list(range(torch.cuda.device_count()))
        return devlist
    elif use_gpu_str == 'a' and torch.cuda.device_count() > 1:
        # 使用除最后一个GPU外的所有GPU
        devlist = list(range(torch.cuda.device_count() - 1))
        return devlist
    else:
        devlist = []
        d = use_gpu_str.split(',')
        for dv in d:
            devlist.append(int(dv))
        return devlist


class EarlyStopping:
    def __init__(self, target='min', patience=7, verbose=False, save_path='./output/', mark='si'):
        self.target = target  # 目标['min','max','both'] min: 指loss越小越好，max:指acc越大越好，both:同时考虑loss和acc
        self.patience = patience  # 连续多少次未优化,则Earlystop
        self.verbose = verbose  # 是否打印输出信息
        self.counter = 0  # 持续未优化次数-计数器
        self.best_loss = None  # 最优结果
        self.best_acc = 0.  # 最优结果
        self.early_stop = False  # 是否结束
        self.save_path = save_path
        self.mark = mark
        self.save_epoch = False  # 是否在每次优化后保存模型(默认为False, 会覆盖同名文件), 若为True, 则会在优化后保存不同的文件(用epoch号命名)

    def update_min(self, check_loss):
        better = False
        prev_Value = self.best_loss
        if self.best_loss is None:
            prev_Value = float('inf')
            self.best_loss = check_loss
            better = True
        elif check_loss < self.best_loss:
            prev_Value = self.best_loss
            self.best_loss = check_loss
            better = True
        return better, prev_Value

    def update_max(self, check_acc):
        better = False
        prev_Value = self.best_acc
        if check_acc > self.best_acc:
            prev_Value = self.best_acc
            self.best_acc = check_acc
            better = True
        return better, prev_Value

    def save_model(self, epochNO, model):
        if self.save_epoch:
            mdfile = os.path.join(self.save_path, f'{self.mark}_epoch{epochNO}.pth')
        else:
            mdfile = os.path.join(self.save_path, f'{self.mark}.pth')
        torch.save(model.state_dict(), mdfile)

    def __call__(self, check_loss, check_acc, epochNO, model, rank=0):
        if self.target == 'min':
            get_better, prev_Value = self.update_min(check_loss)
            if get_better:
                self.counter = 0
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: loss improved from {prev_Value:.6f} to {check_loss:.6f} at epoch {epochNO}')
            else:
                self.counter += 1
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: loss {check_loss:6f} not improved from {prev_Value:.6f} at epoch {epochNO}')
                    print(f'{timestr()}EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            es_flag = get_better
        elif self.target == 'max':
            get_better, prev_Value = self.update_max(check_acc)
            if get_better:
                self.counter = 0
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: acc improved from {prev_Value:.6f} to {check_acc:.6f} at epoch {epochNO}')
            else:
                self.counter += 1
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: acc {check_acc:6f} not improved from {prev_Value:.6f} at epoch {epochNO}')
                    print(f'{timestr()}EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            es_flag = get_better
        else:
            # self.target == 'both':
            get_better_loss, prev_Value_loss = self.update_min(check_loss)
            get_better_acc, prev_Value_acc = self.update_max(check_acc)
            if get_better_loss:
                self.counter = 0
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: loss improved from {prev_Value_loss:.6f} to {check_loss:.6f} at epoch {epochNO}')
            elif get_better_acc:
                self.counter = 0
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: acc improved from {prev_Value_acc:.6f} to {check_acc:.6f} at epoch {epochNO}')
            else:
                self.counter += 1
                if self.verbose and rank == 0:
                    print(
                        f'{timestr()}Validation: loss {check_loss:6f} not improved from {prev_Value_loss:.6f} at epoch {epochNO}')
                    print(
                        f'{timestr()}Validation: acc {check_acc:6f} not improved from {prev_Value_acc:.6f} at epoch {epochNO}')
                    print(f'{timestr()}EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            es_flag = get_better_loss or get_better_acc
        if es_flag and rank == 0:
            self.save_model(epochNO, model)
        return es_flag


class EarlyStoppingV2:
    def __init__(self, target_name: list, target_type: list, patience=7, verbose=False, save_path='./output/',
                 mark='si', save_every_epoch=False, save_every_better=False):
        self.target_name = target_name  # 目标['CELoss','ACC1','F1Score']
        self.target_type = target_type  # 目标类型['min','max','max']
        assert len(target_name) == len(target_type) and len(target_type) > 0, "target_name, target_type的长度不一致"
        # 这里的target可以是一个列表，表示同时考虑多个指标，比如['min','max'], 注意后续传入的指标也需按相同的顺序传入
        self.CheckNum = len(target_name)  # 目标个数
        self.patience = patience  # 连续多少次未优化,则Earlystop
        self.verbose = verbose  # 是否打印输出信息
        self.counter = 0  # 持续未优化次数-计数器
        self.best_result = []  # 最优结果
        self.early_stop = False  # 是否结束
        self.save_path = save_path
        self.mark = mark
        self.save_every_epoch = save_every_epoch  # 是否在每个epoch都保存模型
        self.save_every_better = save_every_better  # 是否在每次优化后保存模型(默认为False, 会覆盖同名文件), 若为True, 则会在优化后保存不同的文件(用epoch号命名)

    def update_target(self, index, check_result):
        """
        检测一项指标是否有提升
        :param index: 待检测指标的索引
        :param check_result: 待检测指标的值
        :return: 返回是否有提升，以及之前的最优值
        """

        target_type = self.target_type[index]  # 目标类型
        prev_Value = self.best_result[index]  # 之前的最优值
        if target_type == 'min':
            # 目标类型为最小值
            if check_result < prev_Value:
                # 新的值小于之前的最优值
                self.best_result[index] = check_result
                return True, prev_Value
            else:
                # 新的值大于等于之前的最优值
                return False, prev_Value
        elif target_type == 'max':
            # 目标类型为最大值
            if check_result > prev_Value:
                # 新的值大于之前的最优值
                self.best_result[index] = check_result
                return True, prev_Value
            else:
                # 新的值小于等于之前的最优值
                return False, prev_Value
        else:
            raise ValueError("target_type只能是'min'或'max'")

    def save_model_every_epoch(self, epochNO, model):
        mdfile = os.path.join(self.save_path, f'{self.mark}_epoch{epochNO}.pth')
        print(f'{timestr()}=====Save model to {mdfile}=====')
        torch.save(model.state_dict(), mdfile)

    def save_model_better(self, epochNO, model, every_epoch=False):
        if self.save_every_better:
            mdfile = os.path.join(self.save_path, f'{self.mark}_better_epoch{epochNO}.pth')
        else:
            # 覆盖保存最好的结果
            mdfile = os.path.join(self.save_path, f'{self.mark}.pth')
        print(f'{timestr()}=====Save model to {mdfile}=====')
        torch.save(model.state_dict(), mdfile)
        if every_epoch:
            pass

    def __call__(self, result_to_check: list, epochNO, model, rank=0):
        """q
        :param result_to_check: 待检查的指标列表，与self.target的长度一致
        :param epochNO: 当前epoch
        :param model: 模型
        :param rank: 进程号
        :return: 返回early_stop的标志
        """

        assert len(result_to_check) == self.CheckNum, "result_to_check的长度与self.target的长度不一致"
        verbose = self.verbose and rank == 0  # 是否打印输出信息

        get_better = False
        if len(self.best_result) == 0:
            # 第一次调用，初始化best_result
            self.best_result = copy.deepcopy(result_to_check)
            get_better = True
            if verbose:
                for i in range(self.CheckNum):
                    target_name = self.target_name[i]  # 目标名称
                    check_result = result_to_check[i]  # 待检测指标
                    print(
                        f'{timestr()}Validation: {target_name} improved to {check_result:.6f} at epoch {epochNO}')
        else:
            # 后续调用，检测是否有指标提升
            for i in range(self.CheckNum):
                target_name = self.target_name[i]  # 目标名称
                check_result = result_to_check[i]  # 待检测指标
                c_better, prev_Value = self.update_target(i, check_result)
                if c_better:
                    # 当前这个指标有提升
                    if verbose:
                        print(
                            f'{timestr()}Validation: {target_name} improved from {prev_Value:.6f} to {check_result:.6f} at epoch {epochNO}')
                else:
                    # 当前这个指标没有提升
                    if verbose:
                        print(
                            f'{timestr()}Validation: {target_name} {check_result:6f} not improved from {prev_Value:.6f} at epoch {epochNO}')
                get_better = get_better or c_better
        if get_better:
            # 本轮有指标提升
            self.counter = 0  # 计算器清零
            if rank == 0:
                # 保存模型参数
                self.save_model_better(epochNO, model)
        else:
            # 所有待检测指标都没有提升
            self.counter += 1
            if verbose:
                print(f'{timestr()}EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        if rank == 0 and self.save_every_epoch:
            # 每个epoch都保存模型
            self.save_model_every_epoch(epochNO, model)

        return get_better


class Analysis:
    def __init__(self, pred_len, num_class_value, zxd_list=None):
        """
        :param pred_len  int  表示预测天数
        :param num_class_value float[] 浮点数list,表示分类的数据点位, 真实的类别是len(num_class)+1
        :param zxd_list float[] 浮点数list,表示要记录的置信度几个档位
        """
        if zxd_list is None:
            zxd_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pred_len = pred_len
        self.num_class_value = num_class_value
        self.num_class = len(num_class_value) + 1
        self.zxd_list = sorted(zxd_list)  # 置信度按从小到大排序

        self.totalCount = 0  # 总样本数
        self.accTotal = 0  # 总命中数
        self.accTotal_TOP2 = 0  # 总命中TOP2数
        self.countDay = [0 for _ in range(self.pred_len)]  # 统计每天的样本数
        self.accDay = [0 for _ in range(self.pred_len)]  # 统计每天命中数
        self.cateCountReal = [0 for _ in range(self.num_class)]  # 统计真实的区间分布
        self.cateCountPred = [0 for _ in range(self.num_class)]  # 统计预测的区间分布
        self.cateCountPredOK = [0 for _ in range(self.num_class)]  # 统计预测各区间正确数

        # 按照置信度档位,分别统计分天数据
        self.day_analysis = {}  # key为置信度, value=(各天高置信度计数[],各天高置信且正确计数[])
        for d in self.zxd_list:
            d_pred = [0 for _ in range(self.pred_len)]
            d_pred_right = [0 for _ in range(self.pred_len)]
            self.day_analysis[d] = (d_pred, d_pred_right)
        # 按照置信度档位,分别统计分区间数据
        self.class_analysis = {}  # key为置信度, value=(各区间高置信度计数[],各区间高置信且正确计数[])
        for d in self.zxd_list:
            c_pred = [0 for _ in range(self.num_class)]
            c_pred_right = [0 for _ in range(self.num_class)]
            self.class_analysis[d] = (c_pred, c_pred_right)

        # 2024-04-20增加混淆矩阵
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        # self.batch_losses_log = []

    def reset(self):
        self.totalCount = 0  # 总样本数
        self.accTotal = 0  # 总命中数
        self.accTotal_TOP2 = 0  # 总命中TOP2数
        self.countDay = [0 for _ in range(self.pred_len)]  # 统计每天的样本数
        self.accDay = [0 for _ in range(self.pred_len)]  # 统计每天命中数
        self.cateCountReal = [0 for _ in range(self.num_class)]  # 统计真实的区间分布
        self.cateCountPred = [0 for _ in range(self.num_class)]  # 统计预测的区间分布
        self.cateCountPredOK = [0 for _ in range(self.num_class)]  # 统计预测各区间正确数

        # 按照置信度档位,分别统计分天数据
        self.day_analysis = {}  # key为置信度, value=(各天高置信度计数[],各天高置信且正确计数[])
        for d in self.zxd_list:
            d_pred = [0 for _ in range(self.pred_len)]
            d_pred_right = [0 for _ in range(self.pred_len)]
            self.day_analysis[d] = (d_pred, d_pred_right)
        # 按照置信度档位,分别统计分区间数据
        self.class_analysis = {}  # key为置信度, value=(各区间高置信度计数[],各区间高置信且正确计数[])
        for d in self.zxd_list:
            c_pred = [0 for _ in range(self.num_class)]
            c_pred_right = [0 for _ in range(self.num_class)]
            self.class_analysis[d] = (c_pred, c_pred_right)

        # 2024-04-20增加混淆矩阵
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        # self.batch_losses_log = []

    def count(self, label, pred, padding_flag=None):
        """
        :param label: 真实标签[b,pred_len],
        :param pred:  模型预测数据[b,pred_len,num_class],需要进行softmax处理才能有置信度数据
        :param padding_flag:  预测数据是否有padding,如果有padding,则不统计该样本 [b,pred_len]
        """

        label = label.to('cpu')
        pred = F.softmax(pred, dim=-1)
        max_values, max_indices = torch.topk(pred, 2, dim=-1)
        max_values = max_values.to('cpu')
        max_indices = max_indices.to('cpu')
        if padding_flag is not None:
            padding_flag = padding_flag.to('cpu')

        B, L, C = pred.shape  # [B,5,10]
        for row in range(B):
            for day in range(L):
                if padding_flag is not None and padding_flag[row][day]:
                    # 该样本为padding,不统计
                    continue
                rL = int(label[row][day])  # 真实的价格所在区间
                pP, pL = float(max_values[row, day, 0]), int(max_indices[row, day, 0])  # pP为TOP1的置信度, pL为TOP1预测的价格区间
                pP2, pL2 = float(max_values[row, day, 1]), int(
                    max_indices[row, day, 1])  # pP2为TOP2的置信度, pL2为TOP2预测的价格区间
                self.totalCount += 1  # 总计数+1
                self.countDay[day] += 1  # 分天计数+1
                self.cateCountReal[rL] += 1  # 真实区间+1
                self.cateCountPred[pL] += 1  # 预测区间+1
                match = 0

                # 2024-04-20增加混淆矩阵统计
                self.confusion_matrix[rL][pL] += 1

                if pL == rL:
                    self.cateCountPredOK[pL] += 1  # pL区间预测正确+1
                    self.accTotal += 1  # 真实总命中数+1
                    self.accTotal_TOP2 += 1  # 真实总命中TOP2数+1
                    self.accDay[day] += 1  # 当日命中+1
                    match = 1
                elif pL2 == rL:
                    self.accTotal_TOP2 += 1  # 真实总命中TOP2数+1

                # 下面统计各置信度之下的天/区间数据
                for checkPP in self.zxd_list:
                    if pP < checkPP:
                        break
                    # pp>=checkPP
                    # 分天数据更新
                    d_pred, d_pred_right = self.day_analysis[checkPP]
                    d_pred[day] += 1  # checkPP置信度---col天计数
                    d_pred_right[day] += match  # checkPP置信度---col天正确计数

                    # 分区间数据更新
                    c_pred, c_pred_right = self.class_analysis[checkPP]
                    c_pred[pL] += 1  # checkPP置信度---预测区间计数
                    c_pred_right[pL] += match  # checkPP置信度---预测区间正确计数

    def get_class_name(self):
        r = []
        for x in self.num_class_value:
            x = round(x * 100, 2)
            if x <= 1e-8:
                r.append('-<=' + str(abs(x)))
            else:
                r.append("+<=" + str(x))
        x = self.num_class_value[-1]
        x = round(x * 100, 2)
        r.append("+>" + str(x))
        return r

    def get_f1_score(self, outstr=False):
        """
        计算F1分数
        """
        # 1、计算各类别的准确率（实际为精确率），即预测为C类且实际为C类的数据的比例
        # 2、计算各类别的召回率，即实际为C类的数据中，预测为C类的数据的比例
        # 3、计算各类别的F1分数，即 2PR/(P+R)
        # 4、计算总体的F1分数，即各类别F1分数的平均值

        f1_core_list = []
        for d in range(self.num_class):
            rec = self.cateCountPredOK[d] / (self.cateCountReal[d] + 1e-5)
            acc = self.cateCountPredOK[d] / (self.cateCountPred[d] + 1e-5)
            f1_core = 2 * rec * acc / (rec + acc + 1e-8)
            f1_core_list.append(round(f1_core, 5))
        if outstr:
            return ','.join(str(num) for num in f1_core_list)
        else:
            d_f1_core = sum(f1_core_list) / len(f1_core_list)
            return round(d_f1_core, 5)

    def get_total_acc(self):
        """
        计算总准确率(TOP1准确率
        """
        v = round(self.accTotal / (self.totalCount + 1e-5) * 100, 4)
        return v

    def get_total_acc_top2(self):
        """
        计算总准确率(TOP2准确率)
        """
        v = round(self.accTotal_TOP2 / (self.totalCount + 1e-5) * 100, 4)
        return v

    def get_kappa(self):
        """
        计算kappa系数
        """
        total = np.sum(self.confusion_matrix)
        p0 = np.trace(self.confusion_matrix) / total
        pe = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / (total * total)
        kappa_score = (p0 - pe) / (1 - pe)
        return kappa_score

    def get_mcc(self):
        sum_rows = np.sum(self.confusion_matrix, axis=1)
        sum_cols = np.sum(self.confusion_matrix, axis=0)
        TP = np.diag(self.confusion_matrix)
        TN = np.sum(self.confusion_matrix) - (sum_rows + sum_cols - TP)
        FP = sum_cols - TP
        FN = sum_rows - TP

        numerator = TP * TN - FP * FN
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc_scores = numerator / (denominator + 1e-8)

        return np.mean(mcc_scores)

    def printInfo(self, save_to_filename='', model_info=''):
        print(f"================分析统计=================")
        print(f"=====1.各天准确率分析(预测正确/预测总数)=====")
        for d in range(self.pred_len):
            v = round(self.accDay[d] / (self.countDay[d] + 1e-5) * 100, 2)
            print(f"第{d + 1}天:{self.accDay[d]}/{self.countDay[d]}={str(v)}%")
        v = round(self.accTotal / (self.totalCount + 1e-5) * 100, 2)
        v2 = round(self.accTotal_TOP2 / (self.totalCount + 1e-5) * 100, 2)
        print(f"TOP1准确率:{str(v)}%,TOP2准确率:{str(v2)}%,F1Score:{self.get_f1_score()}")
        print(f"=====2.区间分布分析=====")
        rd, pd, rdOK, pdOK = [], [], [], []
        for d in range(self.num_class):
            rd.append(round(self.cateCountReal[d] / (self.totalCount + 1e-5), 4))
            pd.append(round(self.cateCountPred[d] / (self.totalCount + 1e-5), 4))
            rdOK.append(str(round(self.cateCountPredOK[d] / (self.cateCountReal[d] + 1e-5) * 100, 2)) + '%')
            pdOK.append(str(round(self.cateCountPredOK[d] / (self.cateCountPred[d] + 1e-5) * 100, 2)) + '%')
        print(f"价格变化区间---:{liststr(self.get_class_name())}")
        print(f"真实区间分布pct:{liststr(rd)}")
        print(f"预测区间分布pct:{liststr(pd)}")
        # loss = F.cross_entropy(torch.tensor(rd).unsqueeze(0), torch.tensor(pd).unsqueeze(0))
        print(f"真实区间视角acc:{liststr(rdOK)}")
        print(f"预测区间视角acc:{liststr(pdOK)}")
        # print(f"分布loss: {loss}")

        print(f"=====3.各置信度档位分析=====")
        output_file_info = {}
        for checkPP in self.zxd_list:
            info_a, info_b = [], []
            output_file_info[checkPP] = (info_a, info_b)
            info_a.append(checkPP)
            info_b.append('')
            print(f"*****置信度>={checkPP}*****")
            print(f"-----分天数据-----")
            d_pred, d_pred_right = self.day_analysis[checkPP]
            tt, th, tr = 0, 0, 0
            pa, pb = ["高置信度占比-->"], ["高置信中的准确率-->"]
            for day in range(self.pred_len):
                p = round(d_pred[day] / (self.countDay[day] + 1e-5) * 100, 2)
                v = round(d_pred_right[day] / (d_pred[day] + 1e-5) * 100, 2)
                tt += self.countDay[day]
                th += d_pred[day]
                tr += d_pred_right[day]
                print(f"第{day + 1}天:高置信度占比:{str(p)}%,命中率:{str(v)}%")
                pa.append(str(p) + '%')
                pb.append(str(v) + '%')
            p = round(th / (tt + 1e-5) * 100, 2)
            v = round(tr / (th + 1e-5) * 100, 2)
            print(f"总体:高置信度占比:{str(p)}%,命中率:{str(v)}%")
            info_a.append(str(p) + '%')
            info_b.append('')
            info_a.append(str(v) + '%')
            info_b.append('')
            info_a.extend(pa)
            info_b.extend(pb)
            info_a.append("区间高置信度占比-->")
            info_b.append("高置信中的准确率-->")

            print(f"-----分区间数据-----")
            c_pred, c_pred_right = self.class_analysis[checkPP]
            pa, pb = [], []
            for cls in range(self.num_class):
                pa.append(str(round(c_pred[cls] / (self.cateCountPred[cls] + 1e-5) * 100, 2)) + '%')
                pb.append(str(round(c_pred_right[cls] / (c_pred[cls] + 1e-5) * 100, 2)) + '%')
            print(f"价格变化区间------:{liststr(self.get_class_name())}")
            print(f"预测区间高置信度占比:{liststr(pa)}")
            print(f"预测区间高置信度命中:{liststr(pb)}")
            info_a.extend(pa)
            info_b.extend(pb)

        if save_to_filename != '':
            # 保存到文件
            f = open(save_to_filename, "a+", encoding="utf-8-sig")
            tms = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
            print(f"{tms},{model_info.replace(',', '_')}", file=f)
            print(
                f"Top1准确率:,{self.get_total_acc()}%,Top2准确率:,{self.get_total_acc_top2()}%,F1Score:,{self.get_f1_score()},Kappa:,{self.get_kappa()},MCC:,{self.get_mcc()}",
                file=f)
            # 标题栏
            print(f"置信度标准,高置信度占比,高置信中的准确率,分天数据-->", file=f, end='')
            for d in range(self.pred_len):
                print(f",D{d + 1}", file=f, end='')
            print(f",分区间数据-->", file=f, end='')
            cn = self.get_class_name()
            for cls in cn:
                print(f",{cls}", file=f, end='')
            print(f"", file=f)  # 换行
            for checkPP in self.zxd_list:
                pa, pb = output_file_info[checkPP]
                for x in pa:
                    print(f"{x},", file=f, end='')
                print("", file=f)
                for x in pb:
                    print(f"{x},", file=f, end='')
                print("", file=f)
            # 各区间F1分数
            print(f"各区间F1score,,,,,", file=f, end='')
            print(',' * self.pred_len, file=f, end='')
            print(f"{self.get_f1_score(outstr=True)}", file=f)

            print(f"各区间真实分布,,,,,", file=f, end='')
            print(',' * self.pred_len, file=f, end='')
            print(','.join('{:.2f}%'.format(num * 100) for num in rd), file=f)

            print(f"各区间预测分布,,,,,", file=f, end='')
            print(',' * self.pred_len, file=f, end='')
            print(','.join('{:.2f}%'.format(num * 100) for num in pd), file=f)

            print(f"各区间召回率,,,,,", file=f, end='')
            print(',' * self.pred_len, file=f, end='')
            print(','.join(rdOK), file=f)

            print(f"各区间准确率,,,,,", file=f, end='')
            print(',' * self.pred_len, file=f, end='')
            print(','.join(pdOK), file=f)

            print(f"", file=f)  # 换行
            f.close()
