# 演示如何使用模型进行预测
# 当前的模型为：PM_SMV2模型(Ex26_2日频框架 + 5分钟数据分支, 见 ../pmdayV2/pm_model_ms.py)
# 基于Ex26_2的daily_infer_0625.py, 增加: 读取T-1日及之前ms_time_step天的5分钟数据并传入模型
import argparse
import json
import math
import os
import random
import time
import cvxpy as cp
import numpy as np
import torch
import utils_date as DT
import common_utils as CM
import utils as UTILS
import sys
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)  # common_module
sys.path.append(os.path.join(_ROOT, 'pmdayV2'))  # pm_model_ms / csrank

from pm_model_ms import PortfolioModel_SMV2 as PMVA
from minute_data import MinuteBinReader, DEFAULT_MS_DATA_PATH

def active_weight_portfolio(
    A: torch.Tensor,
    B: torch.Tensor,
    active_scale: float = 0.5,
    min_weight: float = 5e-4,
) -> torch.Tensor:
    """
    将模型权重解释为相对指数的主动权重。

    Parameters
    ----------
    A : torch.Tensor
        Shape = (1, M+1)
        第一列为现金，后面为模型股票权重。

    B : torch.Tensor
        Shape = (1, M)
        指数股票权重（无现金）。

    active_scale : float
        主动权重强度。
        0 -> 完全指数
        1 -> 完全采用模型主动权重

    min_weight : float
        股票最小持仓，低于该值直接置0。

    Returns
    -------
    torch.Tensor
        Shape = (1, M+1)
    """

    if A.ndim != 2 or A.shape[0] != 1:
        raise ValueError("A must have shape (1, M+1).")

    if B.ndim != 2 or B.shape[0] != 1:
        raise ValueError("B must have shape (1, M).")

    if A.shape[1] != B.shape[1] + 1:
        raise ValueError("A should contain one extra cash column.")

    # 保留现金
    cash = A[:, :1]

    # 股票部分
    model_stock = A[:, 1:]

    # 原股票总仓位（1-cash）
    target_stock_weight = model_stock.sum(dim=1, keepdim=True)

    # 模型权重 -> 主动权重（和为0）
    active = model_stock - model_stock.mean(dim=1, keepdim=True)

    # 控制主动权重大小
    active = active * active_scale

    # 加到指数权重
    stock = B + active

    # 非负
    stock = torch.clamp(stock, min=0.0)

    # 去掉碎仓
    stock[stock < min_weight] = 0.0

    # 极端情况保护
    if stock.sum() <= 1e-12:
        stock = B.clone()

    # 保持原股票仓位（现金保持不变）
    stock = stock / stock.sum(dim=1, keepdim=True)
    stock = stock * target_stock_weight

    # 拼回现金
    out = torch.cat([cash, stock], dim=1)

    return out

def optimize_portfolio_with_cash(A_tensor, B_tensor, alpha=0.5, w_min=0.0005):
    """
    带现金项的凸优化组合优化器

    :param A_tensor: 形状 (1, 1+M)，第 0 列为现金，后面 M 列为股票
    :param B_tensor: 形状 (1, M) 或 (M,)，全为股票成分股权重
    :param alpha: 模型偏向系数 (0 ~ 1)
    :param w_min: 股票最低仓位截断阈值
    :return: 形状 (1, 1+M) 的最终权重张量
    """
    device = A_tensor.device
    dtype = A_tensor.dtype

    # 1. 统一 B 的形状，并在最前面插入“现金”列 (权重为0)
    if B_tensor.dim() == 1:
        B_tensor = B_tensor.unsqueeze(0)  # 确保是二维 (1, M)

    # 构造一个 1x1 的全 0 张量代表现金
    cash_column = torch.zeros((B_tensor.shape[0], 1), dtype=dtype, device=device)

    # 将现金列拼接在 B_tensor 最前面，此时 B_padded 形状也变成 (1, 1+M)
    B_padded = torch.cat([cash_column, B_tensor], dim=1)

    # 2. 转为 Numpy 压平，方便 CVXPY 处理
    A = A_tensor.detach().cpu().numpy().flatten()
    B = B_padded.detach().cpu().numpy().flatten()

    N = len(A)  # N = 1 + M

    # 归一化 (确保两者和都为 1.0)
    A = A / (np.sum(A) + 1e-8)
    B = B / (np.sum(B) + 1e-8)

    # ==========================================
    # 阶段一：CVXPY 连续优化
    # ==========================================
    w = cp.Variable(N)

    objective = cp.Minimize(
        alpha * cp.sum_squares(w - A) +
        (1 - alpha) * cp.sum_squares(w - B)
    )

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"求解失败: {prob.status}")

    w_opt = w.value

    # ==========================================
    # 阶段二：截断与资金再分配
    # ==========================================
    # 分离出现金和股票
    cash_weight = w_opt[0]
    stock_weights = w_opt[1:]

    # 【截断逻辑】：只对股票进行最低仓位限制，现金不需要
    invalid_mask = (stock_weights > 0) & (stock_weights < w_min)

    # 计算被截断掉的“碎股资金”总额
    truncated_funds = np.sum(stock_weights[invalid_mask])

    # 将碎股清零
    stock_weights[invalid_mask] = 0.0

    # 【资金再分配逻辑】：你可以选择 方案A 或 方案B (这里默认方案A)

    # 方案 A：重新按比例归一化全局（股票同比例放大）
    w_opt[0] = cash_weight
    w_opt[1:] = stock_weights
    w_opt = w_opt / np.sum(w_opt)

    # 方案 B：直接把砍掉的碎股资金全部扔回现金池里 (取消上面三行注释下面两行)
    # w_opt[1:] = stock_weights
    # w_opt[0] = cash_weight + truncated_funds

    # ==========================================
    # 阶段三：打包回 Tensor
    # ==========================================
    w_final_tensor = torch.tensor(w_opt, dtype=dtype, device=device).unsqueeze(0)

    return w_final_tensor


def load_ss_model(ssmodel_path, device):
    model_args = CM.load_args_from_file(ssmodel_path)
    ss_model = PS_Model(model_args)
    state_dict_pre = torch.load(ssmodel_path, map_location='cpu', weights_only=True)
    ss_model.load_state_dict(state_dict_pre, strict=False)
    # 冻结模型参数
    for param in ss_model.parameters():
        param.requires_grad = False
    ss_model.to(device)
    ss_model.eval()
    print(f"****SS评价模型参数加载成功:{ssmodel_path}****")
    return ss_model


def build_model(model_args, model_name, modelType):
    """
    根据模型文件名, 加载模型参数, 实例化模型
    :param model_args: 模型参数
    :param model_name: 模型文件名
    :param modelType: 模型类型[A,B,SUB01,SUPER]
    :return: 模型实例
    """

    baseName = os.path.basename(model_name)
    if baseName.upper().startswith("SCP_"):
        # 加载scripted_model, 不需要模型源文件
        model = torch.jit.load(model_name)
        model.eval()
        print(f"无模型源码方式,加载scripted_model:{model_name}成功")
        return model, True
    else:
        modelType = str(modelType).upper()
        if modelType == 'RON':
            print("回测模型为-Ron mask_mf 502")
            model = PMRon(model_args)
        elif modelType == 'RONWIND':
            print("回测模型为-RonWind mask_mf 503")
            model = PMRonWind(model_args)
        elif modelType == 'A':
            print("回测模型为:pmva")
            model = PMVA(model_args)
        else:
            raise ValueError(f"不支持模型类型{modelType}")

        state_dict_pre = torch.load(model_name, map_location='cpu', weights_only=True)
        ret = model.load_state_dict(state_dict_pre, strict=False)
        print(f"带模型源码方式,加载模型参数{model_name}:", ret)
        model.eval()

        # print("使用torch.compile进行加速")
        # ts = time.time()
        # model = torch.compile(model, dynamic=True)
        # print(f"torch.compile耗时:{time.time() - ts}")

        return model, False


def load_model_args(load_model_file):
    if not os.path.exists(load_model_file):
        raise FileNotFoundError(f"模型文件{load_model_file}不存在")

    # 加载模型基础参数
    base_args = CM.load_base_args(load_model_file)
    if base_args is None:
        raise ValueError(f"模型参数文件加载失败,{load_model_file}")

    args = argparse.Namespace()
    for k, v in base_args.items():
        setattr(args, k, v)
    # for base_arg_name in base_args.keys():
    #     if base_arg_name in CM.model_base_args:
    #         setattr(args, base_arg_name, base_args[base_arg_name])
    # if "maxw" not in base_args.keys():
    #     setattr(args, "maxw", 0.)
    # if "cash_layer" not in base_args.keys():
    #     setattr(args, "cash_layer", 0)
    if "priceRelatedField" not in base_args.keys():
        # 加个补丁,老的basepm文件中没有保存这个字段,而现在有需要了, 需要加个默认值 对应的是104/106通道的价格类字段列表
        # 新的304通道模型, 已经在basepm中保存了这个字段了,就不需要了
        setattr(args, "priceRelatedField", ['open', 'close', 'high', 'low', 'pre_close', 'avg_price', '35_close',
                                            '60m_avg_price',
                                            'daily_avg_price', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'])
    # # print(args)
    return args


def make_infer_data(args, tradeDay, look_back_days, base_price_type='open', skipLoss=1., last_weight=None, withBJ=True,
                    shuffle=0):
    """
    读取推理所需要的原始数据
    :param args
    :param tradeDay: 交易日期
    :param look_back_days: 回看窗口的长度
    :param base_price_type: 交易日当天的基准价格类型, 'open'表示开盘价, '935'表示9:35收盘价
    :param skipLoss: 跳过开盘跌幅大于skipLoss的股票
    :param last_weight: 上一天的权重向量,如果为None,则使用全现金持仓
    :param withBJ: 是否包含北交所股票
    :param shuffle:表示按股票代码排序  1:表示随机排序  这个参数对selfAttn模型产生重大影响
    """

    # STEP1: 获取全部的股票代码列表
    all_stocklist, abs_all = UTILS.get_stock_list(UTILS.g_stock_info_path, withBJ=withBJ)

    sc = len(all_stocklist)
    if sc <= 100:
        raise ValueError(f"股票字典表中的股票数{sc}过少, 请检查股票字典表是否正确")
    if shuffle == 1:
        random.shuffle(all_stocklist)
    print(f"股票字典表中的股票数:{sc}")

    # ADD 20251125 排除大模型已经给出重大负面信息的股票
    all_stocklist, rmv = UTILS.remove_llm_bad(all_stocklist, tradeDay)
    print(f"大模型负面信息排除{rmv}只股票,剩余数量{len(all_stocklist)}")

    # ADD 20260107 排除过去15个交易日内曾经挂过'退'标的股票
    all_stocklist, rmv = UTILS.remove_quit_stock(all_stocklist, tradeDay)
    if rmv > 0:
        print(f"--QUIT--排除退标股票{rmv}只,剩余数量{len(all_stocklist)}")

    # STEP2: 从文本文件中读取原始数据
    pre_tradeDay = DT.get_next_trade_day(tradeDay, -1)
    print(
        f"{DT.timestr()}tradeDay{tradeDay}-读取最后日期为{pre_tradeDay}的连续{look_back_days}天数据{len(all_stocklist)}...")
    timeStart = time.time()
    timeStepsData = UTILS.get_last_timesteps_from_path(csv_path=UTILS.g_csv_path, timesteps=look_back_days,
                                                       last_date=pre_tradeDay, stocklist=all_stocklist)
    if timeStepsData is None:
        raise ValueError("没有足够的回看窗口数据")
    else:
        print('读取数据成功,数据维度:', timeStepsData.shape)
    print(f"读取数据耗时:{(time.time() - timeStart):.2f}秒")

    # 读取出来的数据是(M,L,C)的形式,M是股票数, 但由于传入的股票有可能并不能完整构造出过去80天的数据,则需要重新构造出完成的股票清单
    # 先构造一个临时的股票代码字典
    tmp_stock_dict = {}
    for stock_code in all_stocklist:
        full_name = stock_code
        simple_name = int(stock_code.split('.')[0])
        tmp_stock_dict[simple_name] = full_name
    timeStepsData = torch.from_numpy(timeStepsData)  # [M, L, C]
    M, L, C = timeStepsData.shape
    refer_stocklist = []  # 重新构造的股票清单,先将它清空
    for i in range(M):
        stock_simple_code = int(timeStepsData[i, 0, 0].item())
        full_name = tmp_stock_dict[stock_simple_code]
        refer_stocklist.append(full_name)  # 只加入本次读取成功了的股票全名
    sc = len(refer_stocklist)
    print(f"去掉读取失败的股票,重新构造的推理股票清单长度:{sc}")

    if shuffle == 0:
        # 20251118修复: 因为selfAttn只能在组内(200只股票)完成,因此所有股票的排序如果变化(即便总股票数不变),也会导致推理结果细微的差别,
        # 在很段的时间内运行两遍代码,输出的权重肉眼可见了.  为了去除这样的随机性, 我们将股票按其代码进行排序
        # 构造一个临时的字典用于保存股票在原始数据timeStepsData中的位置
        pos_dic = {}
        index = 0
        for stock_code in refer_stocklist:
            pos_dic[stock_code] = index
            index += 1
        # 对refer_stocklist进行排序
        refer_stocklist = sorted(refer_stocklist)
        # 构造一个下标数组
        newpos = []
        for stock_code in refer_stocklist:
            newpos.append(pos_dic[stock_code])
        # 重新排列原始数据
        timeStepsData = timeStepsData[newpos]

    today = DT.timestr(1)

    # Step2: 读取今天的开盘价数据
    print(f"T1日用于对数据进行归一的基准价为:{base_price_type}")
    opent1_file = os.path.join(UTILS.g_open_t1_path, f"open_price_{tradeDay}.csv")
    if tradeDay < today:
        # 历史数据,直接从数据文件中读取
        opent1_data, success_count = UTILS.read_price_from_buffer(refer_stocklist, 'open', tradeDay)
    else:
        # 从tushare获取基准价格数据
        opent1_data, success_count = UTILS.read_open_price_from_csv(refer_stocklist, opent1_file)
    if success_count == 0:
        print(f"{DT.timestr()}获取今日{tradeDay}开盘价失败")
        exit()
    print(f"{DT.timestr()}获取今日{tradeDay}开盘价成功,成功获取{success_count}只股票")
    print(f"{DT.timestr()}进一步查看是否有开盘价为0或者除权派息等特殊情况的股票...")

    st_and_cq_list, _ = UTILS.read_price_from_buffer(refer_stocklist, 'stcq', tradeDay)

    rm_type = ['无开盘价', 'STCQ', '开盘价异常小值', '除权派息']
    rm_stock = {0: 0, 1: 0, 2: 0, 3: 0}
    for stock_code in refer_stocklist:
        if not opent1_data.__contains__(stock_code):
            opent1_data[stock_code] = [0, '']
            rm_stock[0] += 1
            continue
        if st_and_cq_list is not None and stock_code in st_and_cq_list and st_and_cq_list[stock_code][0] > 0.99:
            opent1_data[stock_code][0] = 0.
            # print(f"****************命中STCQ***{tradeDay}-{stock_code}*******")
            rm_stock[1] += 1
            continue
        # sn_chinese = opent1_data[stock_code][1] # XT开盘价文件的第3个字段已改为最新价,不再是公司名称,所以不能用了
        if opent1_data[stock_code][0] <= 0.0001:
            # print(f"{stock_code}-{sn_chinese},", end='')
            opent1_data[stock_code][0] = 0.
            rm_stock[2] += 1
            continue
        # if "XD" in sn_chinese or "XR" in sn_chinese or "DR" in sn_chinese:
        #     # print(f"{stock_code}-{sn_chinese},", end='')
        #     opent1_data[stock_code][0] = 0.
        #     rm_stock[3] +=1

    rm_count = sum(list(rm_stock.values()))
    if rm_count > 0:
        for k, count in rm_stock.items():
            if count <= 0:
                continue
            reason = rm_type[k]
            print(f"{DT.timestr()}*****发现异常-{reason}-数量-{count}")
        print(f"{DT.timestr()}*****发现异常剔除总数量-{rm_count}")

    # 构造出今天的开盘价数据(同时也过滤掉开盘价为0的股票)
    open_t1 = []
    keep_index = []
    new_refer_stocklist = []

    close_i = UTILS.g_field_index['close']
    amount_i = UTILS.g_field_index['amount']
    yesterDayClose = timeStepsData[:, -1, close_i]  # 昨天的收盘价 [M]
    am5T = 0
    am5TBJ = 0
    for idx in range(len(refer_stocklist)):
        stock_code = refer_stocklist[idx]
        today_open_price = opent1_data[stock_code][0]
        if today_open_price <= 0.0001:
            # 开盘价为0, 跳过,不参与推理
            continue
        # 过滤掉开盘跌幅大于skipLoss的股票
        # 先得到昨天的收盘价
        yesterday_close_price = yesterDayClose[idx].item()
        if (today_open_price / yesterday_close_price - 1) <= -skipLoss:
            # print(f"{stock_code}开盘跌幅大于{skipLoss}%, 跳过")
            continue

        # 在这里可以插入一个检查点:  如果回看窗口中某只股票在过去最近的5天处于ST或者停牌状态, 我们就跳过它,不加入推理股票池!!! 20250929
        # 数据在 timeStepsData[idx,:,:]  [L,C]
        if UTILS.G_ALL_STOCK_DATA.is_stock_st_or_suspend(timeStepsData[idx, :, :]):
            # 在回看数据中有任意一天ST或者停牌
            continue

        # 20260304 增加一个排除逻辑: 过于5个交易日的日均成交额低于1000万,则排除它 (源数据中成交额的单位是千元)
        amount5 = float(timeStepsData[idx, -5:, amount_i].mean().item()) * 0.1  # 转换为万元
        if amount5 < args.amount5:
            # 过去5个交易日的平均成交额低于下限,剔除它
            # print(timeStepsData[idx,-5:,amount_i])
            # print(f"{amount_i}-----{stock_code}--{tradeDay}--{amount5}")
            am5T += 1
            if 'BJ' in stock_code:
                am5TBJ += 1
            continue

        # ADD 20251127 指定股票池进行推理, 成为可配置选项
        if args.pool is not None:
            if isinstance(args.pool, list):
                if stock_code not in args.pool:
                    continue
            elif isinstance(args.pool, dict):
                key = str(tradeDay)
                day_code = args.pool[key]
                # print(f"POOL{key}--len:{len(day_code)}")
                if stock_code not in day_code:
                    continue

        # ADD 20260119 限定进入推理池的股票的最小市值
        if args.min_total_mv > 0 or args.max_total_mv > 0:
            # 如果有限制条件,则要进行判断
            tmvIdx = UTILS.g_field_index['total_mv']
            stockTMV = int(timeStepsData[idx, 0, tmvIdx].item() / 10000.)  # 股票总市值(亿)
            if args.min_total_mv > 0 and stockTMV < args.min_total_mv:
                # 股票的总市值不足,则跳过
                continue
            if 0 < args.max_total_mv < stockTMV:
                # 股票的总市值超范围,也跳过
                continue

        new_refer_stocklist.append(stock_code)
        open_t1.append(today_open_price)
        keep_index.append(idx)

    print(f"AM5,{tradeDay},{am5T},{am5TBJ}")

    # 真正需要进行推理的总股票数
    refer_stocklist = new_refer_stocklist  # 这个list才是最后真正送入模型进行推理的股票清单
    M = len(refer_stocklist)

    open_t1 = torch.tensor(open_t1)  # [M]
    open_t1 = open_t1.unsqueeze(0)  # [1,M]
    timeStepsData = timeStepsData[keep_index]  # [M,L,C]
    timeStepsData = timeStepsData.unsqueeze(0)  # [1,M,L,C]

    # 构造上一天的权重向量
    prev_w = torch.zeros(1, 1 + M)  # [1,1+M]
    prev_w[:, 0] = 1.0

    # 精准对比测试时, 可以将某一天的[前权重]手动设置在这里

    if last_weight is not None:
        # 如果存在上一天的权重,则要根据它来设定今天推理时要输入的prev_w权重
        sv = 0.  # 所有股票的总权重
        for scode, weight in last_weight.items():
            if scode == 'CASH' or scode == 'cash':
                continue
            if not scode in refer_stocklist:
                continue
            idx = refer_stocklist.index(scode)
            if idx >= 0:
                # 昨天持仓的股票,在今天推理股票中的位置
                prev_w[0, 1 + idx] = weight
                sv += weight
                # print(f"昨天持仓:{scode},权重:{weight}")
        prev_w[0, 0] = 1.0 - sv  # 剩余的权重归一化到剩余的资金上

    return timeStepsData, open_t1, prev_w, refer_stocklist


def save_result(targetFile, total_weight):
    if not os.path.exists("./output/"):
        os.makedirs("./output/")
    with open(f"./output/{targetFile}", "w") as f:
        json.dump(total_weight, f)


def trunc_float(x, d):
    f = 10 ** d
    x = math.floor(x * f) / f
    return x


def resort_weight(unsorted_weight):
    """
    对于一个权重组合进行排序,如有现金项则放首位
    """

    sw = {}
    if unsorted_weight.__contains__('CASH'):
        sw['CASH'] = unsorted_weight['CASH']
        unsorted_weight.pop('CASH')
    sorted_w = sorted(unsorted_weight.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_w)):
        stock_code = sorted_w[i][0]
        stock_weight = sorted_w[i][1]
        sw[stock_code] = stock_weight
    return sw


def handle_stop_stock_ifany(last_weight, total_weight, tradeDay):
    """
    如果昨日持仓的股票在今天是处于停牌状态, 则需要无条件保留其权重
    输入的两个权重是字典表格式
    """

    if last_weight is None or len(last_weight) <= 1:
        # 昨日持仓为空或者为全现金持仓
        return total_weight
    last_stock = list(last_weight.keys())
    stop_list = UTILS.get_stop_stocks(last_stock, int(tradeDay))
    if len(stop_list) <= 0:
        # 没有今天停牌股票
        return total_weight

    # 发现有需要处理的这种股票
    # step1: 构造出一个新的字典表, 先将这些股票无条件置入
    new_weight = {"CASH": 0.}
    used_weight = 0.
    for stc in stop_list:
        new_weight[stc] = last_weight[stc]
        used_weight += last_weight[stc]  # 统计累计已消耗掉的权重之和
        print(f"******停牌股票:{stc} at {tradeDay}******")

    # step2: 将今日权重图中的停牌股票删除(如果有的话)
    for stc in stop_list:
        total_weight.pop(stc, 0)

    # step3: 计算剩余需要投资的股票的权重之和
    total_weight.pop('CASH', 0)  # 先将今日权重图中的现金项删除
    stock_w = sum(total_weight.values())  # 股票项之和
    possible_w = 1.0 - used_weight  # 当前可用的权重

    if possible_w <= 1e-4:
        # 如果可用权重已经非常小,则其他股票全部忽略
        new_weight['CASH'] = possible_w
    elif stock_w <= possible_w:
        # 剩余股票需要的权重在可用权重范围内,则直接复制即可
        new_weight.update(total_weight)
        new_weight['CASH'] = possible_w - stock_w
    else:
        # 否则, 剩余股票所需要的权重大于当前可用的权重, 则需要对所有股票的权重进行等比例缩小,以使得总权重为1
        scale_ratio = possible_w / stock_w
        for k, v in total_weight:
            # 剩余所有股票都按比例缩小
            total_weight[k] = v * scale_ratio
        new_weight.update(total_weight)
        new_weight['CASH'] = possible_w - stock_w

    new_weight = resort_weight(new_weight)
    return new_weight


def make_ss_score(args, inputSeq, futureSeq, globalSeq):
    IB, IM, IL, IC = inputSeq.shape
    s = []
    with torch.no_grad():
        G = args.G
        if IM <= G:
            ss_score = args.ss_model(inputSeq, futureSeq, globalSeq)  # [B,M,M]
            ss_score = ss_score.mean(dim=1).to(args.device)  # [B,M]
            s.append(ss_score)
        else:
            LEFT = 0
            for i in range(0, IM, G):
                LEFT = IM - i
                if LEFT < G:
                    # 最后一组,有可能数目非常小,导致全局信息不足
                    break
                iS = inputSeq[:, i:i + G, :, :]
                fS = futureSeq[:, i:i + G, :, :]
                ss_score = args.ss_model(iS, fS, globalSeq)  # [B,M,M]
                ss_score = ss_score.mean(dim=1).to(args.device)  # [B,M]
                s.append(ss_score)
            if LEFT > 0:
                # 最后一组不足G数量
                i = IM - G
                iS = inputSeq[:, i:i + G, :, :]
                fS = futureSeq[:, i:i + G, :, :]
                ss_score = args.ss_model(iS, fS, globalSeq)  # [B,M,M]
                ss_score = ss_score.mean(dim=1).to(args.device)  # [B,M]
                ss_score = ss_score[:, -LEFT:]
                s.append(ss_score)
    s = torch.cat(s, dim=1).contiguous()
    # 将ss_score复制扩展为 [B, M, L, 1]
    s = s.reshape(IB, IM, 1, 1).expand(-1, -1, IL, -1)  # [B, M, L, 1]
    return s


# factor_exp [M,10] 这个是全部股票T0日暴露
# w_raw [1+M]
# factor_exp, w_raw, index_info = make_sub_list(factor_exp, w_raw, args.minw)
def make_sub_list(factor_exp, w_raw, minw):
    factor_exp_list = factor_exp.tolist()
    w_raw_list = w_raw.tolist()
    M = len(w_raw_list) - 1
    index_info = {}  # 用一个字典来记录保留下来的股票序号对应于原始数据中的序号
    ki = 0

    fe = []
    we = [0.]
    tw = 0.
    for i in range(M):
        vw = w_raw_list[i + 1]
        if vw >= minw:
            # 保留股票
            index_info[i] = ki
            ki += 1
            fe.append(factor_exp_list[i])
            we.append(vw)
            tw += vw

    we[0] = 1. - tw  # cash

    fe = np.array(fe)
    we = np.array(we)
    return fe, we, index_info


# 此时 mw的形状为 [1+S] S是在通过优化器之前有权重的股票
# mw = restore_weights(M,mw,index_info)
def restore_weights(M, mw, index_info):
    nw = [0.] * (M + 1)  # [1+M]
    mwl = mw.tolist()
    nw[0] = mwl[0]
    for ki, tki in index_info.items():
        nw[ki + 1] = mwl[tki + 1]
    return np.array(nw)

def sector_neutral_inference(logits, factor_tensor):
    """
    logits: (B, 1+M) 模型的原始输出打分，第0位是现金，后M位是股票
    factor_tensor: (B, M, 1, C) 通道特征，C>=151
                   通道 107: 中证1000成分股权重 (非成分股为 0 或 NaN)
                   通道 120-150: 31个行业的 One-Hot 编码
    返回: (B, 1+M) 严格行业中性化后的归一化持仓权重
    """
    B, M = logits.shape[0], logits.shape[1] - 1

    # ==========================================
    # 1. 第一步：确定现金比例
    # ==========================================
    # 在全局维度先做一次 softmax，唯一的目的是拿到现金的真实意图比例
    global_w = F.softmax(logits, dim=1)
    cash_w = global_w[:, 0:1]  # (B, 1) 现金的绝对权重
    stock_pool_w = 1.0 - cash_w  # (B, 1) 留给所有股票的总资金比例

    # ==========================================
    # 2. 第二步：提取基准指数的行业配比 (Benchmark Sector Weights)
    # ==========================================
    # 提取中证1000权重，并处理可能的 NaN
    # (B, M)
    zz1000_weights = factor_tensor[:, :, 0, 104]
    zz1000_weights = torch.nan_to_num(zz1000_weights, nan=0.0)

    # 提取 31 个行业的 One-Hot 矩阵
    # industry_onehot: (B, M, 31)
    industry_onehot = factor_tensor[:, :, 0, 119:150]

    # 计算中证1000指数在各行业的真实配比
    # (B, 1, M) @ (B, M, 31) -> (B, 1, 31) -> (B, 31)
    # 这就是基准在这个行业里到底放了多少钱
    benchmark_sector_weights = torch.bmm(
        zz1000_weights.unsqueeze(1),
        industry_onehot
    ).squeeze(1)

    # 安全归一化 (防除零)
    bench_sum = benchmark_sector_weights.sum(dim=-1, keepdim=True) + 1e-8
    benchmark_sector_weights = benchmark_sector_weights / bench_sum

    # ==========================================
    # 3. 第三步：模型在行业内部挑选个股
    # ==========================================
    stock_logits = logits[:, 1:]  # 剥离现金，只看股票 (B, M)

    # 将 Logits 强行隔离进 31 个行业
    # masked_logits 形状: (B, 31, M)
    # 非本行业的股票打分被置为 -1e9，在 softmax 后权重绝对为 0
    masked_logits = stock_logits.unsqueeze(1) - (1.0 - industry_onehot.transpose(1, 2)) * 1e9

    # 在行业内部独立做 Softmax (如果用了 entmax15，实盘会完全没有碎股)
    # sector_internal_w 形状: (B, 31, M)
    sector_internal_w = F.softmax(masked_logits, dim=-1)

    # ==========================================
    # 4. 第四步：宏观到微观的资金下发
    # ==========================================
    # 把模型预留的股票总资金，按基准行业比例，切分给 31 个行业
    # target_sector_capital: (B, 31, 1)
    target_sector_capital = (stock_pool_w * benchmark_sector_weights).unsqueeze(-1)

    # 行业资金 * 行业内部分配比例
    # (B, 31, M) * (B, 31, 1) -> (B, 31, M)
    allocated_stock_w = sector_internal_w * target_sector_capital

    # 把所有 31 个行业的分配加起来，得到每只股票最终的权重
    # final_stock_w: (B, M)
    final_stock_w = allocated_stock_w.sum(dim=1)
    print(f"每批次股票总权重和: {final_stock_w.sum(dim=1).tolist()} (预期接近 1.0)")

    # ==========================================
    # 5. 第五步：组装最终张量
    # ==========================================
    # 拼回现金项 (B, 1+M)
    final_w = torch.cat([cash_w, final_stock_w], dim=1)

    return final_w

def model_ms_days(m):
    """模型分钟分支需要的回看天数(ms_time_step), 由模型结构推出"""
    rep = m.representation_ms
    return rep.num_patches * rep.patch_size // 48


def make_minute_seq(model_list, args, inputSeq, tradeDay, refer_stocklist):
    """
    为推理股票池读取5分钟数据 [1, M, D, 48, C], D为各模型所需天数的最大值; 没有模型使用分钟数据时返回None
    分钟窗口最后一天 = 日线回看窗口最后一天, 并用日线数据中的日期字段做校验
    """
    ms_models = [m for m in model_list if getattr(m, 'use_minute', False)]
    if len(ms_models) == 0:
        return None
    if args.mask_mf == 7:
        raise ValueError("mask_mf=7 会在股票维插入现金股票, 与分钟数据的股票维不一致, 不支持")
    days = max(model_ms_days(m) for m in ms_models)

    last_day = DT.get_next_trade_day(tradeDay, -1 - args.ext_day)
    # 用日线数据的日期字段核对窗口最后一天
    YI = UTILS.g_field_index['year']
    ymd = inputSeq[0, :, -1, YI:YI + 3].round().long()
    day_dates = (ymd[:, 0] * 10000 + ymd[:, 1] * 100 + ymd[:, 2]).tolist()
    n_match = sum(1 for d in day_dates if d == int(last_day))
    if n_match < 0.5 * len(day_dates):
        raise ValueError(f"日线回看窗口最后一天与分钟窗口最后一天{last_day}不一致(仅{n_match}/{len(day_dates)}只股票匹配)")

    if getattr(args, 'ms_reader', None) is None:
        args.ms_reader = MinuteBinReader(getattr(args, 'ms_data_path', DEFAULT_MS_DATA_PATH),
                                         workers=getattr(args, 'ms_workers', 16))
    t0 = time.time()
    ms_seq = args.ms_reader.read(refer_stocklist, last_day, days, min_cover=getattr(args, 'ms_min_cover', 0.0))
    print(f"{DT.timestr()}分钟数据读取完成{tuple(ms_seq.shape)}, 耗时{time.time() - t0:.2f}秒")
    assert ms_seq.shape[1] == inputSeq.shape[1], "分钟数据与日线数据的股票数不一致"
    return ms_seq.to(args.device)


def infer_one_day(model_list, model_args, args, tradeDay, last_weight=None, withBJ=True, shuffle=0):
    # STEP1: 准备原始数据, 比如推理日期为Today, 需要准备Today之前的若干天数据,当天的开盘价数据(或者是935数据),以及未来若干天的数据
    base_price_type = 'open'
    if model_args.price_target == '935':
        base_price_type = '935'
    # 注意: BatchSize=1, 所有的张量返回的数据形状都应该是[B...]的形式, 其中B=1
    print(f"{DT.timestr()}准备推理所需数据...")
    inputSeq, base_price, prev_w, refer_stocklist = make_infer_data(args, tradeDay=tradeDay,
                                                                    look_back_days=model_args.time_step + args.ext_day,
                                                                    base_price_type=base_price_type,
                                                                    skipLoss=args.skipLoss,
                                                                    last_weight=last_weight, withBJ=withBJ,
                                                                    shuffle=shuffle)
    # //inputSeq  -->
    if args.mask_mf == 7:
        # WZ 的另一个模型, 它需要人工构造0号股票,用于代表全局/现金特征
        '''构造代表现金的股票'''
        args.channel_names = ['gen_exchange', 'gen_market', 'gen_industry', 'gen_year', 'gen_month', 'gen_day',
                              'gen_week', 'gen_cq', 'open', 'high', 'low', 'close', 'pre_close',
                              'change', 'pct_chg', 'vol', 'amount', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                              'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
                              'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'buy_sm_vol',
                              'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol',
                              'buy_md_amount', 'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount',
                              'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount',
                              'sell_elg_vol', 'sell_elg_amount', 'net_mf_vol', 'net_mf_amount', 'up_limit',
                              'down_limit', 'call_auction_close', 'call_auction_open',
                              'call_auction_high', 'call_auction_low', 'call_auction_vol', 'call_auction_amount',
                              'call_auction_vwap', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
                              'adj_factor', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 'kdj_bfq', 'kdj_d_bfq',
                              'kdj_k_bfq', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250',
                              'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq', 'macd_dea_bfq',
                              'macd_dif_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq', 'rsi_bfq_12',
                              'rsi_bfq_24', 'rsi_bfq_6', 'wr_bfq', 'wr1_bfq', 'XIN9', 'HSI', 'HKAH', 'DJI', 'SPX',
                              'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price',
                              'news_micro_score', 'news_macro_score', 'is_suspend', 'st', 'limit_mark',
                              'gen_padding_flag']

        args.cash_share_index = [args.channel_names.index(x) + 2 for x in
                                 ['gen_year', 'gen_month', 'gen_day', 'gen_week', 'XIN9', 'HSI', 'HKAH', 'DJI', 'SPX',
                                  'IXIC', 'news_macro_score']]

        cash_stock = torch.zeros_like(inputSeq[:, [0], :, :].clone())  # 构建全0的现金股票数据 [B,1,L,C]
        cash_stock[..., args.cash_share_index] = inputSeq[:, 0, :,
                                                 args.cash_share_index].clone()  # 现金股票的持仓量等于原数据中第一个股票的持仓量
        inputSeq = torch.cat([cash_stock, inputSeq], dim=1)  # 在资产维度上将现金股票添加到原数据中 [B, M+1, L, C]

        cash_base = torch.ones_like(base_price[:, [0]].clone()) * 1e-8  # 构建现金股票的基准价格 [B,1]
        base_price = torch.cat([cash_base, base_price], dim=1)  # [B, M+1]

    inputSeq = inputSeq.to(args.device).to(torch.float32)  # [B,M,L,C] C=104 106
    style_factor = None
    if args.DS == 'day134' or args.DS == 'wd395':
        # 只有134数据集中才有这10个因子暴露数据
        style_factor = inputSeq[..., 311:321]

    base_price = base_price.to(args.device).to(torch.float32)  # [B,M]
    prev_w = prev_w.to(args.device).to(torch.float32)  # [B,1+M]
    factor_tensor_industy_reweight = inputSeq[:, :, -1, :].unsqueeze(2)

    # STEP2: 对数据进行特别处理
    M = inputSeq.shape[1]  # 股票数
    if args.mask_mf == 7:
        M -= 1
    lastDayWeight = None
    if args.mask_mf == 502:
        # Ron B1V5 微扰模型, 需要根据当前推理目标确定lastDayWeight参数的值
        lastDayWeight = inputSeq[:, :, -1, 104].clone()
        if args.indexName == 'qa':
            lastDayWeight[:, :] = 0.
    # all_stock_code = inputSeq[0,:,0,0].long().clone()  #所有股票的代码
    # stock_type_T0 = inputSeq[:, :, -1, -2].int().clone()  # [B,M]股票类型
    stock_type_T1, _ = UTILS.read_price_from_buffer(refer_stocklist, 'limit_mark', tradeDay)
    tmpList = []
    for x, v in stock_type_T1.items():
        tmpList.append(int(round(v[0])))
    assert len(tmpList) == M, f"读取T1日股票类型数据不一致:{M} vs {len(tmpList)}"
    stock_type_T1 = torch.tensor(tmpList).int().unsqueeze(0).to(args.device)  # [B,M]

    # 20250917 ADD 如果未读取到任何数据(或者说股票数量小于500都认为异常) 比如全天所有股票都是padding来的(因为数据源本身的问题), 但是当天是一个合法的交易日
    # 此时为了保证回测推理能正常延续, 就直接返回上一日的权重
    if M < 100:
        # if last_weight is not None:
        #     targetFile = UTILS.make_output_json_name(args, tradeDay)
        #     print(f"{DT.timestr()}推理结果保存到文件:./output/{targetFile}")
        #     save_result(targetFile, last_weight)
        #     return len(last_weight) - 1, copy.deepcopy(last_weight)
        # else:
        #     next_w = prev_w
        raise ValueError(f"推理股票池太小了:{M}")
    if args.ext_day > 0:
        # 如果有额外读取的数据,则在此将它删除掉
        inputSeq = inputSeq[:, :, : -args.ext_day, :]
        print('check shape for infer with gap', inputSeq.shape)

    # 5分钟数据: 与日线回看窗口的最后ms_time_step天对齐(窗口最后一天为T-1, 有ext_day时再往前移ext_day天)
    ms_seq = make_minute_seq(model_list, args, inputSeq, tradeDay, refer_stocklist)
    # orig_data = inputSeq.clone()  # 保留一份原始数据
    # orig_weight = {} # 保留模型的原始输出(全部的股票及其权重), key=stock_code, value=[总市值,权重...]  做数据分析时使用

    factor_tensor = None
    daily_factor_exp = None
    lambda_vec = None
    if args.mask_mf in [45, 47, 48] or args.optimizer > 0:
        # 使用优化器(只能在day134数据集下使用)/或者是使用了因子暴露数据的模型
        pre_tradeDay = DT.get_next_trade_day(tradeDay, -1)
        factor = args.zz1000exp[pre_tradeDay]
        # 转成 tensor
        factor_tensor = torch.tensor(factor, device=args.device)  # [10]
        print('因子提取', factor_tensor, type(factor))
        daily_factor_exp = inputSeq[:, :, -1, 108:118]  # [B,M,10]
        # 注意这里的下标位置是从108开始的连续10个字段为134数据集(原始全通道数据)中的10个风格因子暴露数据
        # 因为我们实际上并不可能知道在T1日(交易日)的各股票的暴露数据, 所以只能假定所有股票的因子暴露值从T0日到T1日是不变的!!!
        # 后续就根据这些暴露值去计算模型输出的PM组合的总的因子暴露
        # factor_tensor = factor_tensor[0:args.optimizer]  # 目标因子数(最多是10个)
        # daily_factor_exp = daily_factor_exp[:, :, 0:args.optimizer]
        vec = [args.opv_f] * args.op_fn
        lambda_vec = np.array(vec)

    Barra_return = Industry_exp_ID = None
    if args.mask_mf in [601, 602]:
        #Barra_return = inputSeq[:, 0, :, 150:192].unsqueeze(1)  # 取出十个因子收益
        #Industry_exp_ID = inputSeq[:, :, -1, 119:150]  # 取出31个行业因子暴露
        factor_last_step = style_factor[:, :, -1, :]
        mask = torch.zeros(1, 10, device=args.device)
        if args.Factor_constraint == 0:
            print('无约束方案')
            mask[0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=mask.dtype, device=mask.device)
        elif args.Factor_constraint == 1:
            print('全约束方案')
            mask[0, :] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=mask.dtype, device=mask.device)
        elif args.Factor_constraint == 2:
            print('严格风控')
            mask[0, :] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=mask.dtype, device=mask.device)
        elif args.Factor_constraint == 3:
            print('风格轮动')
            mask[0, :] = torch.tensor([0, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=mask.dtype, device=mask.device)
        else :
            print('风格轮动2')
            mask[0, :] = torch.tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=mask.dtype, device=mask.device)

        if args.indexTarget == 'qa':
            factor_exposure = torch.mean(factor_last_step, dim=1)  # dim=1 对应 M 维度
            daily_factor_exp = factor_exposure
            Index_weights_formodel = torch.full((1, M), 1.0 / M).to(args.device)
        else:
            if args.indexTarget == 'zz500':
                index_pos = 303
            elif args.indexTarget == 'hs300':
                index_pos = 302
            elif args.indexTarget == 'zz1000':
                index_pos = 304
            elif args.indexTarget == 'gz2000':
                index_pos = 305
            Index_weight = inputSeq[:, :, -1, index_pos]

            # 1. 检查成分股数量
            weight_mask = Index_weight > 0  # B * M
            stock_counts = weight_mask.sum(dim=1)  # B
            print('每个batch的成分股数量:', args.indexName, stock_counts)

            # 2. 检查权重之和
            weight_sum = Index_weight.sum(dim=1)  # B
            print('每个batch的权重之和:', weight_sum)

            # 3. 计算加权平均因子暴露 -> B * F
            # Index_weight: B * M -> B * M * 1
            # factor_last_step: B * M * F
            weights = Index_weight.unsqueeze(-1)  # B * M * 1
            Index_weights_formodel = Index_weight
            weight_sums = weight_sum.unsqueeze(-1)  # B * 1

            # 加权求和再除以权重之和
            weighted_factor = (weights * factor_last_step).sum(dim=1)  # B * F
            index_factor_exposure = weighted_factor / weight_sums  # B * F
            print('index_factor_exposure shape:', index_factor_exposure.shape, index_factor_exposure)  # B * F
            daily_factor_exp = index_factor_exposure
        if args.mask_mf == 601:
            core_indices = [0, 8, 9]
            daily_factor_exp = daily_factor_exp[:, core_indices]
    '''
    if args.mask_mf in [603]:
        Barra_return = inputSeq[:, 0, :, 150:192].unsqueeze(1)  # 取出十个因子收益
        Industry_exp_ID = inputSeq[:, :, -1, 119:150]  # 取出31个行业因子暴露 '''

    some_weights = None
    if args.mask_mf == 503:
        some_weights = inputSeq[:, :, -1, 304].clone()

    # 归一化数据
    print(f"{DT.timestr()}数据归一化处理...")
    # 在normalizer里,会将第一个通道"原始股票代码清除掉",并将需要标准化的字段进行标准化
    #inputSeq = data_normalizer(inputSeq, normalize=model_args.data_norm == 1)
    #last_four = inputSeq[:, :, :, -4:]
    if not hasattr(model_args, 'pre_known_future'):
        model_args.pre_known_future = 8
    #futureSeq = data_normalizer.make_futuer_seq(inputSeq, model_args.patch_size, model_args.pre_known_future)
    close_t0 = inputSeq[:, :, -1, model_args.close_price_index].clone()  # [B,M]t0收盘价
    #globalSeq = inputSeq.clone()
    if not hasattr(model_args, 'cq_index'):
        model_args.cq_index = 9
    #globalSeq[:, :, :, model_args.cq_index] = 0.  # 除权除息数据不参与模型训练
    # 以T0日的股票类型来判断T1日开盘价格是否处于涨/跌停状态
    # is_up_stop, is_down_stop = CM.cal_if_updown_stop(close_t0, base_price, stock_type_T0, tradeDay)
    # 以T1日股票类型来判断T1日开盘价格是否处于涨/跌停状态
    is_up_stop_T1, is_down_stop_T1 = CM.cal_if_updown_stop(close_t0, base_price, stock_type_T1, tradeDay)

    # 将is_up_stop[0,:]从张量转换为一个CPU上的list对象
    # iusT0 = is_up_stop[0].tolist()
    # iusT1 = is_up_stop_T1[0].tolist()
    # idsT0 = is_down_stop[0].tolist()
    # idsT1 = is_down_stop_T1[0].tolist()
    # for si in range(M):
    #     if iusT0[si]!=iusT1[si]:
    #         print(f"LIMIT-根据T0和T1日{tradeDay}股票类型判定涨停状态不一致:{iusT0[si]} vs {iusT1[si]}")
    #     if idsT0[si]!=idsT1[si]:
    #         print(f"LIMIT-根据T0和T1日{tradeDay}股票类型判定跌停状态不一致:{idsT0[si]} vs {idsT1[si]}")

    #inputSeq = CM.make_price_relative_v2(inputSeq, model_args.price_channels, base_price)

    if args.mask_mf in [4, 40] and len(args.setZeroIndex) > 0:
        # 需要强制清零的通道
        inputSeq[:, :, :, args.setZeroIndex] = 0.

    #inputSeq = inputSeq[:, :, :, args.keep_fields]
    #inputSeq[:, :, :, -4:] = last_four
    #futureSeq = futureSeq[:, :, :, args.keep_fields]
    #globalSeq = globalSeq[:, :, :, args.keep_fields]

    if args.mask_mf == 43:
        ss_score = make_ss_score(args, inputSeq, futureSeq, globalSeq)
        # 预训练评分子模型, 需要以下特殊处理
        # 1. 调用 ss_model 计算当前时刻所有股票互相之间的评价分数
        # with torch.no_grad():
        #     ss_score = args.ss_model(inputSeq, futureSeq, globalSeq)  # [B,M,M]
        # 2. 对futureSeq [B,M,P,C] 的最后一个维度加1,并且置为0
        f_pad = torch.zeros(futureSeq.shape[0], futureSeq.shape[1], futureSeq.shape[2], 1).to(args.device)
        futureSeq = torch.cat([futureSeq, f_pad], dim=-1).contiguous()
        # 3. 对globalSeq [B,M,L,C] 的最后一个维度加1,并且置为0
        g_pad = torch.zeros(globalSeq.shape[0], globalSeq.shape[1], globalSeq.shape[2], 1).to(args.device)
        globalSeq = torch.cat([globalSeq, g_pad], dim=-1).contiguous()
        # 4. 计算当前时刻的股票评价分数的均值
        # ss_score = ss_score.mean(dim=1).to(args.device)  # [B,M]
        # 将ss_score复制扩展为 [B, M, L, 1]
        # ss_score = ss_score.reshape(IB, IM, 1, 1).expand(-1, -1, inputSeq.shape[2], -1)  # [B, M, L, 1]
        # 5. 将ss_score 与 inputSeq 合并,作为模型的输入
        inputSeq = torch.cat([inputSeq, ss_score], dim=-1).contiguous()  # [B, M, L, C+1]
        # 到这里时, 输入的三个时序对象inputSeq,futureSeq,globalSeq的最后维度都是C+1了!!!

    # STEP3: 推理
    print(f"{DT.timestr()}开始进行模型推理,股票池总数:{M}...")
    timeStart = time.time()
    mtmodels = len(model_list) > 1
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        with (torch.no_grad()):
            mws = []
            for m in model_list:
                if args.mask_mf in [601, 602]:
                    # 10F/5F---xj
                    # 10F/5F---xj
                    '''mw = m.infer(prev_w, inputSeq, futureSeq, globalSeq, style_factor, daily_factor_exp, None,
                                 torch.tensor(args.t).to(args.device),
                                 torch.tensor(args.G).to(args.device))'''
                    #wt_next, logits = m.forward(prev_w, inputSeq, Barra_return, Industry_flow_amount, style_factor, daily_factor_exp, Industry_exp_ID)
                    '''196 old code'''
                    if args.multitask == 0:
                        if getattr(m, 'use_minute', False):
                            wt_next = m.forward(prev_w, inputSeq, daily_factor_exp, mask, Index_weights_formodel,
                                                ms_seq=ms_seq[:, :, -model_ms_days(m):])
                        else:
                            wt_next = m.forward(prev_w, inputSeq, daily_factor_exp, mask, Index_weights_formodel)
                    elif args.multitask == 1:
                        wt_next = m.forward_factor(prev_w, inputSeq, Barra_return, style_factor, daily_factor_exp, mask, Industry_exp_ID)
                    else:
                        wt_next = m.forward_quant(prev_w, inputSeq, Barra_return, style_factor, daily_factor_exp, mask, Industry_exp_ID)
                    '''227'''
                    #wt_next = m.forward(prev_w, inputSeq, Barra_return, Industry_flow_amount, style_factor, daily_factor_exp, Industry_exp_ID)
                    mw = wt_next
                    
                    if args.indexpull != 0:
                        print('indexpull being called', args.indexpull)
                        mw = active_weight_portfolio(mw, Index_weight, active_scale = args.indexpull, min_weight = 5e-4)
                    #mw = sector_neutral_inference(mw, factor_tensor_industy_reweight)
                    #mw = optimize_portfolio_with_cash(mw, Index_weight, alpha=0.8, w_min=0.0005)
                elif args.mask_mf in [603]:
                    mw = m(prev_w, inputSeq, Barra_return, style_factor, daily_factor_exp, Industry_exp_ID)
                elif args.mask_mf == 502:
                    # WZ的模型,未实现infer接口,所以仍然用forword方式来调用推理
                    mw = m(prev_w, inputSeq, futureSeq, lastDayWeight, torch.tensor(args.t).to(args.device))
                elif args.mask_mf == 503:
                    # Ron 基于wind311数据训练的模型,推理时只有两个参数
                    mw = m(inputSeq, some_weights)
                elif args.mask_mf == 6 or args.mask_mf == 7:
                    # WZ的模型,未实现infer接口,所以仍然用forword方式来调用推理
                    mw = m(prev_w, inputSeq, futureSeq, globalSeq)
                elif args.mask_mf == 9:
                    # WZ带温度参数模型,但不需要G参数
                    mw = m.infer(prev_w, inputSeq, futureSeq, globalSeq, torch.tensor(args.t).to(args.device))
                elif args.mask_mf in [42, 401, 402, 403, 701, 702]:  # 多头多任务模型E,以及一些变种
                    # LHM 多任务模型,用t来指定任务(0,1,2---3) 这里t是一个整数, t可以是0,1,2表示3个特定任务
                    result = m.infer(prev_w, inputSeq, futureSeq, globalSeq, torch.tensor(args.G).to(args.device),
                                     torch.tensor(float(args.t)).to(args.device))
                    if isinstance(result, tuple):
                        mw = result[0]
                    else:
                        mw = result
                elif args.mask_mf == 44:
                    # LHM 简化双头模型,用t来指定任务, 这里需要把t转换为[B,1]形状的long张量,数值的范围只能在0~5之间,其中0~4表示特定任务,5表示不指定
                    task = [[int(args.t)]]
                    task = torch.tensor(task).long().to(args.device)
                    mw = m.infer(prev_w, inputSeq, futureSeq, globalSeq, torch.tensor(args.G).to(args.device), task)
                else:
                    # 0/4/5 实现了infer接口,参数需要G
                    mw = m.infer(prev_w, inputSeq, futureSeq, globalSeq, torch.tensor(args.G).to(args.device))

                if args.optimizer > 0 and args.DS == 'day134':
                    print("-----经过优化器对模型输出的权重进行因子约束变换-----")
                    target_exp = factor_tensor[:args.op_fn].cpu().numpy()  # [10] 这个是目标暴露
                    factor_exp = daily_factor_exp[:, :, :args.op_fn].squeeze(dim=0).cpu().numpy()
                    w_raw = mw.squeeze(dim=0).cpu().numpy()

                    # target_exp [10]
                    # factor_exp [M,10] 这个是全部股票T0日暴露
                    # w_raw [1+M]
                    uv2 = args.opv == 1
                    index_info = None

                    if uv2:
                        factor_exp, w_raw, index_info = make_sub_list(factor_exp, w_raw, args.minw)
                        print("-----经过优化器2.0(限定有权股票范围内优化)-----")

                    mw = solve_portfolio_soft_constraints(
                        w_raw=w_raw,
                        factor_exp=factor_exp,
                        target_exp=target_exp,
                        lambda_vec=lambda_vec,
                        max_weight=0.03,
                        turnover_lambda=args.op_to
                    )

                    if uv2:
                        # 此时 mw的形状为 [1+S] S是在通过优化器之前有权重的股票
                        mw = restore_weights(M, mw, index_info)

                    '''恢复权重到原始的张量结构'''
                    mw = torch.from_numpy(mw).to(args.device)
                    mw = mw.unsqueeze(dim=0)

                if mtmodels:
                    mw = CM.smooth_weight(mw, smooth_weight=args.minw, single_max_weight=args.max_sw,
                                          movecash=args.movecash)
                    # 做推理测试时,对输出权重进行再处理,去除所有非法的加/减仓操作
                    mw = CM.deal_illeagle_weight(prev_w, mw, is_up_stop_T1, is_down_stop_T1)
                mws.append(mw)
    if len(mws) == 1:
        next_w = mws[0]
    else:
        next_w = torch.stack(mws, dim=0)
        next_w = next_w.mean(dim=0)
    print(f"{DT.timestr()}模型推理完成,耗时{(time.time() - timeStart):.2f}秒")

    # 对模型输出的原始权重进行记录,留待数据分析时使用
    # next_w 的形状为 [1,1+M]
    # tmvIdx = UTILS.g_field_index['total_mv']
    # for i in range(1,1+M):
    #     # 对每只股票,获取其股票代码
    #     stock_ID = all_stock_code[i-1]
    #     # 该股票的市值
    #     stock_TMV = int(orig_data[0, i, -1, tmvIdx].item() / 10000.)  # 股票总市值(亿)
    #     # 模型输出的权重
    #     stock_org_w = float(next_w[0,i].item())

    # STEP5: 后处理

    # ADD 20251203 对北交所股票做总额上限
    if args.__contains__('BJMAX') and args.BJMAX > 0:
        next_w = CM.smooth_weight_BJMAX(refer_stocklist, next_w, args.BJMAX)

    # 将权重中过小的值全部清0
    print(f"{DT.timestr()}对模型输出权重进行后处理...")
    next_w = CM.smooth_weight(next_w, smooth_weight=args.minw, single_max_weight=args.max_sw, movecash=args.movecash)
    # 做推理测试时,对输出权重进行再处理,去除所有非法的加/减仓操作
    next_w = CM.deal_illeagle_weight(prev_w, next_w, is_up_stop_T1, is_down_stop_T1)
    # ADD 20260304 在回测时做更严格的限制, 如果交易日收盘是涨停状态, 则默认今天的增仓无法完成
    # 这里可以复用上面的这个deal_illeagle_weight方法,传入is_up_stop参数,而此时该参数代表的是当日收盘是否涨停(而不是开盘了)
    # 读取交易日的收盘价
    if tradeDay < DT.timestr(1):
        cp, tl = UTILS.read_price_from_buffer(refer_stocklist, 'close', tradeDay)
        if tl != len(refer_stocklist):
            # 读取收盘价失败
            raise ValueError(f"读取交易日{tradeDay}收盘价失败")
        close_t1 = []
        if args.mask_mf == 7:
            close_t1.append(0.)
        for xc in refer_stocklist:
            close_t1.append(cp[xc][0])
        # 转换为tensor
        close_t1 = torch.tensor(close_t1).to(args.device)  # [M]
        close_t1 = close_t1.unsqueeze(0)  # [1,M]
        # 再次调用判断涨跌停方法
        is_close_up_stop, is_close_down_stop = CM.cal_if_updown_stop(close_t0, close_t1, stock_type_T1, tradeDay)
        # 再次调用处理非法交易的方法来处理非法的增仓行为
        next_w = CM.deal_illeagle_weight(prev_w, next_w, is_close_up_stop, is_close_down_stop)
    nw = next_w[0].cpu().numpy()

    # 取出所有股票名称
    take_stock = {}
    total_weight = {"CASH": 0.}
    tsw = 0.  # 所有股票的权重总和
    tsw_real = 0.

    for i in range(1, 1 + M):
        if nw[i] < 1e-6:
            continue
        # 如果nw[i]是NaN,也忽略
        if math.isnan(nw[i]):
            continue
        sw = float(nw[i])  # 得到股票的权重值(0~1)
        tsw_real += sw
        sw = trunc_float(sw, 6)  # 保留若干位小数(截断)
        tsw += sw
        take_stock[refer_stocklist[i - 1]] = sw

    # 按权重排序,然后将排序后的结果追加到total_weight中
    sorted_take_stock = sorted(take_stock.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_take_stock)):
        total_weight[sorted_take_stock[i][0]] = sorted_take_stock[i][1]
    cash_weight = 1.0 - tsw
    if cash_weight < 0:
        # 按道理是不应该执行这到里的!!!
        cash_weight = 0.
    total_weight['CASH'] = cash_weight

    # eps = 1e-4
    # if tsw > tsw_real + eps:
    #     xx = tsw / tsw_real
    #     print(f"BBP,{tsw_real:.6f},{tsw:.6f},{xx:.4f}")
    #     if xx > 1.01:
    #         print("BTX:",tsw_real_list)
    #         print("BTX:",sum(tsw_real_list))
    #         print("BTX:",tsw_list)
    #         print("BTX:",sum(tsw_list))
    # elif tsw < tsw_real - eps:
    #     print(f"BBQ,{tsw_real:.6f},{tsw:.6f},{tsw / tsw_real:.4f}")

    # 20250919 增加一个处理逻辑: 如果昨天已经持仓的股票, 今天处于停牌状态, 则该股票的权重应该保持完全不变,因为已无法交易!!!
    total_weight = handle_stop_stock_ifany(last_weight, total_weight, int(tradeDay))

    # 20260610 ADD for DEBUG 设定一个最大持股数,此时就全仓(除非没有股票有权重)
    if args.MS > 0:
        total_weight = make_full_stock_weight(total_weight, args.MS)

    targetFile = UTILS.make_output_json_name(args, tradeDay)
    print(f"{DT.timestr()}推理结果保存到文件:./output/{targetFile}")
    save_result(targetFile, total_weight)

    # poolFile = UTILS.make_output_pool_name(args, tradeDay)
    # print(f"{DT.timestr()}推理股票池保存到文件:./output/{poolFile}")
    # save_result(poolFile, refer_stocklist)

    return M, total_weight


def make_full_stock_weight(total_weight, MaxStockNum):
    """
    给定一个输出权重字典,将其修改为全仓
    total_weight本身已经是排序好的(cash在最前面,后面按权重由大到小排序)
    """

    AC = len(total_weight)
    if AC <= 1:
        return total_weight
    tsw = 0.
    fullWeight = {'CASH': 0.}
    SC = 0
    for k, v in total_weight.items():
        if k.upper() == 'CASH':
            continue
        # 找到一只股票
        tsw += v
        fullWeight[k] = v
        SC += 1
        if SC >= MaxStockNum:
            break

    new_t = 0.
    for k, v in fullWeight.items():
        if k.upper() == 'CASH':
            continue
        vv = trunc_float(v / tsw, 6)
        fullWeight[k] = vv
        new_t += vv
    fullWeight['CASH'] = 1. - new_t
    return fullWeight


def solve_portfolio_with_cash_dynamic(
        w_raw,  # shape: [1 + M]
        factor_exp,  # shape: [M, F]
        target_exp,  # shape: [F]
        max_weight=0.1,  # 单票最大权重（股票部分）
        min_cash_weight=0.0,  # 最小现金仓位（可选）
        turnover_penalty=0,  # 与原始权重距离的惩罚强度
        solver="ECOS"  # 推荐：ECOS / OSQP / SCS
):
    """
    A股 Long-Only 指增组合优化器（动态 M 版本）

    支持：
    ----------------------------------------
    1. 每天股票数 M 动态变化
    2. cash + stock
    3. 多因子暴露严格约束
    4. long-only（严格）
    5. 单票最大权重（严格）
    6. sum(w)=1（严格）
    7. 最小化与原始组合的距离

    参数说明：
    ----------------------------------------
    w_raw:
        原始模型输出（含现金）
        shape = [1 + M]

        例如：
        [cash, stock1, stock2, ...]

    factor_exp:
        股票因子暴露（不含 cash）
        shape = [M, F]

    target_exp:
        benchmark 的目标暴露
        shape = [F]

    返回：
    ----------------------------------------
    w_opt:
        优化后的最终持仓（含现金）
        shape = [1 + M]
    """

    ########################################################
    # Step 0
    # 基本检查
    ########################################################

    w_raw = np.asarray(w_raw, dtype=np.float64)
    factor_exp = np.asarray(factor_exp, dtype=np.float64)
    target_exp = np.asarray(target_exp, dtype=np.float64)

    M = factor_exp.shape[0]
    F = factor_exp.shape[1]

    print('cvxshape check', w_raw.shape[0], M)
    assert w_raw.shape[0] == M + 1
    assert target_exp.shape[0] == F

    ########################################################
    # Step 1
    # Optimization Variable
    #
    # w = [cash, stock1, ..., stockM]
    ########################################################

    w = cp.Variable(M + 1)

    ########################################################
    # Step 2
    # cash exposure = 0
    ########################################################

    cash_exp = np.zeros((1, F))

    factor_ext = np.vstack([
        cash_exp,
        factor_exp
    ])  # shape = [1+M, F]

    ########################################################
    # Step 3
    # Constraints
    ########################################################

    constraints = [w >= 0, cp.sum(w) == 1.0, w[1:] <= max_weight]

    # ------------------------------------------------------
    # 1. long-only
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 2. sum(w) = 1
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 3. 单票最大权重（仅股票）
    # ------------------------------------------------------

    # ------------------------------------------------------
    # 4. 最小现金仓位（可选）
    # ------------------------------------------------------
    '''
    constraints.append(
        w[0] >= min_cash_weight
    )'''

    # ------------------------------------------------------
    # 5. 因子暴露严格约束
    #
    # factor_ext.T @ w = target_exp
    # ------------------------------------------------------

    constraints.append(
        factor_ext.T @ w == target_exp
    )

    ########################################################
    # Step 4
    # Objective
    #
    # 最小化：
    # 与原始组合的距离
    ########################################################

    objective = cp.Minimize(
        turnover_penalty *
        cp.sum_squares(w - w_raw)
    )

    ########################################################
    # Step 5
    # Solve
    ########################################################

    problem = cp.Problem(
        objective,
        constraints
    )

    try:
        problem.solve(
            solver=solver,
            verbose=False
        )
    except Exception as e:
        print(f"Solver failed: {e}")
        return None

    ########################################################
    # Step 6
    # 检查结果
    ########################################################

    if w.value is None:
        print("Optimization infeasible.")
        return None

    w_opt = np.array(w.value)

    return w_opt


"""
 # 核心：每个因子的控制强度
    lambda_vec = np.array([
        50,  # size（强）
        50,  # beta（强）
        10,  # momentum
        10,  # liquidity
        10,
        5,
        5,
        5,
        5,
        5
    ])
"""


def solve_portfolio_soft_constraints(
        w_raw,  # [1+M]
        factor_exp,  # [M, F]
        target_exp,  # [F]
        lambda_vec,  # [F] 每个因子的权重
        max_weight=0.02,
        min_cash_weight=0.0,
        turnover_lambda=1.0,  # 跟原始组合的距离权重
        solver="ECOS"
):
    """
    Soft Constraint Portfolio Optimizer（推荐版本）

    支持：
    ----------------------------------
    ✔ 动态股票池 M
    ✔ 多因子软约束（10个 lambda）
    ✔ long-only
    ✔ 单票上限
    ✔ cash
    ✔ 完全可控 trade-off

    """

    # ===============================
    # 基本处理
    # ===============================

    w_raw = np.asarray(w_raw, dtype=np.float64)
    factor_exp = np.asarray(factor_exp, dtype=np.float64)
    target_exp = np.asarray(target_exp, dtype=np.float64)
    lambda_vec = np.asarray(lambda_vec, dtype=np.float64)

    M = factor_exp.shape[0]
    F = factor_exp.shape[1]

    # ===============================
    # 变量
    # ===============================

    w = cp.Variable(M + 1)  # 含 cash

    # ===============================
    # 扩展因子（cash=0）
    # ===============================

    cash_exp = np.zeros((1, F))
    factor_ext = np.vstack([cash_exp, factor_exp])  # [1+M, F]

    # ===============================
    # Constraints（必须保留）
    # ===============================

    constraints = [w >= 0, cp.sum(w) == 1, w[1:] <= max_weight, w[0] >= min_cash_weight]

    # long-only

    # sum=1

    # 单票上限（只对股票）

    # 现金下限

    # ===============================
    # Soft factor penalty
    # ===============================

    exposure = factor_ext.T @ w  # [F]

    factor_penalty = cp.sum(
        cp.multiply(
            lambda_vec,
            cp.square(exposure - target_exp)
        )
    )

    # ===============================
    # 主目标（贴近模型输出）
    # ===============================

    turnover_penalty = cp.sum_squares(w - w_raw)

    # ===============================
    # 最终 objective
    # ===============================

    objective = cp.Minimize(
        turnover_lambda * turnover_penalty
        + factor_penalty
    )

    # ===============================
    # 求解
    # ===============================

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=solver, verbose=False)
    except Exception as e:
        print("Solver error:", e)
        return None

    if w.value is None:
        print("Infeasible problem")
        return None

    return np.array(w.value)
