# 演示如何使用模型进行预测
# 当前的模型为：PM_SMV2模型
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

sys.path.append("../")
#from pmva.pm_model import PortfolioModel_SMV2 as PMVA
#from pmvb.pm_model import PortfolioModel_SMV2 as PMVB
#from pmve.pm_model_ve import PortfolioModel_SMV2 as PMVE
#from pmvei.pm_model_vei import PortfolioModel_SMV2 as PMVEI
#from pmvf.pm_model_vf import PortfolioModel_SMV2 as PMVF
#from pmvg.pm_model_vg import PortfolioModel_SMV2 as PMVG
#from pmvh.pm_model_vh import PortfolioModel_SMV2 as PMVH
#from pmvi.pm_model_vi import PortfolioModel_SMV2 as PMVI
#from pmvp.pm_model_vp import PortfolioModel_SMV2 as PMVP
#from pmvs.yy_ps_model import PS_Model
#from pmvf_ft.pm_model_vfft import SuperPM as PMVF_FT

from pmdayV2.pm_model import PortfolioModel as PMRon
from pmdayV2.pm_model import PortfolioModel as PMRonWind
#from XJ.stock_pm_model_v2_5F import PortfolioModel_SMV2 as PMXJ5F
#from XJ.stock_pm_model_v2_10F import PortfolioModel_SMV2 as PMXJ10F


# @ 若虚和周华的带源码回测,打开下面这一行, 带源码回测时指定模型类型名称关键字为W
# from pmdayV2.pm_model import PortfolioModel_SMV2 as PMWZ
# from pmvc.pm_model import PortfolioModel_SMV2 as PMVC
# from pmvd.pm_model import PortfolioModel_SMV2 as PMVD
# from pmWZ.pm_model import PortfolioModel_SMV2 as PMWZ
# from sub01.sub01_model import Sub01 as PMSUB01
# from SuperPM.Super_model import SuperPM as PMSUPER

# def load_ss_model(ssmodel_path, device):
#     model_args = CM.load_args_from_file(ssmodel_path)
#     ss_model = PS_Model(model_args)
#     state_dict_pre = torch.load(ssmodel_path, map_location='cpu', weights_only=True)
#     ss_model.load_state_dict(state_dict_pre, strict=False)
#     # 冻结模型参数
#     for param in ss_model.parameters():
#         param.requires_grad = False
#     ss_model.to(device)
#     ss_model.eval()
#     print(f"****SS评价模型参数加载成功:{ssmodel_path}****")
#     return ss_model


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
        # # 加载模型源文件, 并实例化模型   !!!带源码的回测方式已放弃!!!
        # elif modelType == 'W':
        #     # print("回测模型来自WZ, pmdayV2.pm_model")
        #     # model = PMWZ(model_args)
        #     # 这里是带着源码去创建模型实例, 如果有特殊处理可以加在这里
        # elif modelType == 'F':
        #     print("回测模型为:pmvf--简化双头模型 mask_mf 44")
        #     model = PMVF(model_args)
        # elif modelType == 'FT':
        #     print("回测模型为-pmvf-ft mask_mf 45")
        #     model = PMVF_FT(model_args)
        # elif modelType == 'G':
        #     print("回测模型为-pmvg mask_mf 47")
        #     model = PMVG(model_args)
        # elif modelType == 'H':
        #     print("回测模型为-pmvh mask_mf 48")
        #     model = PMVH(model_args)
        # elif modelType == 'I':
        #     print("回测模型为-pmvi mask_mf 402")
        #     model = PMVI(model_args)
        # elif modelType == 'P':
        #     print("回测模型为-pmvp mask_mf 701")
        #     model = PMVP(model_args)
        if modelType == 'RON':
            print("回测模型为-Ron mask_mf 502")
            model = PMRon(model_args)
        elif modelType == 'RONWIND':
            print("回测模型为-RonWind mask_mf 503")
            model = PMRonWind(model_args)
        # elif modelType == 'XJ5F':
        #     print("回测模型为-xj5f mask_mf 601")
        #     model = PMXJ5F(model_args)
        # elif modelType == 'XJ10F':
        #     print("回测模型为-xj10f mask_mf 602")
        #     model = PMXJ10F(model_args)
        else:
            raise ValueError(f"不支持模型类型{modelType}")
        # elif modelType == 'C':
        #     print("回测模型为:pmvc(含预训练模块)")
        #     model = PMVC(model_args)
        # elif modelType == 'D':
        #     print("回测模型为:pmvc(含预训练模块)")
        #     model = PMVD(model_args)
        # elif modelType == 'SUB01':
        #     print("回测模型为:sub01")
        #     model = PMSUB01(model_args)
        # else:
        #     print("回测模型为:SuperPM")
        #     model = PMSUPER(model_args)

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
        if UTILS.G_ALL_STOCK_DATA.is_stock_st_or_suspend(timeStepsData[idx, -10:, :]):
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


def infer_one_day(model_list, model_args, args, tradeDay, last_weight=None, withBJ=True, shuffle=0):
    # STEP1: 准备原始数据, 比如推理日期为Today, 需要准备Today之前的若干天数据,当天的开盘价数据(或者是935数据),以及未来若干天的数据
    base_price_type = 'open'
    # 注意: BatchSize=1, 所有的张量返回的数据形状都应该是[B...]的形式, 其中B=1
    print(f"{DT.timestr()}准备推理所需数据...")
    inputSeq, base_price, prev_w, refer_stocklist = make_infer_data(args, tradeDay=tradeDay,
                                                                    look_back_days=model_args.time_step + args.ext_day,
                                                                    base_price_type=base_price_type,
                                                                    skipLoss=args.skipLoss,
                                                                    last_weight=last_weight, withBJ=withBJ,
                                                                    shuffle=shuffle)
    # //inputSeq  -->
    inputSeq = inputSeq.to(args.device).to(torch.float32)  # [B,M,L,C] C=104 106
    style_factor = None
    if args.DS == 'day134':
        # 只有134数据集中才有这10个因子暴露数据
        style_factor = inputSeq[..., 108:118]

    base_price = base_price.to(args.device).to(torch.float32)  # [B,M]
    prev_w = prev_w.to(args.device).to(torch.float32)  # [B,1+M]

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
        inputSeq = inputSeq[:, :, args.ext_day:, :]
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

    if args.mask_mf in [601, 602]:
        pre_tradeDay = DT.get_next_trade_day(tradeDay, -1)
        factor = args.zz1000exp[pre_tradeDay]
        # 2. 转成 tensor
        factor_tensor = torch.tensor(factor, device=args.device)
        print('因此提取', factor_tensor, type(factor))

        # 如果五因子约束用这个
        if args.mask_mf == 601:
            core_indices = [0, 2, 4, 6, 9]
            daily_factor_exp = factor_tensor[core_indices].unsqueeze(dim=0)
        else:
            daily_factor_exp = factor_tensor.unsqueeze(dim=0)  # 十因子约束目标
        # print('shape check', daily_factor_exp.shape, daily_factor_exp)

    some_weights = None
    if args.mask_mf == 503:
        benchmark_weights = inputSeq[:, :, -1, 304].clone()
        # benchmark_weights = 1/inputSeq.shape[1] * torch.ones_like(inputSeq[:, :, -1, 304])

    # 归一化数据
    print(f"{DT.timestr()}数据归一化处理...")
    if not hasattr(model_args, 'pre_known_future'):
        model_args.pre_known_future = 8
    close_t0 = inputSeq[:, :, -1, model_args.close_price_index].clone()  # [B,M]t0收盘价
    if not hasattr(model_args, 'cq_index'):
        model_args.cq_index = 9
    # 以T0日的股票类型来判断T1日开盘价格是否处于涨/跌停状态
    # is_up_stop, is_down_stop = CM.cal_if_updown_stop(close_t0, base_price, stock_type_T0, tradeDay)
    # 以T1日股票类型来判断T1日开盘价格是否处于涨/跌停状态
    is_up_stop_T1, is_down_stop_T1 = CM.cal_if_updown_stop(close_t0, base_price, stock_type_T1, tradeDay)

    # STEP3: 推理
    print(f"{DT.timestr()}开始进行模型推理,股票池总数:{M}...")
    timeStart = time.time()
    mtmodels = len(model_list) > 1
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        with (torch.no_grad()):
            mws = []
            for m in model_list:
                if args.mask_mf == 502:
                    # WZ的模型,未实现infer接口,所以仍然用forword方式来调用推理
                    mw = m(prev_w, inputSeq, lastDayWeight, torch.tensor(args.t).to(args.device))
                elif args.mask_mf == 503:
                    # Ron 基于wind311数据训练的模型,推理时只有两个参数
                    rep, mw = m(inputSeq, dev=0.025, benchmark_weights=benchmark_weights, prev_w=prev_w)

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
