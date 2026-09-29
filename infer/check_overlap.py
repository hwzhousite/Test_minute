#进行批量测试
import argparse
import copy
import os
import sys
sys.path.append("../")
import common_utils as CM
import cal_value as CalTool
import utils as UTILS

def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--minw', type=float, default=0.001, help='最小权重, 默认为0.001')
    parser.add_argument('--skipLoss', type=float, default=1.0,
                        help='跳过开盘跌幅大于skipLoss的股票, 默认为1.0(即100%,不跳过)')
    parser.add_argument('--cfd', type=str, default='20250101', help='交易日期起始点')
    parser.add_argument('--cld', type=str, default='20250624', help='交易日期终止点')
    args = parser.parse_args()
    return args

def read_all_weights(tradeDay,md):
    weights = {}
    for k,v in md.items():
        weightFile = f"./output/{tradeDay}_{v}.json"
        if not os.path.exists(weightFile):
            return None
        w= CalTool.read_pmfile(weightFile)
        if w is None:
            return None
        if w.__contains__("CASH"):
            w.pop("CASH")
        weights[k] = w
    return weights

def cal_ol(key,day_weights):
    allKeys = key.split('-')
    code_list = set(day_weights[allKeys[0]].keys())
    for k in allKeys[1:]:
        # 取交集
        code_list = code_list & set(day_weights[k].keys())
    wl = []
    for k in allKeys:
        tmp = 0.
        for c in code_list:
            tmp += day_weights[k][c]
        wl.append(tmp)
    return len(code_list),sum(wl)/len(wl)

def cal_overlap(target_md,day_weights):
    ovs = copy.deepcopy(target_md)
    dsc = {}

    temp = {}
    for k,v in ovs.items():
        ovs[k], takeW = cal_ol(k,day_weights)
        temp[f"{k}_t"] = takeW
    for k in list(day_weights.keys()):
        dsc[k] = len(day_weights[k])
    for k,v in temp.items():
        ovs[k]=v
    return ovs,dsc

if __name__ == '__main__':
    args = get_args()
    if not CM.is_valid_date(args.cfd):
        raise ValueError(f"交易日期起始点{args.cfd}格式不正确, 请使用YYYYMMDD格式")
    if not CM.is_valid_date(args.cld):
        raise ValueError(f"交易日期终止点{args.cld}格式不正确, 请使用YYYYMMDD格式")

    tradeDayList = CM.trade_date_list(args.cfd,args.cld)
    tradeDayList = UTILS.get_rid_of_none_trading_days(tradeDayList)
    if len(tradeDayList) == 0:
        raise ValueError(f"交易日期起始点{args.cfd}和终止点{args.cld}之间没有交易日")

    check_models= {
        "5mONE": "/data/lhm_share/model_file/pmv2_0627_5m_epoch10.pth",
        "5mK5": "/data/lhm_share/model_file/pmv2_0626_5m_epoch85.pth",
        # "60m": "/data/lhm_share/model_file/pmv2_0629_60m_epoch9.pth",
        # "day": "/data/lhm_share/model_file/pmv2_0628_day_epoch14.pth"
    }

    md = {}
    for k,v in check_models.items():
        mark = os.path.basename(v)
        mark = os.path.splitext(mark)[0]
        mark = f"{mark}_s{args.skipLoss}_m{args.minw}"
        md[k] = mark

    target_md = {}
    key_list = list(md.keys())
    KL = len(key_list)
    for i in range(KL-1):
        for j in range(i+1,KL):
            k = f"{key_list[i]}-{key_list[j]}"
            target_md[k] = 0.
    if KL > 2:
        kt = '-'.join(key_list)
        target_md[kt] = 0.

    result_day = {}
    stock_count = {}
    for tradeDay in tradeDayList:
        day_weights = read_all_weights(tradeDay,md)
        if day_weights is None:
            continue
        ovs,dsc = cal_overlap(target_md,day_weights)
        result_day[tradeDay] = ovs
        stock_count[tradeDay] = dsc

    # 输出结果
    print("日期",end=',')
    print(",".join([f"{k}股票数" for k in list(md.keys())]),end=',')
    kl = list(target_md.keys())
    kllen= len(kl)
    print(",".join([f"{k}重合数,{k}重合权重" for k in kl]))
    for tradeDay,ovs in result_day.items():
        print(tradeDay,end=',')  # 日期
        # 每个模型的投资股票数
        for k in list(md.keys()):
            print(stock_count[tradeDay][k],end=',')
        # 重合数

        for i in range(kllen):
            k = kl[i]
            print(ovs[k],end=',')
            tk = f"{k}_t"
            print(f"{ovs[tk]:.2f}", end=',' if i<kllen-1 else '')  # 平均权重
        print()

    print("=====DONE=====")
