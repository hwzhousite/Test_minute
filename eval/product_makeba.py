"""
尝试建立更多维度的评估指标
为了方便团队协作共享, 我们规定当前脚本的基本输入是一个模型回测时一段时间内连续N个交易日的持仓文件
文件为json格式
"""
import argparse
import json
import os
from time import time
import utils as UTILS
import common_utils as CM
import utils_barra as BA
import utils_date as DT

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    # 程序会从上面这个path指定的路径下去查找所有的json文件, 并且要求json文件名中包含下面的match字符串
    # 如果match字符串为空,则默认查找path路径下所有的json文件
    # 另一个默认前提: 所有json文件名都以8位的日期开头
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--keepcash', type=int, default=1, help='保留现金')

    args = parser.parse_args()

    assert args.path != '' and os.path.exists(args.path), f"必须指定合法存在的权重文件路径:{args.path}"
    return args


if __name__ == '__main__':
    args = get_args()

    # 检查权重目录下是否已经存在了barra分析数据
    bafile = os.path.join(args.path,"barra.json")
    # if os.path.exists(bafile):
    #     raise ValueError(f"{args.path}路径下已存在barra.json文件")

    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights = UTILS.read_weigh_from_path(args.path, args.match, args.match_year)
    if len(weights) < 5:
        print(f"权重数据太少{len(weights)},不足以进行区间分析,exit")
        exit(0)
    print(f"{CM.timestr()}读取权重数据完成,天数{len(weights)},耗时{time() - ts:.2f}秒")

    ba_dict = {}
    print(f"{CM.timestr()}开始进行barra因子暴露和收益分析...")
    DCN = 1
    for day,weight in weights.items():
        if DCN % 100 == 0:
            print(f"{CM.timestr()}---#{DCN}-{day}...")
        DCN +=1
        try:
            # 20260609 修复, 我们的权重是当日内完成的,且以次一交易日的对应价格来计算价值
            # 而且大多数回测都是以收盘价(复权后)来计算, 那么barra因子就要往后错一天才对
            # 我们的模型收益是在次日才兑现的, 所以barra也应该是次日的
            NDay = DT.get_real_next_trade_day(tradeDay=day,next_day=True)
            qa_exp,qa_value,qa_alpha,pm_exp,pm_value,pm_alpha = BA.get_barra(NDay,weight,keep_cash=args.keepcash==1)
            ba_dict[day]={'qa_exp':qa_exp,
                          'qa_value':qa_value,
                          'qa_alpha':qa_alpha,
                          'pm_exp':pm_exp,
                          'pm_value':pm_value,
                          'pm_alpha':pm_alpha}
        except Exception as e:
            print(e)

    print(f"{CM.timestr()}分析完成,数据写入文件...")
    # 将字典对象写入文件
    with open(bafile, 'w', encoding="utf-8-sig") as file:
        json.dump(ba_dict, file, indent=2)
    dayList = sorted(list(ba_dict.keys()))
    print(f"DONE--生成barra数据分析文件{bafile},days={len(dayList)},{dayList[0]}~{dayList[-1]}")








