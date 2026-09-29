"""
生成指数基金的barra数据文件, 比如: indexName参数给定为hs300
则程序会从 /data/indexData/目录下去读取名称为hs300stock.json文件, 这个文件中有每天的成分股数据(股票及其权重)
读取完成后, 以每天的权重作为一个组合去计算它的因子暴露和收益等等数据, 再回写并覆盖位于/data/indexData/下的hs300_barra.json文件
前提是: 周华提供的barra因子数据API可用, 并且他已经更新其因子基础数据到了某个特定的日期
在运行本指令时, 需要留意, 可将match_year参数设置为如: 20170101,20260520 即无论成分股数据是否有0520之后的数据, 但barra数据只计算到0520为止
"""
import argparse
import json
import os
from time import time
import utils as UTILS
import common_utils as CM
import utils_barra as BA

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--indexName', type=str, default='', help='指数名称')
    # 程序会从上面这个path指定的路径下去查找所有的json文件, 并且要求json文件名中包含下面的match字符串
    # 如果match字符串为空,则默认查找path路径下所有的json文件
    # 另一个默认前提: 所有json文件名都以8位的日期开头
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--DEBUG', type=int, default=0, help='DEBUG')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # 检查权重目录下是否已经存在了barra分析数据
    bafile = os.path.join('/data/indexData/',f"{args.indexName}_barra.json")
    # if os.path.exists(bafile):
    #     raise ValueError(f"{args.path}路径下已存在barra.json文件")

    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights = UTILS.read_weight_from_index(indexName=args.indexName, match_year=args.match_year)
    if len(weights) < 5:
        print(f"权重数据太少{len(weights)},不足以进行区间分析,exit")
        exit(0)
    print(f"{CM.timestr()}读取权重数据完成,天数{len(weights)},耗时{time() - ts:.2f}秒")

    ba_dict = {}
    print(f"{CM.timestr()}开始进行barra因子暴露和收益分析...")
    DCN = 1
    for day,weight in weights.items():
        # print("try------------",day)
        if DCN % 100 ==0:
            print(f"{CM.timestr()}---#{DCN}-{day}...")
        DCN +=1
        try:
            qa_exp,qa_value,qa_alpha,pm_exp,pm_value,pm_alpha = BA.get_barra(day,weight)
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
    if args.DEBUG !=1:
        with open(bafile, 'w', encoding="utf-8-sig") as file:
            json.dump(ba_dict, file, indent=2)
    dayList = sorted(list(ba_dict.keys()))
    print(f"DONE--生成{args.indexName}的barra数据分析文件{bafile},days={len(dayList)},{dayList[0]}~{dayList[-1]}")








