"""
找最相似的区间,相似的判断条件是barra因子暴露向量
"""
import argparse
import json
import utils_date as DT
import product_check_barra_style as BAT
import common_utils as CM
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    parser.add_argument('--base_area', type=str, default='20180101,20251231', help='作为标准的历史区间')
    parser.add_argument('--check_area', type=str, default='20260310,20260420', help='待检测区间')
    args = parser.parse_args()
    with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
        DT.g_calendar = json.load(f)
    return args


def make_base_vector(bData, baseDayList, i, CL):
    """
    从基准区中指定的位置开始取连续的若干天的数据,汇总成一个表征向量
    """
    bV = []
    for index in range(i, i + CL):
        bV.extend(bData[baseDayList[index]])
    return bV


def dot_product_similarity(a, b):
    """
    计算点积相似度
    当向量归一化后，等价于余弦相似度
    """
    return np.dot(a, b)


def cosine_similarity(a, b):
    """
    计算余弦相似度
    返回值范围：[-1, 1]，1表示完全相同，-1表示完全相反
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)  # 加小值防止除零


if __name__ == '__main__':
    args = get_args()

    # 读取barra数据
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    print(f"{CM.timestr()}读取barra数据...")
    ba_dict = BAT.read_badic(args)
    # 当前存在barra数据的所有日期列表
    allDaysList = list(ba_dict.keys())
    allDaysList = sorted(allDaysList)
    print(f"{CM.timestr()}全部因子数据区间{allDaysList[0]}~{allDaysList[-1]},总天数{len(allDaysList)}")

    fc = ['size', 'comovement']
    # fc.extend(BA.style_factor1)  # 10个风格因子+1个国家因子
    # fc.extend(BA.industry_factor)  # 31个行业因子

    bArea = args.base_area.split(',')
    cArea = args.check_area.split(',')

    # 构造两个区间的因子暴露数据字典, k=日期, v=[42个因子暴露值]
    print(f"{CM.timestr()}构造因子数据序列...")
    bData = {}
    cData = {}
    for d in allDaysList:
        pm_exp = ba_dict[d]['pm_exp']
        fexp = []  # 生成该日期截面上的42个因子暴露值列表
        for f in fc:
            fv = pm_exp[f]
            fexp.append(fv)
        if bArea[0] <= d <= bArea[1]:
            bData[d] = fexp
        if cArea[0] <= d <= cArea[1]:
            cData[d] = fexp
    print(f"{CM.timestr()}基准区天数{len(bData)},待检区天数{len(cData)}")

    # 生成待检测向量
    print(f"{CM.timestr()}生成待检测向量...")
    CL = len(cData)  # 待检区的天数n
    CVector = []  # 待检区的全部序列n*42
    for _, vl in cData.items():
        CVector.extend(vl)
    print(f"{CM.timestr()}待检测天数{CL},向量长度{len(CVector)}")
    assert CL * len(fc) == len(CVector), "待检向量长度异常"

    # 遍历所有基准区
    baseDayList = list(bData.keys())
    BL = len(baseDayList)
    max_sim = -1.
    max_day = baseDayList[0]
    assert len(baseDayList) >= CL, f"基准区长度{BL}必须大于待检区长度{CL}"
    for i in range(0, BL - CL + 1):
        checking_day = baseDayList[i]
        BVector = make_base_vector(bData, baseDayList, i, CL)
        assert len(BVector) == len(CVector), f"基准区{i}-{checking_day}的表征向量长度异常"
        sim = cosine_similarity(CVector, BVector)
        print(f"{checking_day},{sim:.6f}", end='')
        if sim >= max_sim:
            max_sim = sim
            max_day = checking_day
            print("***")
        else:
            print()
    print(f"\nMax={max_day},{max_sim:.6f}")
