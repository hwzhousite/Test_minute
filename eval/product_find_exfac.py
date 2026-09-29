import json
import os.path
import platform
import numpy
import utils_make_factor_val as UTMF

FACTOR = ['size收益', 'non_linear_size收益', 'momentum收益',
          'liquidity收益', 'book_to_price收益', 'leverage收益', 'growth收益',
          'earnings_yield收益', 'beta收益', 'residual_volatility收益', 'comovement收益']

FACTOR_ALL = ['size收益', 'non_linear_size收益', 'momentum收益',
              'liquidity收益', 'book_to_price收益', 'leverage收益', 'growth收益',
              'earnings_yield收益', 'beta收益', 'residual_volatility收益', 'comovement收益',
              '银行收益', '计算机收益', '环保收益', '商贸零售收益', '电力设备收益', '建筑装饰收益', '建筑材料收益',
              '农林牧渔收益', '电子收益', '交通运输收益', '汽车收益', '纺织服饰收益', '医药生物收益', '房地产收益',
              '通信收益',
              '公用事业收益', '综合收益', '机械设备收益', '石油石化收益', '有色金属收益', '传媒收益', '家用电器收益',
              '基础化工收益', '非银金融收益', '社会服务收益', '轻工制造收益', '国防军工收益', '美容护理收益',
              '煤炭收益',
              '食品饮料收益', '钢铁收益']

FACTOR_FILE = r"/data/indexData/barra_factor_value.json"
if platform.system() == 'Windows':
    FACTOR_FILE = r"D:\stock_data\barra_factor_value.json"

"""
依赖全局数据文件: /data/indexData/barra_factor_value.json文件
该文件是一个Dict被dump为json格式, key为日期,value是一个list,长度为42,表示42个因子的收益
如果文件不存在,则需要调用 utils_make_factor_val.py来生成
提供两个API:
(1) find_factor_exnormal(['20190101','20260518'],['20260312','20260518'])
给定两个时间区间,第一个是基准区间,第二个是待检测区间
step1. 计算检测区的各因子累计收益
step2. 滑动窗口(长度与检测相同)在整个基准区内计算出各因子收益的均值/标准差
step3. 计算检测区各因子收益与基准区的偏离度,并按偏离度绝对值由高到低排列输出
返回一个字典 {k=因子名称,value=[abs_div, div]}

(2)find_all_exnormal(totalArea,DS,fi,maxDiv)
在给定的时间区间内,查找任意连续DS天内,因子收益偏离度大于某个值的区间
totalArea是长为2的list,为区间的首末日日期,DS为滑动窗口的长度(天),fi为整数表示第几个风格因子(或者风格因子名称字符串)
maxDiv为要求的偏离度阈值
返回数据为字典格式{key=numberID, value=[因子名称, 起始日期, 结束日期, 因子收益偏离度]}, 字典按起始日期项升序排列
"""


def check_factor_file():
    if not os.path.exists(FACTOR_FILE):
        UTMF.make_barra_factor_value()


def make_fv_value(data, startDate, endDate):
    dayList = list(data.keys())
    DL = len(dayList)
    vvlist = []
    for i in range(DL):
        d = dayList[i]
        if d < startDate:
            continue
        if d > endDate:
            break
        v = data[d]
        vvlist.append(v)
    if len(vvlist) == 0:
        return None, 0
    # 计算累乘数据
    xv = [1.] * len(FACTOR)
    for v in vvlist:
        for i in range(len(FACTOR)):
            xv[i] = xv[i] * (1. + v[i])
    return xv, len(vvlist)


def make_fv_value_2(data, k, ds):
    dayList = list(data.keys())
    vvlist = []
    for i in range(k, k + ds):
        d = dayList[i]
        v = data[d]
        vvlist.append(v)
    # 计算累乘数据
    xv = [1.] * len(FACTOR)
    for v in vvlist:
        for i in range(len(FACTOR)):
            xv[i] = xv[i] * (1. + v[i])
    return xv


def find_top_K(days, fi, total_ffv, DS, fmean, fstd, maxdv, topK):
    div = {}
    startIndex = 0
    TD = len(total_ffv)
    while True:
        fv = total_ffv[startIndex][fi]  # 因子收益
        ssv = abs(fv - fmean) / fstd
        rssv = (fv - fmean) / fstd
        if ssv >= maxdv:
            # 找到一个
            div[startIndex] = [days[startIndex], days[startIndex + DS - 1], ssv, rssv]
            startIndex += DS
        else:
            startIndex += 1
        if startIndex >= TD:
            break
    if len(div) <= 0:
        return []
    div = dict(sorted(div.items(), key=lambda x: x[1][2], reverse=True))
    res = []
    for k, v in div.items():
        res.append(v)
        if 0 < topK <= len(res):
            break
    return res


def find_factor_exnormal(totalArea, checkArea):
    """
    给定时间区间, 计算该区间的因子偏离程度, 输出全部因子的偏离度
    totalArea[s,e] 这是整体区间
    checkArea[s,e] 这是检测区间
    返回一个字典 {k=因子名称,value=[abs_div, div]}
    """
    check_factor_file()
    with open(FACTOR_FILE, encoding="utf-8-sig") as file:
        data = json.load(file)
    days = list(data.keys())
    for k in days:
        if k < totalArea[0] or k > totalArea[-1]:
            data.pop(k)
    days = list(data.keys())

    # 计算给定日期区间内的所有因子收益
    startDate = checkArea[0]
    endDate = checkArea[1]
    fv_dic_check, ds = make_fv_value(data, startDate, endDate)
    print(f"基准区,{days[0]},{days[-1]},{len(days)}")
    print(f"检测区,{startDate},{endDate},{ds}")
    # print(len(fv_dic_check), fv_dic_check)

    # 计算所有连续ds天的所有因子收益的统计数据
    DL = len(days)
    total_ffv = []
    for i in range(0, DL - ds + 1):
        ffv = make_fv_value_2(data, i, ds)
        total_ffv.append(ffv)
    total_data = numpy.array(total_ffv)
    # print(total_data.shape)
    total_mean = numpy.mean(total_data, axis=0)
    total_std = numpy.std(total_data, axis=0)
    # print(total_mean)
    # print(total_std)

    # 查找最大偏离度前5名的因子
    div = {}
    for i in range(len(FACTOR)):
        name = FACTOR[i]
        check_v = fv_dic_check[i]
        factor_mean = float(total_mean[i])
        factor_std = float(total_std[i])
        dv = (check_v - factor_mean) / factor_std
        absdv = abs(check_v - factor_mean) / factor_std
        div[name] = [round(absdv, 2), round(dv, 2)]
    div = dict(sorted(div.items(), key=lambda x: x[1][0], reverse=True))
    return div


def find_all_exnormal(totalArea, DS, fi, maxDiv):
    """
    在给定的时间区间内,查找任意连续DS天内,因子收益偏离度大于某个值的区间
    totalArea是长为2的list,为区间的首末日日期,DS为滑动窗口的长度(天),fi为整数表示第几个风格因子(或者风格因子名称字符串)
    maxDiv为要求的偏离度阈值
    返回数据为字典格式{key=numberID, value=[因子名称, 起始日期, 结束日期, 因子收益偏离度]}, 字典按起始日期项升序排列
    """

    if isinstance(fi, str):
        fi = FACTOR.index(fi)

    check_factor_file()

    with open(FACTOR_FILE, encoding="utf-8-sig") as file:
        data = json.load(file)
    days = list(data.keys())
    for k in days:
        if k < totalArea[0] or k > totalArea[-1]:
            data.pop(k)
    days = list(data.keys())
    print(f"总区间{days[0]}-{days[-1]}-总长度{len(days)}-采样天数{DS}-偏离度{maxDiv}")
    # 计算所有连续DS天的所有因子收益的统计数据
    DL = len(days)
    total_ffv = []
    for i in range(0, DL - DS + 1):
        ffv = make_fv_value_2(data, i, DS)
        total_ffv.append(ffv)
    total_data = numpy.array(total_ffv)
    total_mean = numpy.mean(total_data, axis=0)
    total_std = numpy.std(total_data, axis=0)

    sssv = {}
    name = FACTOR[fi]
    topKinfo = find_top_K(days, fi, total_ffv, DS, float(total_mean[fi]), float(total_std[fi]), maxDiv, -1)
    TEX = 1
    if len(topKinfo) > 0:
        for xf in topKinfo:
            if xf[2] >= maxDiv:
                sssv[TEX] = [name, xf[0], xf[1], round(xf[3], 2)]
                TEX += 1
    sssv = dict(sorted(sssv.items(), key=lambda x: x[1][1], reverse=False))
    return sssv


if __name__ == '__main__':
    fe = find_factor_exnormal(['20190101', '20260518'], ['20260312', '20260518'])
    print("因子,偏离度")
    for k, v in fe.items():
        print(f"{k},{v[1]:.2f}")

    print()
    fv = find_all_exnormal(['20190101', '20260518'], 44, 'size收益', 2.)
    print("因子,起始日期,结束日期,偏离度")
    for k, v in fv.items():
        print(f"{v[0]},{v[1]},{v[2]},{v[3]}")

    print()
    fv = find_all_exnormal(['20190101', '20260518'], 44, 'momentum收益', 2.)
    print("因子,起始日期,结束日期,偏离度")
    for k, v in fv.items():
        print(f"{v[0]},{v[1]},{v[2]},{v[3]}")
