"""
尝试建立更多维度的评估指标
为了方便团队协作共享, 我们规定当前脚本的基本输入是一个模型回测时一段时间内连续N个交易日的持仓文件
文件为json格式
"""
import argparse
import json
import os
import random
import numpy as np
import common_utils as CM
import matplotlib
from matplotlib.dates import AutoDateLocator
matplotlib.use('Agg')  # 设置后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import utils_barra as BA
import utils_calc as BC

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    # 程序会从上面这个path指定的路径下去查找所有的json文件, 并且要求json文件名中包含下面的match字符串
    # 如果match字符串为空,则默认查找path路径下所有的json文件
    # 另一个默认前提: 所有json文件名都以8位的日期开头
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--match_year', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='eval', help='标识,输出某些文档时以此命名')
    parser.add_argument('--out_path', type=str, default='', help='如果有输出文件时,保存到这个路径下')
    parser.add_argument('--baseArea', type=str, default='20190101,20251231', help='基准区')
    parser.add_argument('--checkArea', type=str, default='20260101,20261231', help='待检测区')
    parser.add_argument('--N', type=int, default=12, help='数据聚类的类别数')
    parser.add_argument('--window', type=int, default=5, help='窗口大小(天)')
    parser.add_argument('--stride', type=int, default=1, help='滚动步长,默认与window相同,则无重叠')
    args = parser.parse_args()
    assert args.window > 0, f"窗口长度必须大于1"
    if args.stride < 1 or args.stride > args.window:
        args.stride = args.window
    return args


def make_day_value_list(ba_dict, sname, f=None):
    """
    从barra字典中提取分日序列数据
    # ba_dict[day] = {'qa_exp': qa_exp, #--{'factor':value}
    #                 'qa_value': qa_value, #--{'factor':value}
    #                 'qa_alpha': qa_alpha, #--value
    #                 'pm_exp': pm_exp, #--{'factor':value}
    #                 'pm_value': pm_value, #--{'factor':value}
    #                 'pm_alpha': pm_alpha} #--value
    """
    dayList = list(ba_dict.keys())
    valueList = []
    if str(sname).endswith('alpha'):
        for d in dayList:
            v = ba_dict[d][sname]
            valueList.append(v)
    else:
        for d in dayList:
            v = ba_dict[d][sname][f]
            valueList.append(v)
    return dayList, valueList


def plot_image(args, title, tdays, data):
    """
    title[]  t0为当前这个图片展示的主要信息,它也会是文件名的构成部分
             t1为横坐标名称
             t2为纵坐标名称
    tdays 作为横坐标的日期序列
    data  数据序列字典[key=name, value=[]]
    """

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # kksize = 8
    # if len(tdays) < 50:
    #     kksize = 16
    # plt.rcParams['xtick.labelsize'] = kksize  # 设置所有图表x轴刻度标签的默认大小
    # plt.rcParams['ytick.labelsize'] = kksize  # 设置所有图表y轴刻度标签的默认大小

    # 1. 将日期字符串转换为datetime对象
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]  # 请根据实际格式修改日期解析格式

    # 2. 创建图形并绘制折线
    MK = ['.', ',', 'o', 'v', '^', '<', '>']
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    LS = ['-', '--', '-.', ':']
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    plt.figure(figsize=(32, 16))  # 设置图片大小，根据需要调整
    for dn, dv in data.items():
        p_mk = MK[mi]
        mi = (mi + 1) % len(MK)
        p_line = LS[ml]
        ml = (ml + 1) % len(LS)
        p_color = CO[mc]
        mc = (mc + 1) % len(CO)
        plt.plot(x, dv, label=dn, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)  # 绘制数据b

    # 3. 设置横坐标日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 设置日期格式
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 设置刻度间隔，避免过于密集
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签以避免重叠

    # 4. 添加标题和标签
    plt.title(f'{title[0]}-{args.mark}', fontsize=48)
    plt.xlabel(title[1], fontsize=24)
    plt.ylabel(title[2], fontsize=24)
    plt.legend()  # 显示图例

    # 5. 保存图片到文件
    imgname = f'{args.mark}_{title[0]}_{tdays[0]}_{tdays[-1]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return [imgname]


def output_csv(args, dayList, valueList, flag):
    fn = f'{args.mark}-{flag}.csv'
    fn = os.path.join(args.out_path, fn)
    f = open(fn, "w", encoding='utf-8-sig')
    print("日期,value", file=f)
    for i in range(len(dayList)):
        print(f"{dayList[i]},{valueList[i]}", file=f)
    f.close()


def make_image(args, dayList, valueList, flag):
    """
    根据日历数据序列绘制拆线图
    """

    images = []
    # 生成图片
    titles = [f"DAY-{flag}", 'date', 'value']
    data_dict = {
        f"{flag}": valueList,
    }
    images.extend(plot_image(args, titles, dayList, data_dict))
    return images


def split_data_list(dayList, valueList, baseArea, checkArea):
    """
    dayList,valueList是两个同等长度的列表
    baseArea,checkArea  str 'yyyymmdd,yyyymmdd'
    把日期列表和值列表按基础区和待检测区分开
    """

    baseD = []
    baseV = []
    checkD = []
    checkV = []
    baseDate = str(baseArea).split(',')
    checkDate = str(checkArea).split(',')
    DL = len(dayList)
    for i in range(DL):
        d = dayList[i]
        v = valueList[i]
        if baseDate[0] <= d <= baseDate[1]:
            baseD.append(d)
            baseV.append(v)
        if checkDate[0] <= d <= checkDate[1]:
            checkD.append(d)
            checkV.append(v)
    return baseD, baseV, checkD, checkV


def make_area_info(baseDays, baseValues, window, stride, CAL='average'):
    """
    baseDays 日期序列, baseValues 值序列, window窗口长度, stride滑动步长
    """
    baseInfo = {}
    WL = len(baseValues)
    if WL < window:
        raise ValueError(f"序列长度{WL}小于窗口长度{window}")
    dList = []
    vList = []
    w = window if CAL == 'average' else 1
    for i in range(0, WL - window + 1, stride):
        mark_d = baseDays[i]
        tmpVlist = baseValues[i:i + window]
        mark_v = sum(tmpVlist)/w # windows长序列的合计 or 均值
        dList.append(mark_d)
        vList.append(mark_v)
    baseInfo['days'] = dList
    baseInfo['values'] = vList
    return baseInfo


def make_bar_data(tvalues, N=20):
    """
    给定一个list,将它分成若干档,求得各档位内的计数
    """
    vmin = min(tvalues)
    vmax = max(tvalues)
    gap = (vmax - vmin) / N
    tname = []
    tcount = []
    for i in range(N):
        if i == 0:
            tn = f"~{gap:.4f}"
        elif i == N - 1:
            tn = f"{gap * (N - 1):.4f}~"
        else:
            tn = f"{i * gap:.4f}~{(i + 1) * gap:.4f}"
        tname.append(tn)
        tcount.append(0)

    for x in tvalues:
        seg = int((x - vmin) / gap)
        if seg < 0:
            seg = 0
        if seg >= N:
            seg = N - 1
        tcount[seg] = tcount[seg] + 1
    for i in range(N):
        tcount[i] = round(tcount[i] / len(tvalues), 2)
    return tname, tcount

def set_xticks_dynamic(ax, x_dates, date_strings, max_ticks=20):
    """
    动态设置x轴刻度，避免重叠
    ax: 坐标轴对象
    x_dates: datetime对象列表
    date_strings: 日期字符串列表，与x_dates对应
    max_ticks: 最大刻度数量
    """
    n = len(x_dates)
    if n <= max_ticks:
        # 如果日期数量小于等于最大刻度数，则全部显示
        ax.set_xticks(x_dates)
        ax.set_xticklabels(date_strings, rotation=45, ha='right')
    else:
        # 计算间隔
        interval = n // max_ticks
        if interval == 0:
            interval = 1
        # 选择要显示的刻度索引
        indices = list(range(0, n, interval))
        # 确保最后一个刻度包含在内
        if indices[-1] != n-1:
            indices.append(n-1)
        # 设置刻度位置和标签
        ax.set_xticks([x_dates[i] for i in indices])
        ax.set_xticklabels([date_strings[i] for i in indices], rotation=45, ha='right')

def plot_combined_charts(args, f, baseInfo, checkInfo):
    """
    对给出的基础信息和检测信息进行图形化展示
    baseInfo,checkInfo {'days': daylist, 'values':valuelist} ...
    """

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    locator = AutoDateLocator(minticks=5, maxticks=20)  # 设置最小和最大刻度数

    # 绘制基础信息的拆线图
    tdays = baseInfo['days']
    tvalues = baseInfo['values']
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]

    # 创建画布和子图
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(32, 32))

    # 样式选项
    MK = ['.', 'o', 'v', '^', '<', '>', 's']
    LS = ['-', '--', '-.', ':']
    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # 1. 绘制折线图（上部）
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    p_mk = MK[mi]
    p_line = LS[ml]
    p_color = CO[mc]
    ax1.plot(x, tvalues, label=f"{f}标准区间{tdays[0]}-{tdays[-1]}", marker=p_mk, linestyle=p_line,
             color=p_color, linewidth=2, markersize=5)

    ax1.set_title(f"{f}-D{args.window}S{args.stride}-采样点{len(tdays)}-标准区{tdays[0]}-{tdays[-1]}", fontsize=24)
    ax1.set_ylabel(f'D{args.window}均值', fontsize=18)
    # ax1.legend(fontsize=12)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    # ax1.xaxis.set_major_locator(locator)
    set_xticks_dynamic(ax1, x, tdays, max_ticks=40)
    ax1.grid(True, alpha=0.3)

    mean_val = np.mean(tvalues)
    std_val = np.std(tvalues)
    stats_text = f"均值 = {mean_val:.3f}\n标准差 = {std_val:.3f}"
    ax1.text(0.002, 0.998, stats_text,
            transform=ax1.transAxes,
            fontsize=24,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # plt.figtext(0.5, 0.01, "这是图表下方的说明文字",ha='center', fontsize=24, style='italic')

    # 2. 绘制柱状图（下部）
    # 首先生成分档数据(默认将数据分为N个类别,统计不同类别的计数)
    tName, tCount = make_bar_data(tvalues,N=args.N)
    bars = ax2.bar(tName, tCount, color=plt.cm.tab20c(np.arange(len(tName))))
    ax2.set_title(f"{f}-D{args.window}S{args.stride}-{args.N}档分布-标准区{tdays[0]}-{tdays[-1]}", fontsize=24)
    ax2.set_xlabel(f"D{args.window}均值区间", fontsize=24)
    ax2.set_ylabel("分布占比", fontsize=24)

    # 在每个柱子上显示权重值
    for bar, weight in zip(bars, tCount):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{weight*100:.2f}%',
                ha='center', va='bottom', fontsize=10)

    # 旋转x轴标签以便更好地显示
    # plt.xticks(rotation=45)
    # 调整布局并保存图片
    # plt.tight_layout()
    # ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 绘制检测区间信息的拆线图
    tdays = checkInfo['days']
    tvalues = checkInfo['values']
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)
    p_mk = MK[mi]
    p_line = LS[ml]
    p_color = CO[mc]
    ax3.plot(x, tvalues, label=f"{f}检测区间{tdays[0]}-{tdays[-1]}", marker=p_mk, linestyle=p_line,
             color=p_color, linewidth=2, markersize=5)
    ax3.set_title(f"{f}-D{args.window}S{args.stride}-采样点{len(tdays)}-检测区{tdays[0]}-{tdays[-1]}", fontsize=24)
    ax3.set_ylabel(f'D{args.window}均值', fontsize=18)
    ax3.legend(fontsize=12)
    # ax3.xaxis.set_major_locator(locator)
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    set_xticks_dynamic(ax3, x, tdays, max_ticks=40)
    ax3.grid(True, alpha=0.3)
    # mean_val = np.mean(tvalues)
    # std_val = np.std(tvalues)
    # stats_text = f"均值 = {mean_val:.3f}\n标准差 = {std_val:.3f}"
    results = BC.calculate_deviation_metrics(mean_val, std_val, tvalues)
    stats_text = results['detailed_description']
    ax3.text(0.002, 0.998, stats_text,
            transform=ax3.transAxes,
            fontsize=24,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 首先生成分档数据(默认将数据分为N个类别,统计不同类别的计数)
    tName, tCount = make_bar_data(tvalues, N=args.N)
    bars = ax4.bar(tName, tCount, color=plt.cm.tab20c(np.arange(len(tName))))
    ax4.set_title(f"{f}-D{args.window}S{args.stride}-{args.N}档分布-检测区{tdays[0]}-{tdays[-1]}", fontsize=24)
    ax4.set_xlabel(f"D{args.window}均值区间", fontsize=24)
    ax4.set_ylabel("分布占比", fontsize=24)
    # 在每个柱子上显示权重值
    for bar, weight in zip(bars, tCount):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{weight * 100:.2f}%',
                 ha='center', va='bottom', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 设置整个图的总标题
    # fig.suptitle(f'{f}-{args.mark}', fontsize=36, y=0.98)

    # 调整布局
    # 旋转x轴标签以便更好地显示
    plt.xticks(rotation=45)
    # 调整布局并保存图片
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # 保存图片
    imgname = f'{args.mark}-{f}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"生成组合图表<{image_filename}>成功...")
    return [imgname]

def make_cal_value_list(valueList):
    """
    将一个每日相对价值涨跌幅序列转换为一个累计价值序列
    """
    nv = []
    v = 1.
    for x in valueList:
        v = v * (1. + x)
        nv.append(v)
    return nv

def read_index_badic(args):
    """
    读取某个指数的barra数据
    """

    bafile = os.path.join('/data/indexData/', f"{args.indexName}_barra.json")
    if not os.path.exists(bafile):
        print(f"{bafile}文件不存在")
        return None
    with open(bafile, 'r', encoding='utf-8-sig') as file:
        ba_dict = json.load(file)
    return ba_dict


def read_badic(args):
    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取barra数据...")
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
    return ba_dict

if __name__ == '__main__':
    args = get_args()

    ba_dict = read_badic(args)

    # 生成每日数据序列-因子暴露
    exp_f = {}
    images = []
    # 因子暴露
    for f in BA.style_factor1:
        dayList, valueList = make_day_value_list(ba_dict, 'pm_exp', f)
        exp_f[f] = [dayList, valueList]
        # images.extend(make_image(args, dayList, valueList, f"暴露at风格-{f}"))
        if f == 'size':
            output_csv(args, dayList, valueList, f"暴露at风格-{f}")
    # 行业暴露
    # for f in BA.industry_factor:
    #     dayList, valueList = make_day_value_list(ba_dict, 'pm_exp', f)
    #     images.extend(make_image(args, dayList, valueList, f"暴露at行业-{f}"))
    # PDF.create_pdf_with_fpdf(images, f"{args.mark}-因子暴露", f"{args.mark}-因子暴露.pdf")

    # 因子收益
    value_f = {}
    images = []
    for f in BA.style_factor1:
        dayList, valueList = make_day_value_list(ba_dict, 'pm_value', f)
        value_f[f] = [dayList, valueList]
        # images.extend(make_image(args, dayList, valueList, f"收益at风格-{f}"))
    # 行业收益
    # for f in BA.industry_factor:
    #     dayList, valueList = make_day_value_list(ba_dict, 'pm_value', f)
    #     images.extend(make_image(args, dayList, valueList, f"收益at行业-{f}"))
    # alpha
    dayList, valueList = make_day_value_list(ba_dict, 'pm_alpha')
    value_f['alpha'] = [dayList, valueList]
    output_csv(args, dayList, valueList, f"日度alpha收益")
    v_cl = make_cal_value_list(valueList)
    output_csv(args, dayList, v_cl, f"累计alpha收益")

    # images.extend(make_image(args, dayList, valueList, "收益alpha"))

    # PDF.create_pdf_with_fpdf(images, f"{args.mark}-因子收益", f"{args.mark}-因子收益.pdf")

    # 检查因子暴露和因子收益的波动情况
    # 需要指定一个基准区间和一个回测区间,计算基准区间上的数据分布,然后计算测试区间的分离程度(平均偏离度/均值偏离度)
    for f in BA.style_factor1:
        dayList, valueList = exp_f[f]
        baseDays, baseValues, checkDays, checkValues = split_data_list(dayList, valueList, args.baseArea,
                                                                       args.checkArea)
        # 计算基准区间的时序值
        baseInfo = make_area_info(baseDays, baseValues, args.window, args.stride)
        # 计算检测区间的时序值
        checkInfo = make_area_info(checkDays, checkValues, args.window, args.stride)
        # 绘制图形
        plot_combined_charts(args, f"EXP-{f}", baseInfo, checkInfo)

    vf = ['alpha']
    vf.extend(BA.style_factor1)
    for f in vf:
        dayList, valueList = value_f[f]
        baseDays, baseValues, checkDays, checkValues = split_data_list(dayList, valueList, args.baseArea,
                                                                       args.checkArea)
        # 计算基准区间的时序值
        baseInfo = make_area_info(baseDays, baseValues, args.window, args.stride)
        # 计算检测区间的时序值
        checkInfo = make_area_info(checkDays, checkValues, args.window, args.stride)
        # 绘制图形
        plot_combined_charts(args, f"VAL-{f}", baseInfo, checkInfo)
