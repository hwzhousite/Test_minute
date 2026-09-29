"""
尝试建立更多维度的评估指标
为了方便团队协作共享, 我们规定当前脚本的基本输入是一个模型回测时一段时间内连续N个交易日的持仓文件
文件为json格式
"""
import argparse
import os
import random
from time import time
import numpy as np
import utils as UTILS
import common_utils as CM
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates


def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--tradeCost', type=float, default=0.001, help='交易成本(双向都计算)')
    parser.add_argument('--DS', type=str, default='day1062', help='数据集名称[day104,day106]')
    parser.add_argument('--targetPrice', type=str, default='close_hfq', help='交易价格默认为全天均价')
    parser.add_argument('--out_path', type=str, default='', help='如果有输出文件时,保存到这个路径下')

    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--path', type=str, default='', help='json文件所在的路径')
    # 程序会从上面这个path指定的路径下去查找所有的json文件, 并且要求json文件名中包含下面的match字符串
    # 如果match字符串为空,则默认查找path路径下所有的json文件
    # 另一个默认前提: 所有json文件名都以8位的日期开头
    parser.add_argument('--match', type=str, default='', help='匹配关键字,允许为空')
    parser.add_argument('--mark', type=str, default='eval', help='标识,输出某些文档时以此命名')

    args = parser.parse_args()

    assert args.path != '' and os.path.exists(args.path), f"必须指定合法存在的权重文件路径:{args.path}"
    if args.out_path != '' and (not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    if args.DS == 'day304':
        args.g_csv_path = r"/data/raw_generated_data/wind/304f_pad_csv"
        args.scaler_file = r"/data/yy_data/wind/304f_20100104/scaler_info.json"
    elif args.DS == 'day104':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/104f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/104f_250618/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    elif args.DS == 'day106':
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/106f_pad_csv"  # 过去80天数据读取路径
        args.scaler_file = r"/data/yy_data/tushare_data/106f_250619/scaler_info.txt"  # 数据集对应的字段字典表文件路径
    else:  # day111
        args.g_csv_path = r"/data/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = r"/data/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.DS, args.g_csv_path)

    return args


def reverse_dict_valueIsInt(ad):
    reversed_dic = {}
    for k, v in ad.items():
        v = int(round(v))
        reversed_dic[v] = k
    return reversed_dic


def read_price(priceType, weights):
    """
    读取所有交易日的股票交易价格,对每个交易日来说,都要先生成一个前日/今日这2天
    所有持仓股票并集的所有价格信息,因为后续在计算邻近两日的相对价值时要用到
    """

    priceDict = {}  # 生成价格数据字典, key为日期, value是另一个字典[key=stockcode,value=price]
    dayslist = list(weights.keys())
    daysLen = len(dayslist)
    for i in range(daysLen):
        dA = dayslist[i]  # 交易日
        sA = list(weights[dA].keys())  # 交易日持股
        if i > 0:
            # 从第二个交易日起, 当天的价格信息必须得包含前一天所有持仓的股票
            # 因为对于今日清仓的股票, 在权重表中是不存在的, 必须要加进来, 才能在下面的方法中去取到它在今天的卖出价
            dPrev = dayslist[i - 1]
            sPrev = list(weights[dPrev].keys())
            sA = set(sA) | set(sPrev)
            sA = list(sA)
        if 'CASH' in sA:
            sA.remove('CASH')
        # 从文件中读取这个股票列表在当日的价格
        price = UTILS.read_group_price(sA, priceType, dA, force_read_file=True)
        priceDict[dA] = price
        print(f"{CM.timestr()}#{i + 1}/{daysLen}-{dA}...")
    return priceDict


def make_no_cash_weight(w):
    totalW = sum(w.values())
    totalW -= w['CASH']
    nw = {}
    for c, v in w.items():
        if c == 'CASH':
            continue
        nw[c] = v / totalW
    return nw





def plot_imge_2line(args, title, tdays, data):
    assert len(data) == 2, f"数据序列必须为2"
    keys = list(data.keys())
    # 示例数据：序列A（0-100），序列B（-5-5）
    x = [datetime.strptime(date_str, '%Y%m%d') for date_str in tdays]
    y1 = data[keys[0]]
    y2 = data[keys[1]]

    # 创建图形和第一个纵坐标轴
    fig, ax1 = plt.subplots(figsize=(32, 16))

    # 绘制第一个数据序列（左侧Y轴）
    color1 = 'tab:red'
    ax1.set_xlabel('时间（单位）')
    ax1.set_ylabel(keys[0], color=color1, fontsize=12)
    line1 = ax1.plot(x, y1, color=color1, linewidth=2, label=keys[0])
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(y1), max(y1))  # 明确设置左侧Y轴范围

    # 创建第二个纵坐标轴（共享同一横轴）
    ax2 = ax1.twinx()

    # 绘制第二个数据序列（右侧Y轴）
    color2 = 'tab:blue'
    ax2.set_ylabel(keys[1], color=color2, fontsize=12)
    line2 = ax2.plot(x, y2, color=color2, linestyle='--', linewidth=2, label=keys[1])
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(min(y2), max(y2))  # 明确设置右侧Y轴范围

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 设置日期格式
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 设置刻度间隔，避免过于密集
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签以避免重叠

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 添加标题并展示
    plt.title(f"{title[0]}-{args.mark}")
    plt.tight_layout()
    # plt.show()

    # 保存图片到文件
    imgname = f'{args.mark}_{title[0]}_{tdays[0]}_{tdays[-1]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


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
    return imgname


def plot_image_rect(args, title, tdays, weights):
    """
    title[]  t0为当前这个图片展示的主要信息,它也会是文件名的构成部分
             t1为横坐标名称
             t2为纵坐标名称
    tdays 作为横坐标的序列(字符串,并不要求它是日期)
    data  数据序列字典[key=name, value=[]]
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # kksize = 10
    # if len(tdays)<50:
    #     kksize = 16
    # plt.rcParams['xtick.labelsize'] = kksize  # 设置所有图表x轴刻度标签的默认大小
    # plt.rcParams['ytick.labelsize'] = kksize  # 设置所有图表y轴刻度标签的默认大小

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(32, 16))
    bars = ax.bar(tdays, weights, color=plt.cm.tab20c(np.arange(len(tdays))))

    # 设置图表标题和轴标签
    plt.title(f'{title[0]}-{args.mark}', fontsize=48)
    plt.xlabel(title[1], fontsize=24)
    plt.ylabel(title[2], fontsize=24)

    # 在每个柱子上显示权重值
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{weight:.2f}',
                ha='center', va='bottom', fontsize=10)

    # 旋转x轴标签以便更好地显示
    plt.xticks(rotation=45)

    # 调整布局并保存图片
    plt.tight_layout()

    imgname = f'{args.mark}_{title[0]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_image_rect_multi(args, title, tdays, data_dict):
    # tdays为横坐标序列,  data_dict{key=子序列名,value=[子序列]}
    # 多个数据序列（例如：不同时间段的权重数据）
    series_labels = list(data_dict.keys())
    data_series = list(data_dict.values())

    # 调用函数绘制多序列柱状图
    plot_multi_series_bar_chart(
        tdays=tdays,
        data_series=data_series,
        series_labels=series_labels,
        title=f'{title[0]}',
        xlabel=f'{title[1]}',
        ylabel=f'{title[2]}',
    )

    # 保存图片
    imgname = f'{args.mark}_{title[0]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=400, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def plot_multi_series_bar_chart(tdays, data_series, series_labels, title, xlabel, ylabel, figsize=(32, 16)):
    """
    绘制多序列柱状图

    参数:
    tdays: 横坐标标签（类别）
    data_series: 包含多个数据序列的列表，每个序列对应一个数据系列
    series_labels: 每个数据系列的标签（用于图例）
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    figsize: 图表尺寸
    """

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=figsize)

    n_series = len(data_series)  # 数据系列数量
    n_categories = len(tdays)  # 类别数量

    # 计算柱子的位置和宽度[1,2](@ref)
    x = np.arange(n_categories)  # 类别位置
    width = 0.8 / n_series  # 柱子宽度，根据系列数量动态调整

    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    # 为每个数据系列绘制柱子[1](@ref)
    bars = []
    for i, (data, label) in enumerate(zip(data_series, series_labels)):
        p_color = CO[mc]
        mc = (mc + 1) % len(CO)
        # 计算当前系列柱子的x轴位置（居中排列）[1](@ref)
        x_pos = x + width * (i - (n_series - 1) / 2)
        bar = ax.bar(x_pos, data, width, label=label, color=p_color)  # color=plt.cm.tab20c(i))  # 使用不同的颜色
        bars.append(bar)

        # 在每个柱子上显示数值[1](@ref)
        for j, (bar_obj, value) in enumerate(zip(bar, data)):
            height = bar_obj.get_height()
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10)

    # 设置图表标题和轴标签
    ax.set_title(f'{title}', fontsize=48)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    # 设置x轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(tdays, rotation=45)

    # 添加图例
    ax.legend(fontsize=20)

    # 调整布局
    plt.tight_layout()

    return fig, ax


def is_period_end_day_in_list(daylist, index, period):
    DLen = len(daylist)
    if index < 0 or index >= DLen:
        return False
    if index == DLen - 1:
        return True
    year, month, day, weekday = CM.parse_date(daylist[index])
    y2, m2, d2, w2 = CM.parse_date(daylist[index + 1])
    if period == 'month':
        return m2 != month  # day(i)与day(i+1)不是同一个月了,表明i是月末

    di = datetime.strptime(daylist[index], "%Y%m%d")
    wi = int(di.strftime('%V'))

    dip1 = datetime.strptime(daylist[index + 1], "%Y%m%d")
    wip1 = int(dip1.strftime('%V'))

    return wi != wip1


def function01(args, tdays, model_value, market_value_dict, period='day'):
    # 将净值与大盘绘制到同一图内
    # model_value是每天相对前一天的相对价值, market_value也是

    # 构造出模型的净值序列
    v = 1.
    mdv = []
    for x in model_value.values():
        v = v * x
        mdv.append(v)
    # print("MODEL:",mdv)

    # 构造出大盘的净值序列---对每种指标都生成一个序列
    fdv = {}
    for mz, mzvlist in market_value_dict.items():
        v = 1.
        mvlist = []
        for x in list(mzvlist.values()):
            v = v * x
            mvlist.append(v)
        fdv[mz] = mvlist

    DLen = len(tdays)
    if period == 'week' or period == 'month':
        # 按周/月取值,则需要从完整的日序列数据中挑选出所有周/月末数据项(即每月的最后一个交易日)
        p_end_mdv = []
        # p_end_fdv = []
        p_end_day = []
        newMZ = {}
        for mz in list(market_value_dict.keys()):
            newMZ[mz] = []
        for i in range(DLen):
            if is_period_end_day_in_list(tdays, i, period):
                # 如果第i天刚好是一个周末/月末
                p_end_mdv.append(mdv[i])
                for mz in list(newMZ.keys()):
                    newMZ[mz].append(fdv[mz][i])
                p_end_day.append(tdays[i])
        mdv = p_end_mdv
        fdv = newMZ
        tdays = p_end_day

    # 生成图片
    titles = [f'整体性能-净值曲线-{period}', '日期', '净值']
    data_dict = {
        "基金净值": mdv,
    }
    data_dict.update(fdv)
    return plot_image(args, titles, tdays, data_dict)


def search_value_in_list(x, ex_type_seg):
    """
    在一个有序列表中查找x所在的位置, 即ex_type_seg中n个点把数轴分为n+1份
    返回x在这n+1个区间中的哪一个, 从0开始计数.
    """
    DL = len(ex_type_seg)
    for i in range(DL):
        if x < ex_type_seg[i]:
            return i
    return DL


def function02(args, tdays, model_value, market_index_name, market_value, period='week'):
    DLen = len(tdays)

    # 构造出模型的每周累计净值
    vm = 1.
    vf = 1.
    mdv = []  # 模型在每个期末时的累计净值(本周内的)
    fdv = []  # 大盘在每个期末时的累计净值(本周内的)
    ddv = []  # 期末的日期
    for i in range(DLen):
        if period == 'day':
            mdv.append(model_value[tdays[i]])
            fdv.append(market_value[tdays[i]])
            ddv.appen(tdays[i])
        else:
            vm = vm * model_value[tdays[i]]
            vf = vf * market_value[tdays[i]]
            if is_period_end_day_in_list(tdays, i, period):
                mdv.append(vm)
                fdv.append(vf)
                ddv.append(tdays[i])
                vm = 1.
                vf = 1.

    # 计算每期(日/周/月)的超额
    DLen = len(ddv)
    for i in range(DLen):
        mdv[i] = (mdv[i] - fdv[i]) / fdv[i] * 100.

    # 把每周的超额进行分箱,观察其出现的次数
    # 算法 1.求出最大/最小值, 2.将其均分为20份得到19个点位值
    ex_type = {}
    ex_name = []
    ex_type_seg = []
    ex_max = 10.  # max(mdv)
    ex_min = -10.  # min(mdv)
    tpN = 20
    segLength = (ex_max - ex_min) / tpN
    for i in range(tpN):
        segMin = ex_min + i * segLength
        segMax = ex_min + (i + 1) * segLength
        kn = f"[{segMin:.2f}%,{segMax:.2f}%)"
        ex_name.append(kn)
        ex_type[i] = 0
        if i > 0:
            ex_type_seg.append(segMax)  # 生成了tpN-1个点位值
    for x in mdv:
        xi = search_value_in_list(x, ex_type_seg)  # 查找x在这tpN个区间中的哪一个, 从0开始计数,最大为tpN-1
        ex_type[xi] += 1
    totalTypes = sum(ex_type.values())
    for i in range(len(ex_type)):
        ex_type[i] = (ex_type[i] / totalTypes) * 100

    img = plot_image(args, [f"整体性能-超额收益-{period}-{market_index_name}", '日期', '超额收益%'], ddv,
                     {'超额收益': mdv})
    img2 = plot_image_rect(args,
                           [f"整体性能-超额收益-{period}-{market_index_name}-{tpN}档分布", '超额区间', '次数占比%'],
                           ex_name,
                           list(ex_type.values()))
    return [img, img2]


def resort_dict(unsorted_dict, rerverse=False):
    """
    对于key-value字典进行排序
    """

    sw = {}
    sorted_w = sorted(unsorted_dict.items(), key=lambda x: x[1], reverse=rerverse)
    for i in range(len(sorted_w)):
        stock_code = sorted_w[i][0]
        stock_weight = sorted_w[i][1]
        sw[stock_code] = stock_weight
    return sw


def cal_bj_stock_weight(w):
    bjw = 0
    for k, v in w.items():
        if str(k).endswith('.BJ'):
            bjw += v
    return bjw


def function0301(args, weights, market_value_bj):
    """
    对北交所股票权重占比进行分析
    """

    bj_days = []  # 整个交易期中,恰好与北交所日期区间重合的部分
    bj_weights = []  # 该日期的北交所权重占比
    bj_change = []  # 该日期的北证指数涨跌幅
    for day, w in weights.items():
        # 遍历每一天的投资权重图
        if not market_value_bj.__contains__(day):
            continue
        bj_days.append(day)
        bj_weights.append(cal_bj_stock_weight(w) * 100.)
        bj_change.append((market_value_bj[day] - 1) * 100)
    titles = [f'投资分析-北交所股票权重占比{bj_days[0]}-{bj_days[-1]}', '日期', '权重-指数涨跌幅']
    data_dict = {
        "北交所股票占比": bj_weights,
        "北证指数涨跌幅": bj_change,
    }
    return plot_imge_2line(args, titles, bj_days, data_dict)


def function03(args, weights, all_stock_info, sttop, industry_info, market_info):
    """
    对投资标的进行归类分析
    """

    images = []
    inds_weight = {}  # 各行业总权重
    market_weight = {}  # 各板块总权重
    single_stock_weight = {}  # 单只股票权重
    cmv_type = ['大盘股500+', '中盘股100+', '小盘股50+', '微盘股50-']
    cmv_weight = {0: 0.,
                  1: 0.,
                  2: 0.,
                  3: 0.
                  }

    datafiels = list(UTILS.g_field_index.keys())
    inid = CM.find_str_in_list(datafiels, 'gen_industry')
    maid = CM.find_str_in_list(datafiels, 'gen_market')
    mvid = CM.find_str_in_list(datafiels, 'mv')
    for day, w in weights.items():
        # 遍历每一天的投资权重图
        for stock, stock_weight in w.items():
            if stock == 'CASH':
                continue
            # 获取股票所在的行业和板块
            if all_stock_info.__contains__(stock):
                stockdata = all_stock_info[stock]
                sday = list(stockdata.keys())
                sday = sday[0]  # 最早的一天,不能取最后一天,因为最后一天可能已经退市,全字段都是0
                sdv = stockdata[sday]
                IV = int(round(sdv[inid]))
                MV = int(round(sdv[maid]))
                stock_cmv = float(sdv[mvid]) / 10000.  # 流通市值
            else:
                raise ValueError(f"未能读取股票{stock}数据")

            if stock_cmv >= 500:
                cmv_weight[0] += stock_weight
            elif stock_cmv >= 100:
                cmv_weight[1] += stock_weight
            elif stock_cmv >= 50:
                cmv_weight[2] += stock_weight
            else:
                cmv_weight[3] += stock_weight
            if inds_weight.__contains__(IV):
                inds_weight[IV] = inds_weight[IV] + stock_weight
            else:
                inds_weight[IV] = stock_weight
            if market_weight.__contains__(MV):
                market_weight[MV] = market_weight[MV] + stock_weight
            else:
                market_weight[MV] = stock_weight
            if single_stock_weight.__contains__(stock):
                single_stock_weight[stock] = single_stock_weight[stock] + stock_weight
            else:
                single_stock_weight[stock] = stock_weight

    # 重仓股
    single_stock_weight = resort_dict(single_stock_weight, True)
    totalWeight = sum(single_stock_weight.values())
    topNstock = []
    topNstock_w = []
    stocklist = list(single_stock_weight.keys())
    for i in range(len(stocklist)):
        topNstock.append(stocklist[i])
        topNstock_w.append(single_stock_weight[stocklist[i]] / totalWeight * 100.)
        if i >= sttop - 1:
            break
    stockImg = plot_image_rect(args, [f"投资分析-重仓股TOP{sttop}", '股票代码', '权重占比%'], topNstock, topNstock_w)
    images.append(stockImg)

    # 分行业看投资标的的分布
    insList = list(inds_weight.keys())
    insList = sorted(insList)  # 从小到大排序
    insName = []
    insValue = []
    totalWeight = sum(inds_weight.values())
    for x in insList:
        if industry_info.__contains__(x):
            xname = industry_info[x]
        else:
            xname = '其他'
        insName.append(xname)
        insValue.append(inds_weight[x] / totalWeight * 100)
    # print("---",len(insName),insName)
    # print(len(insValue),insValue)
    stockImg = plot_image_rect(args, [f"投资分析-行业分布", '行业名称', '权重占比%'], insName, insValue)
    images.append(stockImg)

    # 分板块看投资标的的分布
    insList = list(market_weight.keys())
    insList = sorted(insList)  # 从小到大排序
    insName = []
    insValue = []
    totalWeight = sum(market_weight.values())
    for x in insList:
        if market_info.__contains__(x):
            xname = market_info[x]
        else:
            xname = '其他'
        insName.append(xname)
        insValue.append(market_weight[x] / totalWeight * 100)
    # print("---", len(insName), insName)
    # print(len(insValue), insValue)
    stockImg = plot_image_rect(args, [f"投资分析-板块分布", '板块名称', '权重占比%'], insName, insValue)
    images.append(stockImg)

    # 分按市值看投资标的的分布
    totalWeight = sum(cmv_weight.values())
    insValue = []
    for x, v in cmv_weight.items():
        insValue.append(v / totalWeight * 100)
    stockImg = plot_image_rect(args, [f"投资分析-流通市值分布", '按市值分类', '权重占比%'], cmv_type, insValue)
    images.append(stockImg)

    return images


def make_net_win_lost(win_dict, lost_dict):
    stock_net_value = {}
    for day, data in win_dict.items():
        for stock, wl in data.items():
            if stock_net_value.__contains__(stock):
                stock_net_value[stock] += wl
            else:
                stock_net_value[stock] = wl
    for day, data in lost_dict.items():
        for stock, wl in data.items():
            if stock_net_value.__contains__(stock):
                stock_net_value[stock] -= wl
            else:
                stock_net_value[stock] = -wl
    wnet = {}
    lnet = {}
    for x, v in stock_net_value.items():
        if v >= 0:
            wnet[x] = v
        else:
            lnet[x] = -v
    return wnet, lnet


def function04(args, win_dict, lost_dict, all_stock_info, topN, industory_info):
    """
    盈亏分析
    """

    # 根据盈亏明细构造出对于全部所有股票的净盈亏
    net_win, net_lost = make_net_win_lost(win_dict, lost_dict)

    W_inds_weight = {}  # 各行业盈利
    L_inds_weight = {}  # 各行业亏损
    cmv_type = ['大盘股500+', '中盘股100+', '小盘股50+', '微盘股50-']
    cmv_weight_win = {0: 0.,
                      1: 0.,
                      2: 0.,
                      3: 0.
                      }
    cmv_weight_lost = {0: 0.,
                       1: 0.,
                       2: 0.,
                       3: 0.
                       }

    datafiels = list(UTILS.g_field_index.keys())
    inid = CM.find_str_in_list(datafiels, 'gen_industry')
    mvid = CM.find_str_in_list(datafiels, 'mv')
    for stock, win_weight in net_win.items():
        # 遍历所有盈利股票
        # 获取股票所在的行业和板块
        if all_stock_info.__contains__(stock):
            stockdata = all_stock_info[stock]
            sday = list(stockdata.keys())
            sday = sday[0]  # 最近的一天
            sdv = stockdata[sday]
            IV = int(round(sdv[inid]))
            stock_cmv = float(sdv[mvid]) / 10000.  # 流通市值
        else:
            # IV =10000
            raise ValueError(f"----{stock}----")
        if stock_cmv >= 500:
            cmv_weight_win[0] += win_weight
        elif stock_cmv >= 100:
            cmv_weight_win[1] += win_weight
        elif stock_cmv >= 50:
            cmv_weight_win[2] += win_weight
        else:
            cmv_weight_win[3] += win_weight
        if W_inds_weight.__contains__(IV):
            W_inds_weight[IV] = W_inds_weight[IV] + win_weight
        else:
            W_inds_weight[IV] = win_weight
    for stock, lost_weight in net_lost.items():
        # 遍历每一天的盈利占比图
        # 获取股票所在的行业和板块
        if all_stock_info.__contains__(stock):
            stockdata = all_stock_info[stock]
            sday = list(stockdata.keys())
            sday = sday[0]  # 最近的一天
            sdv = stockdata[sday]
            IV = int(round(sdv[inid]))
            stock_cmv = float(sdv[mvid]) / 10000.  # 流通市值
        else:
            raise ValueError(f"----{stock}----")
        if stock_cmv >= 500:
            cmv_weight_lost[0] += lost_weight
        elif stock_cmv >= 100:
            cmv_weight_lost[1] += lost_weight
        elif stock_cmv >= 50:
            cmv_weight_lost[2] += lost_weight
        else:
            cmv_weight_lost[3] += lost_weight
        if L_inds_weight.__contains__(IV):
            L_inds_weight[IV] = L_inds_weight[IV] + lost_weight
        else:
            L_inds_weight[IV] = lost_weight

    imglist = []
    # 盈利个股
    single_stock_weight = resort_dict(net_win, True)
    totalWeight = sum(single_stock_weight.values())
    topNstock = []
    topNstock_w = []
    stocklist = list(single_stock_weight.keys())
    for i in range(len(stocklist)):
        topNstock.append(stocklist[i])
        topNstock_w.append(single_stock_weight[stocklist[i]] / totalWeight * 100.)
        if i >= topN - 1:
            break
    img = plot_image_rect(args, [f"盈亏分析-个股-盈利TOP{topN}", '股票代码', '盈利占比%'], topNstock, topNstock_w)
    imglist.append(img)

    # 亏损个股
    single_stock_weight = resort_dict(net_lost, True)
    totalWeight = sum(single_stock_weight.values())
    topNstock = []
    topNstock_w = []
    stocklist = list(single_stock_weight.keys())
    for i in range(len(stocklist)):
        topNstock.append(stocklist[i])
        topNstock_w.append(single_stock_weight[stocklist[i]] / totalWeight * 100.)
        if i >= topN - 1:
            break
    img = plot_image_rect(args, [f"盈亏分析-个股-亏损TOP{topN}", '股票代码', '亏损占比%'], topNstock, topNstock_w)
    imglist.append(img)

    # 盈利分行业
    insList = list(W_inds_weight.keys())
    insList = sorted(insList)  # 从小到大排序
    insName = []
    insValue = []
    totalWeight = sum(W_inds_weight.values())
    for x in insList:
        if industory_info.__contains__(x):
            xname = industory_info[x]
        else:
            xname = '其他'
        insName.append(xname)
        insValue.append(W_inds_weight[x] / totalWeight * 100)
    img = plot_image_rect(args, [f"盈亏分析-行业分布-盈利", '行业名称', '盈利占比%'], insName, insValue)
    imglist.append(img)

    # 亏损分行业
    insList = list(L_inds_weight.keys())
    insList = sorted(insList)  # 从小到大排序
    insName = []
    insValue = []
    totalWeight = sum(L_inds_weight.values())
    for x in insList:
        if industory_info.__contains__(x):
            xname = industory_info[x]
        else:
            xname = '其他'
        insName.append(xname)
        insValue.append(L_inds_weight[x] / totalWeight * 100)
    img = plot_image_rect(args, [f"盈亏分析-行业分布-亏损", '行业名称', '亏损占比%'], insName, insValue)
    imglist.append(img)

    cmv_data_dict = {}
    # 按市值看盈利的分布
    totalWeight = sum(cmv_weight_win.values())
    insValue = []
    for x, v in cmv_weight_win.items():
        insValue.append(v / totalWeight * 100)
    cmv_data_dict['盈利'] = insValue
    # img = plot_image_rect(args, [f"盈亏分析-流通市值分布-盈利", '按市值分类', '权重占比%'], cmv_type, insValue)
    # imglist.append(img)

    # 按市值看亏损的分布
    totalWeight = sum(cmv_weight_lost.values())
    insValue = []
    for x, v in cmv_weight_lost.items():
        insValue.append(v / totalWeight * 100)
    cmv_data_dict['亏损'] = insValue
    # img = plot_image_rect(args, [f"盈亏分析-流通市值分布-亏损", '按市值分类', '权重占比%'], cmv_type, insValue)
    # imglist.append(img)

    img = plot_image_rect_multi(args, [f"盈亏分析-流通市值分布", '按市值分类', '权重占比%'], cmv_type, cmv_data_dict)
    imglist.append(img)

    return imglist


def function05(args, weights, all_stock_info, HoldMaxDay):
    """
    持股天数及盈亏相关分析, 总体思想: 将所有持股行为统计从买入第1手一直到最后全部清空计为一个mini投资行为
    统计所有mini投资行为的时长分布,指定最长为HoldMaxDay, 则一共分为HoldMaxDay+1个档位
    将所有的mini投资行为分别归类到这些档位, 并分别记录它是赚是赔(忽略中间过程,只考量最后卖出时的股票价格是否高于买入那天的价格)
    最后那一天的持仓不做统计,因为不知道会在什么时候卖出, 如果在查找卖出点时已经到达了所有日期列表的最后一天,则持仓天数就只统计到最后一天(即:假定次日即卖出)
    """

    # 从每日权重图中查找到当日新出现的股票(相对于前一交易日)
    # 目标是构造出一个list holdList[[stock,buyinday,buyinprice,sellprice,keepdays]]
    holdList = []
    daysList = list(weights.keys())
    daysLen = len(daysList)
    for i in range(daysLen - 1):  # 最后那一天的持仓不做统计,因为不知道会在什么时候卖出
        for stock, wt in weights[daysList[i]].items():
            # 遍历当天的所有持仓数据
            if stock == 'CASH':
                continue
            # 判断该股票是否为当天新出现(即前一日期的权重图中没有它!!!)
            if i == 0 or (i > 0 and (not weights[daysList[i - 1]].__contains__(stock))):
                # 当前为首日, 或者非首日但前一日不含这只股票
                # 生成一个需要记录的数据项

                buyInDay = daysList[i]  # 买入日期
                buyInPrice = read_field_from_all(all_stock_info, stock, buyInDay, args.targetPrice)  # 买入价格
                keepDays = 1  # 持有天数
                sellDay = None  # 卖出日期
                for j in range(i + 1, daysLen):
                    # 从次日开始查找这同一只股票,直到它不出现的那一天,即为卖出日
                    # 按照当前的代码逻辑,全局缓存中是一定存在这一天的这只股票价格信息的,否则应该报错中断
                    wj = weights[daysList[j]]
                    if wj.__contains__(stock):
                        # 如果j日继续持仓,则继续向后查找
                        keepDays += 1
                    else:
                        # 否则退出循环
                        keepDays += 1
                        sellDay = daysList[j]
                        break
                keepDays -= 1
                if sellDay is None:
                    # 特殊情况,股票一直持仓到了最后一天
                    sellDay = daysList[-1]
                sellPrice = read_field_from_all(all_stock_info, stock, sellDay, args.targetPrice)  # 卖出价格
                miniData = [stock, buyInDay, buyInPrice, sellPrice, keepDays]
                holdList.append(miniData)

    # 生成统计数据
    keep_dict = {}
    keep_name = []
    for i in range(1, HoldMaxDay + 2):
        keep_dict[i] = [0, 0, 0]  # 持有天数为i天, 数据项为[mini投资行为个数,赚钱次数,赔钱次数]
        if i <= HoldMaxDay:
            keep_name.append(f"持有{i}天")
        else:
            keep_name.append(f"持有{HoldMaxDay}天以上")
    # 填充数据
    totalMini = 0
    for miniData in holdList:
        buyInPrice = miniData[2]
        sellPrice = miniData[3]
        keepDays = miniData[4]
        if keepDays > HoldMaxDay:
            keepDays = HoldMaxDay + 1
        keep_dict[keepDays][0] += 1  # 总次数加1
        totalMini += 1
        if sellPrice >= buyInPrice:
            # 只考虑卖出价是否大于买入价(不考虑交易成本)
            keep_dict[keepDays][1] += 1  # 赚钱的次数加1
        else:
            keep_dict[keepDays][2] += 1  # 亏钱的次数加1

    # 生成绘图数据
    imglist = []
    keep_days_list = []
    for i in range(1, HoldMaxDay + 2):
        pct = keep_dict[i][0] / totalMini * 100.
        keep_days_list.append(pct)
    # 按持有股票的天数分为(HoldMaxDay+1)个档位,绘制柱状图
    img = plot_image_rect(args, [f"持股时长分析-{HoldMaxDay + 1}档-行为分布", '持股天数', 'mini投资次数占比%'],
                          keep_name, keep_days_list)
    imglist.append(img)

    # 同样按这些档位,绘制双柱图,分别显示各个档位上的盈亏比例
    kwin = []
    klost = []
    gap = []
    for i in range(1, HoldMaxDay + 2):
        if keep_dict[i][0] <= 0:
            # 该类总次数为0,无法计算百分比
            wp = 0
            lp = 0
        else:
            wp = keep_dict[i][1] / keep_dict[i][0] * 100.
            lp = keep_dict[i][2] / keep_dict[i][0] * 100.
        kwin.append(wp)
        klost.append(lp)
        gap.append(wp - lp)  # 盈亏差距, 简化来看, 应该这个差距越大越好,表明按mini投资的次数来讲胜率更高
    keep_win_lost_dict = {
        "盈利占比": kwin,
        "亏损占比": klost,
        "盈亏差": gap,
    }
    img = plot_image_rect_multi(args, [f"持股时长分析-{HoldMaxDay + 1}档-结果分布", '持股天数', '盈亏占比%'], keep_name,
                                keep_win_lost_dict)
    imglist.append(img)

    return imglist


def function06(args, weights, baseIndexList, period='week'):
    factor = {'size': '市值', 'beta': 'beta', 'momentum': '动量',
              'non_linear_size': '非线性市值', 'bp': '账面市值比', 'earnings_yield': '盈利能力',
              'growth': '成长', 'leverage': '杠杆', 'liquidity': '流动性', 'volatility': '残差波动率'}

    imgList = []
    all_dict = {}
    # 构造目标数据,二级字典, 一级key为指数名称,二级key为factor名称,三级key为日期列表
    for day, w in weights.items():
        # 遍历所有的日期/权重
        # 生成当天的指标-因子暴露数据
        no_cash_weight = make_no_cash_weight(w)
        dF = UTILS.get_exposure(day, no_cash_weight, baseIndexList)
        for indName, fatorDic in dF.items():
            if not all_dict.__contains__(indName):
                all_dict[indName] = {}
            # 遍历所有的因子
            for fn, fv in fatorDic.items():
                if not all_dict[indName].__contains__(fn):
                    all_dict[indName][fn] = {}
                all_dict[indName][fn][day] = fv

    for baseName in baseIndexList:
        # 对于每个基准指标,都需要生成所有因子暴露对照表
        for fk_id, fc_CHName in factor.items():
            bV1 = list(all_dict[baseName][fk_id].values())
            bV2 = list(all_dict['thisPM'][fk_id].values())
            # 生成图片
            dayList = list(all_dict['thisPM'][fk_id].keys())

            if period != 'day':
                dayList, bV1, bV2 = make_period_avg_datalist(dayList, bV1, bV2, period)

            titles = [f'BARRA-{baseName}-因子-{fc_CHName}-{period}', '日期', '因子暴露值']
            data_dict = {
                f"{baseName}": bV1,
                f"本投资组合": bV2
            }

            # print(f"{baseName}-{fc_CHName}-dL:{len(dayList)}-{len(bV_index_value)}-{len(bV_PM_value)}")
            imgList.append(plot_image(args, titles, dayList, data_dict))

    return imgList


def make_period_avg_datalist(dayList, bV1, bV2, period):
    DL = len(dayList)
    V1 = []
    V2 = []
    ret_V1 = []
    ret_V2 = []
    ret_daylis = []
    for i in range(DL):
        day = dayList[i]
        V1.append(bV1[i])
        V2.append(bV2[i])
        if is_period_end_day_in_list(dayList, i, period):
            ret_daylis.append(day)
            ret_V1.append(sum(V1) / len(V1))
            ret_V2.append(sum(V2) / len(V2))
            V1.clear()
            V2.clear()
    return ret_daylis, ret_V1, ret_V2


def read_field_from_all(all_stock_info, stock, day, fieldName):
    """
    从缓存的全局数据中读取一项, 这里假定所有的参数都是合法有效的
    全局数据格式为{key=stock,value={key=day,value=[...]}}
    """

    if not all_stock_info.__contains__(stock):
        raise ValueError(f"全局缓存中不存在股票{stock}")
    stockData = all_stock_info[stock]
    if not stockData.__contains__(day):
        raise ValueError(f"全局缓存中股票{stock}数据中不存在日期{day}")
    stockData = stockData[day]  # 这里得到了一个valueList

    # 得到当下要读取的这个字段在整个valueList中的下标
    datafiels = list(UTILS.g_field_index.keys())
    pi = CM.find_str_in_list(datafiels, fieldName)
    return stockData[pi]


def make_price_info_from_totalinfo(targetPrice, all_stock_info):
    # 从全局数据中读取价格数据,全局数据格式为{key=stock,value={key=day,value=[...]}}
    # 生成以日期为一级索引/股票为二级索引的字典{key=day,value={key=stock,value=pricevalue}}
    priceInfo = {}
    datafiels = list(UTILS.g_field_index.keys())
    pi = CM.find_str_in_list(datafiels, targetPrice)
    for stock, data in all_stock_info.items():
        # data{key=date,value=[]}
        for day, dv in data.items():
            if not priceInfo.__contains__(day):
                priceInfo[day] = {}
            priceInfo[day][stock] = dv[pi]
    # alldays = list(priceInfo.keys())
    # print(f"全部日期{len(alldays)}----",alldays)
    return priceInfo


def make_barra_factor(weights, baseIndexName):
    """
    生成本投资组合及业绩基准指数的每日因子暴露
    返回数据类型为二级字典 {key=factor,value={key=day,value=v}}
    """

    pmFactor = {}
    baseFactor = {}
    for day, w in weights.items():
        # 遍历所有的日期/权重
        # 生成当天的指标-因子暴露数据
        no_cash_weight = make_no_cash_weight(w)  # 将权重图归化为全仓权重
        dF = UTILS.get_exposure(day, no_cash_weight, [baseIndexName])

        pmf = dF['thisPM']  # 本投资组合的因子暴露值 {key=factorname,value=v}
        for f, v in pmf.items():
            if not pmFactor.__contains__(f):
                pmFactor[f] = {}
            pmFactor[f][day] = v

        pmf = dF[baseIndexName]  # 业绩基准指数的因子暴露值{key=factorname,value=v}
        for f, v in pmf.items():
            if not baseFactor.__contains__(f):
                baseFactor[f] = {}
            baseFactor[f][day] = v
    return pmFactor, baseFactor


def cal_factor_value(pmFactor, factorName):
    """
    计算一个投资组合(业绩基准)在所有投资日期上的平均暴露/标准差/极值比例
    :param pmFactor: 全部因子在所有日期上的暴露字典{key=factorname,value={key=day,value=v}}
    :param factorName: 本次需要计算的因子名称
    :return avg,std,expct
    """

    fdict = pmFactor[factorName]
    vlist = list(fdict.values())
    vlist_np = np.array(vlist)
    avg = float(np.mean(vlist_np))  # 平均值
    std = float(np.std(vlist_np))  # 标准差
    exp = 0
    for v in vlist:
        if v > (avg + std) or v < (avg - std):
            exp += 1
    exp = exp / len(vlist)  # 极端值的占比
    return avg, std, exp


def barra(weights, baseIndexName):
    """
    barra分析
    :param weights: 投资组合的每日权重
    :param baseIndexName: 作为业绩比较基准的指数名称
    """

    # baseIndexName 是下面字典中的一个(中文名称)
    # base_index = {'000001.SH': '上证指数', '399001.SZ': '深证综指', '000905.SH': '中证500',
    #               '932000.SH': '中证2000', '899050.BJ': '北证50', '399005.SZ': '中小板指',
    #               '399006.SZ': '创业板指', '930903.CSI': '中证全指', '000300.SH': '沪深300'}
    factor = {'size': '市值', 'beta': 'beta', 'momentum': '动量',
              'non_linear_size': '非线性市值', 'bp': '账面市值比', 'earnings_yield': '盈利能力',
              'growth': '成长', 'leverage': '杠杆', 'liquidity': '流动性', 'volatility': '残差波动率'}

    # 调用工具包中的方法,生成投资组合与业绩基准指标在各个因子上的暴露(每天)
    # 返回数据类型为二级字典 {key=factor,value={key=day,value=v}}
    pmFactor, baseFactor = make_barra_factor(weights, baseIndexName)

    # 对每个因子循环处理
    for fn in list(factor.keys()):
        # 计算因子暴露值的均值/标准差/极值占比(暴露超过±1标准差的天数比例)
        f_avg, f_std, f_expct = cal_factor_value(pmFactor,fn)
        b_avg, b_std, b_expct = cal_factor_value(baseFactor,fn)
        print(f"*****[{factor[fn]}]因子分析*****")
        print(f"科目,{baseIndexName},本投资组合")
        print(f"平均暴露,{b_avg:.4f},{f_avg:.4f}")
        print(f"标准差,{b_std:.4f},{f_std:.4f}")
        print(f"极值占比,{b_expct * 100:.2f}%,{f_expct * 100:.2f}%")
        print()

    # 组合收益 = 无风险收益 + ∑(因子暴露 × 因子收益) + 特异性收益

    # 方法B：精确计算每日因子贡献
    # 对每个交易日t
    # 因子贡献_kₜ = 组合对因子k的暴露_Exposureₖₜ × 因子k的日收益_fₖₜ
    # 特异性收益ₜ = 组合实际收益ₜ - ∑因子贡献_kₜ
    #
    # # 累计收益分解
    # 累计因子贡献_k = ∑(1 + 因子贡献_kₜ)
    # 的连乘积 - 1
    # 累计特异性收益 = ∑(1 + 特异性收益ₜ)
    # 的连乘积 - 1


# from fpdf import FPDF
#
#
# class PDFWithImages(FPDF):
#     def add_image_with_caption(self, image_path, caption=None):
#         """添加图片和可选的标题"""
#         # 检查图片是否存在
#         if not os.path.exists(image_path):
#             print(f"图片不存在: {image_path}")
#             return
#
#         # 获取图片尺寸
#         from PIL import Image
#         with Image.open(image_path) as img:
#             img_width, img_height = img.size
#
#         # 计算缩放比例，使图片宽度不超过页面宽度的80%
#         max_width = self.w * 0.8
#         scale = min(1.0, max_width / img_width)
#
#         # 计算图片位置使其居中
#         x = (self.w - img_width * scale) / 2
#
#         # 如果当前页面空间不足，添加新页面
#         if self.get_y() + img_height * scale > self.h - 30:
#             self.add_page()
#
#         # 添加图片
#         self.image(image_path, x=x, y=self.get_y(),
#                    w=img_width * scale, h=img_height * scale)
#
#         # 更新Y坐标
#         self.set_y(self.get_y() + img_height * scale + 10)
#
#         # 添加标题（如果有）
#         if caption:
#             self.multi_cell(0, 10, caption, align='C')
#             self.ln(10)
#
#
# def create_pdf_with_fpdf(image_paths, text, output_filename="output.pdf"):
#     """
#     使用fpdf将图片和文字组合成PDF
#
#     参数:
#     image_paths: 图片路径列表
#     text: 要添加的文字
#     output_filename: 输出PDF文件名
#     """
#     pdf = PDFWithImages()
#
#     # 添加页面
#     pdf.add_page()
#
#     # 设置字体（支持中文需要中文字体文件）
#     pdf.set_font("Helvetica", size=20)
#
#     # 添加文字
#     pdf.multi_cell(0, 10, text)
#     pdf.ln(10)
#
#     # 添加图片
#     for i, img_path in enumerate(image_paths):
#         # caption = f"picture {i + 1}"  # 可选的图片标题
#         pdf.add_image_with_caption(img_path)
#
#     # 保存PDF
#     pdf.output(output_filename)
#     print(f"PDF已生成: {output_filename}")

def split_datelist(dayslist, period='week'):
    pd = {}
    DL = len(dayslist)
    mini_list = []
    count = 0
    for i in range(DL):
        mini_list.append(dayslist[i])
        if is_period_end_day_in_list(dayslist,i,period):
            pd[count]=mini_list
            mini_list = []
            count +=1
    return pd


def cal_bj_weight(w):
    bj1 = 0.
    for s,v in w.items():
        if ".BJ" in s:
            bj1 +=v
    bj2 = bj1 / (1. - w['CASH'])
    return bj1, bj2


def cal_avg_index(zzindex, daylist):
    x = []
    for d in daylist:
        x.append(zzindex[d])
    return sum(x) / len(x)

def cal_avg_weight(weights,pweek):
    bjw = []
    for d in pweek:
        w = weights[d]
        b1,b2 = cal_bj_weight(w)
        bjw.append(b1)
    return sum(bjw) / len(bjw)

if __name__ == '__main__':
    args = get_args()

    bzindex = CM.load_index("北证50.json")

    # STEP1: 读取指定路径下的所有权重
    print(f"{CM.timestr()}读取权重数据...")
    ts = time()
    weights = UTILS.read_weigh_from_path(args.path, args.match)
    if len(weights) <=0:
        print(f"权重数据太少{len(weights)},不足以进行区间分析,exit")
        exit(0)
    print(f"{CM.timestr()}读取权重数据完成,天数{len(weights)},耗时{time() - ts:.2f}秒")
    # print(weights.keys())
    # 读取指数数据

    dayslist = list(weights.keys())
    daysLen = len(dayslist)

    preBJindex = -1
    nextValue = None
    print("WEEK,BJ50涨幅,BJ所股票权重占比")

    # 把所有的日期按周切分
    wp = split_datelist(dayslist, period='week')
    prevWeekIndex = -1
    for k,pweek in wp.items():
        thisWeekIndex = cal_avg_index(bzindex,pweek)
        if prevWeekIndex <=0:
            indexRate = 0.
        else:
            indexRate = (thisWeekIndex - prevWeekIndex) / prevWeekIndex
        bjw = cal_avg_weight(weights,pweek)
        print(f"{k},{indexRate*100:.2f}%,{bjw*100:.2f}%")
        prevWeekIndex = thisWeekIndex

    # for i in range(daysLen):
    #     # 计算前后两个交易日期之间的相对收益
    #     dB = dayslist[i]  # 今日
    #     wB = weights[dB]  # 今日持仓
    #     bjw1,bjw2 = cal_bj_weight(wB)
    #     bjindex = bzindex[dB]
    #
    #     if preBJindex < 0:
    #         indexRate = 0.
    #     else:
    #         indexRate = (bjindex - preBJindex) / preBJindex
    #     preBJindex = bjindex
    #
    #     R = nextValue
    #     if R is None:
    #         R = 0.
    #     print(f"{dB},{indexRate*100:.2f}%,{bjw1*100:.2f}%") #,{bjw1*100:.2f}%")
    #     nextValue = indexRate

