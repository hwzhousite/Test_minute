import os
import random
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，适用于无GUI环境
import matplotlib.pyplot as plt
from datetime import datetime


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
        if indices[-1] != n - 1:
            indices.append(n - 1)
        # 设置刻度位置和标签
        ax.set_xticks([x_dates[i] for i in indices])
        ax.set_xticklabels([date_strings[i] for i in indices], rotation=45, ha='right')


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

    # plt.figure(figsize=(32, 16))  # 设置图片大小，根据需要调整

    fig, (ax1) = plt.subplots(1, 1, figsize=(32, 16))

    for dn, dv in data.items():
        p_mk = MK[mi]
        mi = (mi + 1) % len(MK)
        p_line = LS[ml]
        ml = (ml + 1) % len(LS)
        p_color = CO[mc]
        mc = (mc + 1) % len(CO)
        ax1.plot(x, dv, label=dn, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)  # 绘制数据b

    # 3. 设置横坐标日期格式
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))  # 设置日期格式
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 设置刻度间隔，避免过于密集
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标签以避免重叠

    set_xticks_dynamic(ax1, x, tdays, max_ticks=40)

    # 4. 添加标题和标签
    ax1.set_title(f'{title[0]}-{args.mark}', fontsize=48)
    ax1.set_xlabel(title[1], fontsize=24)
    ax1.set_ylabel(title[2], fontsize=24)
    ax1.legend(fontsize=24)  # 显示图例

    # 5. 保存图片到文件
    imgname = f'{args.mark}_{title[0]}_{tdays[0]}_{tdays[-1]}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成'{title[0]}'图表<{image_filename}>成功...")
    return imgname


def draw_line_picture(args, flag, dataInfo, marketInfor=None):
    """
    根据给定的信息绘制折线图(合并或者拆分),但只输出一张画布(即一个png文件)
    dataInfo={
            'title':title[],
            'days':all_Daylist[],
            'value':all_ValList[],
            'split':True,
        }
    """

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    MK = ['.', ',', 'o', 'v', '^', '<', '>']
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    LS = ['-', '--', '-.', ':']
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    # 2. 根据要求创建画布
    if dataInfo['split']:
        # 若干条数据按顺序绘制到同一张画布上
        N = len(dataInfo['title'])  # 子图个数
        fig, ax = plt.subplots(N, 1, figsize=(32, 16 * N))
        # 按顺序绘制所有曲线
        for i in range(N):
            title = dataInfo['title'][i]
            dayList = dataInfo['days'][i]
            valList = dataInfo['value'][i]
            p_mk = MK[mi]
            mi = (mi + 1) % len(MK)
            p_line = LS[ml]
            ml = (ml + 1) % len(LS)
            p_color = CO[mc]
            mc = (mc + 1) % len(CO)
            x = [datetime.strptime(date_str, '%Y%m%d') for date_str in dayList]
            ax[i].plot(x, valList, label=title, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)
            if marketInfor is not None and marketInfor['pos'] == i:
                p_mk = MK[mi]
                mi = (mi + 1) % len(MK)
                p_line = LS[ml]
                ml = (ml + 1) % len(LS)
                p_color = CO[mc]
                mc = (mc + 1) % len(CO)

                # 第二条线（第二个y轴）
                ax2 = ax[i].twinx()
                # ax2.set_ylim(0.5, 2.)  # 设置第二个y轴范围
                ax2.set_ylabel(marketInfor['name'], color='r', fontsize=24)
                ax2.plot(x, marketInfor['value'], label=marketInfor['name'], marker=p_mk, linestyle=p_line,
                         color=p_color, linewidth=1,
                         markersize=3)
                ax2.legend(loc='upper right', fontsize=24)
            # 设置网格线 - 只显示水平网格线
            # y_min, y_max = ax[i].get_ylim()
            # 设置y轴刻度，每隔0.5一个刻度
            # ax[i].set_yticks(np.arange(round(y_min), round(y_max) + 1, 0.5))
            ax[i].yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            ax[i].xaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            set_xticks_dynamic(ax[i], x, dayList, max_ticks=45)
            ax[i].set_title(f'{title}-{args.mark}', fontsize=48)
            ax[i].set_ylabel(title, fontsize=24)
            ax[i].legend(loc='upper left', fontsize=24)  # 显示图例
            ax[i].tick_params(axis='x', labelsize=20)
            ax[i].tick_params(axis='y', labelsize=20)
    else:
        # 所有数据被绘制到同一个子图里
        title = dataInfo['title']
        N = len(title)  # 数据个数
        fig, ax = plt.subplots(1, 1, figsize=(32, 16))
        # 按顺序绘制所有曲线
        dayList = dataInfo['days'][0]
        x = [datetime.strptime(date_str, '%Y%m%d') for date_str in dayList]
        for i in range(N):
            valList = dataInfo['value'][i]
            p_mk = MK[mi]
            mi = (mi + 1) % len(MK)
            p_line = LS[ml]
            ml = (ml + 1) % len(LS)
            p_color = CO[mc]
            mc = (mc + 1) % len(CO)
            ax.plot(x, valList, label=title[i], marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)
        # 设置网格线 - 只显示水平网格线
        # y_min, y_max = ax.get_ylim()
        # 设置y轴刻度，每隔0.5一个刻度
        # ax.set_yticks(np.arange(round(y_min), round(y_max) + 1, 0.5))
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
        set_xticks_dynamic(ax, x, dayList, max_ticks=45)
        ax.set_title(f'{flag}-{args.mark}', fontsize=48)
        ax.set_ylabel(flag, fontsize=24)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(loc='upper left', fontsize=24)  # 显示图例
    fig.subplots_adjust(hspace=0.4)  # 调整纵向间距
    fig.tight_layout(pad=2)  # 添加紧凑布局，调整子图间距

    # 5. 保存图片到文件
    imgname = f'{args.mark}_{flag}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成图表<{image_filename}>成功...")
    return imgname


def draw_line_picture_f(args, flag, dataInfo, G=1):
    """
    根据给定的信息绘制折线图(合并或者拆分),但只输出一张画布(即一个png文件)
    dataInfo={
            'title':title[],  长度为N
            'days':all_Daylist[], 长度为N
            'value':all_ValList[], 长度为N
        }
    G=1 表示每个序列单独绘制一个图
    如果G>1 则表示每连续的G个数据绘制到同一张图片,此时要求N是G的整数倍
    """

    N = len(dataInfo['title'])
    if G > 1:
        assert G == 2, f"暂时只支持2个数据项一起绘制"
        assert N % G == 0, f"分组绘图时,数据项N={N}必须是G={G}的整数倍"

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    MK = ['.', ',', 'o', 'v', '^', '<', '>']
    random.shuffle(MK)
    mi = random.randint(0, len(MK) - 1)
    LS = ['-', '--', '-.', ':']
    random.shuffle(LS)
    ml = random.randint(0, len(LS) - 1)
    CO = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    random.shuffle(CO)
    mc = random.randint(0, len(CO) - 1)

    # 总的子图个数, 应该是 N // G
    PN = N // G
    assert PN > 1 , f"本方法只适用于子图数大于1的情况"
    fig, ax = plt.subplots(PN, 1, figsize=(32, 16 * PN))
    pi = 0
    for i in range(PN):
        # 绘制PN个子图,每个子图中有G个数据项(G=2)
        if G == 1:
            title = dataInfo['title'][pi]
            dayList = dataInfo['days'][pi]
            valList = dataInfo['value'][pi]
            p_mk = MK[mi]
            mi = (mi + 1) % len(MK)
            p_line = LS[ml]
            ml = (ml + 1) % len(LS)
            p_color = CO[mc]
            mc = (mc + 1) % len(CO)
            x = [datetime.strptime(date_str, '%Y%m%d') for date_str in dayList]
            ax[i].plot(x, valList, label=title, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)
            # 设置网格线 - 只显示水平网格线
            # y_min, y_max = ax[i].get_ylim()
            # 设置y轴刻度，每隔0.5一个刻度
            # ax[i].set_yticks(np.arange(round(y_min), round(y_max) + 1, 0.5))
            ax[pi].yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            ax[pi].xaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            set_xticks_dynamic(ax[pi], x, dayList, max_ticks=40)
            ax[pi].set_title(f'{title}-{args.mark}', fontsize=48)
            ax[pi].set_ylabel(title, fontsize=24)
            ax[pi].legend(loc='upper left', fontsize=24)  # 显示图例
            pi +=1
        else:
            # 此时 G=2, 表示两两为一组
            gtitle = f"{dataInfo['title'][2 * i]}-{dataInfo['title'][2 * i+1]}"
            title = dataInfo['title'][2 * i]
            dayList = dataInfo['days'][2 * i]
            valList = dataInfo['value'][2 * i]
            p_mk = MK[mi]
            mi = (mi + 1) % len(MK)
            p_line = LS[ml]
            ml = (ml + 1) % len(LS)
            p_color = CO[mc]
            mc = (mc + 1) % len(CO)
            x = [datetime.strptime(date_str, '%Y%m%d') for date_str in dayList]
            ax[pi].plot(x, valList, label=title, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1,
                           markersize=3)

            p_mk = MK[mi]
            mi = (mi + 1) % len(MK)
            p_line = LS[ml]
            ml = (ml + 1) % len(LS)
            p_color = CO[mc]
            mc = (mc + 1) % len(CO)

            # 第二条线
            title = dataInfo['title'][2 * i + 1]
            dayList = dataInfo['days'][2 * i + 1]
            valList = dataInfo['value'][2 * i + 1]
            # ax2 = ax[pi].twinx()
            # ax2.plot(x, valList, label=gtitle, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)
            # ax2.legend(loc='upper right', fontsize=24)

            ax[pi].plot(x, valList, label=title, marker=p_mk, linestyle=p_line, color=p_color, linewidth=1, markersize=3)
            # ax[pi].legend(loc='upper right', fontsize=24)

            # 设置网格线 - 只显示水平网格线
            # y_min, y_max = ax[i].get_ylim()
            # 设置y轴刻度，每隔0.5一个刻度
            # ax[i].set_yticks(np.arange(round(y_min), round(y_max) + 1, 0.5))
            ax[pi].yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            ax[pi].xaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
            set_xticks_dynamic(ax[pi], x, dayList, max_ticks=40)
            ax[pi].set_title(f'{gtitle}', fontsize=48)
            ax[pi].set_ylabel(gtitle, fontsize=24)
            ax[pi].legend(loc='upper left', fontsize=24)  # 显示图例

            ax[pi].tick_params(axis='x', labelsize=20)
            ax[pi].tick_params(axis='y', labelsize=20)

            pi +=1

    fig.subplots_adjust(hspace=0.4)  # 调整纵向间距
    fig.tight_layout(pad=2)  # 添加紧凑布局，调整子图间距

    # 5. 保存图片到文件
    imgname = f'{args.mark}_{flag}.png'
    image_filename = os.path.join(args.out_path, imgname)
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')  # 保存为PNG文件，设置分辨率和紧凑布局
    plt.close()  # 关闭图形以释放内存
    print(f"生成图表<{image_filename}>成功...")
    return imgname
