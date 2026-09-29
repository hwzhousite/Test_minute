import datetime
import json
import time
import pytz

g_calendar_file = r'/data/raw_generated_data/tushare_data/cfg/calendar.json'
g_calendar = []


def days_between(date1, date2):
    """
    计算两个日期之间的天数
    :param date1: 日期1
    :param date2: 日期2
    :return: 天数
    """
    date1 = datetime.datetime.strptime(date1, '%Y%m%d')
    date2 = datetime.datetime.strptime(date2, '%Y%m%d')
    delta = date2 - date1
    return abs(delta.days) + 1


def date_str(today, fmt, d=0):
    if today is None:
        today = time.strftime(fmt, time.localtime())
    if d != 0:
        dd = datetime.datetime.strptime(today, fmt)
        dd = dd + datetime.timedelta(days=d)
        today = dd.strftime(fmt)
    date_obj = datetime.datetime.strptime(today, fmt)
    return today, date_obj.weekday()


def is_month_end(date):
    """
    判断日期是否是月末
    :param date: 日期字符串
    :return: True/False
    """
    year, month, day, weekday = parse_date(date)
    # 获取下一个日期
    next_day = datetime.datetime(year, month, day) + datetime.timedelta(days=1)
    next_year, next_month, next_day = next_day.year, next_day.month, next_day.day
    if next_month != month:
        return True
    else:
        return False


def make_sub_segment_date(begindate, enddate, subtype='month'):
    """
    生成子区间日期的起止日期
    :param begindate: 起始日期
    :param enddate: 结束日期
    :param subtype: 子区间类型，'month'表示按月分割，'week'表示按周分割
    :return: 子区间日期的起止日期列表
    """

    retlist = []
    alllist = date_list(begindate, enddate, '%Y%m%d')
    if subtype == 'day':
        retlist.append([begindate, enddate])
    elif subtype == 'month':
        m_a = ''
        for date in alllist:
            year, month, day, weekday = parse_date(date)
            if day == 1:
                m_a = date
            elif is_month_end(date):
                m_b = date
                if m_a != '':
                    retlist.append([m_a, m_b])
                m_a = ''
    elif subtype == 'week' or subtype == '2week':
        w_start = None
        DL = len(alllist)
        for i in range(DL):
            date = alllist[i]  # 列表中第i个日期
            # 解析出年/月/日/星期几
            year, month, day, weekday = parse_date(date)
            if 0 <= weekday <= 4:
                # 该日期是一个工作日(简单判定)
                if w_start is None:
                    w_start = date
                w_end = date
                if weekday == 4 or i == DL - 1:
                    retlist.append([w_start, w_end])
                    w_start = None
    # elif subtype == 'week' or subtype == '2week':
    #     w_a = ''
    #     for date in alllist:
    #         year, month, day, weekday = parse_date(date)
    #         if weekday == 0:
    #             w_a = date
    #         elif weekday == 4:
    #             w_b = date
    #             if w_a != '':
    #                 retlist.append([w_a, w_b])
    #             w_a = ''

    if subtype == '2week':
        w2 = len(retlist) // 2
        r = []
        for i in range(w2):
            b = retlist[2 * i][0]
            e = retlist[2 * i + 1][1]
            r.append([b, e])
        retlist = r

    return retlist


def date_list(begin_date, end_date, datefmt):
    # begin_date YYYY-MM-YY
    # end_date YYYY-MM-YY
    retlist = [begin_date]
    if begin_date == end_date:
        return retlist
    stepdays = 1
    if end_date < begin_date:
        stepdays = -1
    bgd = datetime.datetime.strptime(begin_date, datefmt)
    while True:
        edd = bgd + datetime.timedelta(days=stepdays)
        eds = edd.date().strftime(datefmt)
        retlist.append(eds)
        if eds == end_date:
            break
        bgd = edd
    return retlist


def is_valid_date(date_str, fmtstr='%Y%m%d'):
    """
    判断日期字符串是否是有效的日期
    :param date_str:
    :param fmtstr:
    :return:
    """
    try:
        datetime.datetime.strptime(date_str, fmtstr)
        return True
    except ValueError:
        return False


def make_pre_date(oneday, lookbackwindow):
    """
    指定一个输入日期oneday, 计算其前lookbackwindow个交易日的日期
    """

    endday, _ = date_str(oneday, '%Y%m%d', -lookbackwindow * 2)
    datelist = date_list(oneday, endday, '%Y%m%d')
    tradeDays = 0
    for date in datelist:
        if is_valid_trade_date(date):
            tradeDays += 1
            if tradeDays == lookbackwindow:
                return date
    return endday


def parse_date(date):
    """
    解析日期字符串,返回年/月/日/星期几
    :param date: 日期字符串
    :return: 年/月/日/星期几
    """
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    weekday = datetime.datetime(year, month, day).weekday()
    return year, month, day, weekday


def get_next_trade_day(tradeDay, d):
    """
    输入交易日期tradeDay, 计算其后d个交易日的日期
    """

    if d == 0:
        return tradeDay
    nextday = str(tradeDay)
    count = 0
    step = 1 if d > 0 else -1
    while True:
        nextday, _ = date_str(nextday, '%Y%m%d', step)
        if is_valid_trade_date(nextday):
            count += 1
            if count == abs(d):
                return nextday


def week_day(date_str, fmtstr='%Y%m%d'):
    """
    输入日期字符串和格式，返回星期几（0-6，0代表星期一）
    :param date_str:  输入日期str
    :param fmtstr:  日期格式
    :return: int
    """
    date_obj = datetime.datetime.strptime(date_str, fmtstr)
    return date_obj.weekday()


def count_trade_days(begin_date, end_date, datefmt='%Y%m%d'):
    """
    计算两个日期之间的交易日数
    :param begin_date: 开始日期
    :param end_date: 结束日期
    :param datefmt: 日期格式
    :return: 交易日数
    """
    ds = date_list(str(begin_date), str(end_date), datefmt)
    count = 0
    for d in ds:
        if is_valid_trade_date(d):
            count += 1
    return count


def trade_date_list(begin_date, end_date, datefmt='%Y%m%d'):
    # 生成从begin_date到end_date的交易日日期列表
    # begin_date YYYY-MM-YY
    # end_date YYYY-MM-YY

    retlist = []
    if is_valid_trade_date(begin_date):
        retlist = [begin_date]
    if begin_date == end_date:
        return retlist
    stepdays = 1
    if end_date < begin_date:
        stepdays = -1
    bgd = datetime.datetime.strptime(begin_date, datefmt)
    while True:
        edd = bgd + datetime.timedelta(days=stepdays)
        eds = edd.date().strftime(datefmt)
        retlist.append(eds)
        if eds == end_date:
            break
        bgd = edd
    trade_list = []
    for date in retlist:
        if is_valid_trade_date(date):
            trade_list.append(date)
    return trade_list


def is_valid_trade_date(date_str, fmtstr='%Y%m%d'):
    """
    判断日期字符串是否是有效的交易日期
    :param date_str:
    :param fmtstr:
    :return:
    """

    global g_calendar
    # 如果存在日历表,则根据交易日历表来确定
    if g_calendar is not None and len(g_calendar) > 0:
        if date_str in g_calendar:
            return True
        else:
            return False

    if not is_valid_date(date_str, fmtstr):
        return False
    weekday = week_day(date_str, fmtstr)
    if weekday == 5 or weekday == 6:
        return False
    return True


def get_real_next_trade_day(tradeDay, next_day=True):
    d = 1 if next_day else -1
    nextday = str(tradeDay)
    today = timestr(1)
    while True:
        nextday = get_next_trade_day(nextday, d)
        if is_trading_day_by_tushare(nextday) or nextday >= today:
            break
    return nextday


def timestr(d=0):
    # 设置时区为北京时间
    tz = pytz.timezone('Asia/Shanghai')
    # 获取当前时间并转换为北京时间
    nowtime = datetime.datetime.now(tz)

    if d == 0:
        return nowtime.strftime("[%Y-%m-%d %H:%M:%S]")
    elif d == 1:
        return nowtime.strftime("%Y%m%d")
    elif d == 2:
        return nowtime.strftime("%m%d%H%M")
    elif d == 3:
        return nowtime.strftime("%Y%m%d%H%M")
    elif d == 4:
        return nowtime.strftime("%H%M")
    else:
        return nowtime.strftime("[%Y-%m-%d %H:%M:%S]")


def is_trading_day_by_tushare(day):
    # 从本地缓存文件中读取  # 20260129 修改,不再直接访问ts了,只看本地缓存文件
    # 而且放弃原来的单独文件trading_days.json, 统一使用DT.g_calendar_file

    global g_calendar
    global g_calendar_file

    if g_calendar is not None and len(g_calendar) > 0:
        # 日历表已加载到内存了
        return day in g_calendar

    # 否则临时加载它(注意不要去覆盖全局变量)
    with open(g_calendar_file, 'r', encoding='utf-8') as f:
        trading_days = json.load(f)
    return day in trading_days

