# 从的文件中读取权重参数,计算该笔投资交易的收益
import argparse
import json
import os
import utils_date as DT
import utils as UTILS
import common_utils as CM

def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--DS', type=str, default='day106', help='')
    parser.add_argument('--pmfile', type=str, default='', help='')
    parser.add_argument('--targetPrice', type=str, default='', help='交易价格默认为全天均价')
    parser.add_argument('--fuquan', type=int, default=0, help='是否要对价格进行复权操作')

    args = parser.parse_args()
    if args.DS == 'day1062':
        args.g_csv_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/106f_pad_csv_calendar"  # 过去80天数据读取路径
        args.scaler_file = f"{CM.G_ROOT_PATH}/yy_data/tushare_data/ts_260201_106f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 如果使用1062数据集回测,该数据集中的天数据是严格按照交易日历来生成的,没有padd节假日数据了
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
    else:  # day111
        args.g_csv_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/111f_index_data/pad_data"
        args.scaler_file = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/111f_index_data/scaler_info.txt"
    UTILS.G_ALL_STOCK_DATA.reset(args.DS,args.g_csv_path)
    return args


def read_price(tradeDay, price_type, total_weight, fuquan=False):
    today = DT.timestr(1)
    if today == tradeDay:
        # 回测方法不支持读取当天价格,因为当天数据还未生成
        return 0.

    # 如果不是当天,则需要读取历史数据
    # 但是历史数据中没有real
    assert price_type != 'real', "Invalid price type--real not available"

    # ts = time.time()
    # price = {}
    # for code, weight in total_weight.items():
    #     if code == 'CASH':
    #         continue
    #     price[code] = UTILS.read_single_price(UTILS.g_csv_path, UTILS.g_field_index,code, price_type, tradeDay)
    code_list = list(total_weight.keys())
    if 'CASH' in code_list:
        code_list.remove('CASH')
    if len(code_list)>0:
        price = UTILS.read_group_price(code_list, price_type, tradeDay, fuquan=fuquan)
        # print(f"读取价格{len(price)}用时:{time.time() - ts:.2f}")
        return price
    else:
        return {}

def key_up(d):
    new_d = {}
    for k,v in d.items():
        new_d[str.upper(k)] = v
    return new_d


def cal_value_by_pmfile(pmfile, output=True, price_target='', fuquan=False):
    total_weight = UTILS.read_pmfile(pmfile)
    if total_weight is None:
        print(f"从文件{pmfile}中读取投资参数失败")
        return None, None, None, None
    total_weight=key_up(total_weight)

    # 文件名的命名规范为: YYYYMMDD_pmmodel.json
    # 其中YYYYMMDD为投资日期,pmmodel为投资模型名称
    # 读取投资日期
    pmfile = os.path.basename(pmfile)
    pmfile = os.path.splitext(pmfile)[0]
    tradeDay = pmfile[:8]
    if not DT.is_valid_date(tradeDay):
        print("Invalid trade day: ", tradeDay)
        return None, None, None, None
    # 读取投资模型名称
    if price_target=='':
        pmmodel = pmfile[8:]
        buy_price_type = 'open'
        sell_price_type = 'open'
        if '_5m_' in pmmodel:
            buy_price_type = '5m'
            sell_price_type = '5m'
        elif '_60m_' in pmmodel:
            buy_price_type = '60m'
            sell_price_type = '60m'
        elif '_day_' in pmmodel or '_Day_' in pmmodel:
            buy_price_type = 'day'
            sell_price_type = 'day'
    else:
        buy_price_type = price_target
        sell_price_type = price_target

    # 获取此时此刻日期
    today = DT.timestr(1)

    # 获取前一交易日
    prevDay=None
    if output:
        prevDay = DT.get_real_next_trade_day(tradeDay,next_day=False)

    # 获取下一个交易日
    sellDay = DT.get_real_next_trade_day(tradeDay,next_day=True)

    if today == tradeDay:
        # 如果是今天刚刚发生的交易, 则文件中还没有当天的数据, 只能假定买入价为open, 卖出价为当时的最新价
        buy_price_type = 'open'
        sellDay = today
        sell_price_type = 'real'
    else:
        # 如果不是今天刚刚发生的交易, 则买入数据已经存在于数据文件中了,可以直接读取应该的买入标的价格
        # buy_price_type 不用修改
        # 但是卖出的计价数据还需要根据时间来确定
        if sellDay < today:
            # 如果卖出日期已经过了, 则卖出日数据也已经存在了,可直接读取
            # sell_price_type 不用修改
            pass
        else:
            # 卖出日为今天, 数据文件还不存在, 所以需要假定卖出价为当时的最新价
            sellDay = today
            sell_price_type = 'real'

    # 从数据文件中读取买入价格
    prevClose = None
    todayOpen = None
    if output:
        prevClose = read_price(prevDay, 'close', total_weight)
        todayOpen = read_price(tradeDay, 'open', total_weight)
    buy_price = read_price(tradeDay, buy_price_type, total_weight, fuquan=fuquan)

    # 临时测试
    # sellDay = tradeDay  # 假定当天买入且当天卖出
    # sell_price_type = 'close'  # 假定以当天收盘价卖出

    sell_price = read_price(sellDay, sell_price_type, total_weight, fuquan=fuquan)

    # 计算收益
    if output:
        print(f"买入日期:{tradeDay}-买入价格:{buy_price_type}-卖出日期:{sellDay}-卖出价格:{sell_price_type}")
        print(f"股票代码,权重,前收,今开,买入价,卖出价,收益率")
    value = 0.
    total_stock_count = len(total_weight) - 1
    total_win_count = 0
    cash_weight = total_weight['CASH']
    for code, weight in total_weight.items():
        if code == 'CASH':
            if output:
                print(f"{code},{weight:.4f},1,1,1,1,0")
            value += weight
            continue
        if code in sell_price and code in buy_price and buy_price[code]>1e-6:
            sr = sell_price[code] / buy_price[code]
        else:
            sr = 1.
        sv = sr * weight
        value += sv
        if sr > 1:
            total_win_count += 1
        if output:
            print(f"{code},{weight:.4f},{prevClose.get(code,-1):.2f},{todayOpen.get(code,-1):.2f},{buy_price.get(code,-1):.2f},{sell_price.get(code,-1):.2f},{(sr - 1) * 100:.2f}%")
    if output:
        print(f",买入日期,总股票数,盈利股票数,胜率")
        print(f"BUY,{tradeDay},{total_stock_count},{total_win_count},{total_win_count / total_stock_count * 100:.2f}%")
        print(f",卖出日期,总价值,总收益率,投资收益率")
        print(f"SELL,{sellDay},{value:.4f},{(value - 1) * 100:.2f}%,{(value - 1) * 100 / (1 - cash_weight):.2f}%")
    return value,total_stock_count,total_win_count,total_weight


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.pmfile):
        print("File not found: ", args.pmfile)
        exit(1)
    value,_,_,_ = cal_value_by_pmfile(args.pmfile,price_target=args.targetPrice, fuquan=args.fuquan==1)
    print("=====DONE====")
