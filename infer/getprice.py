# 遍历一个指定的目录,找到目录中所有的权重文件, 然后根据指令去生成一个同名的文件,
# 在文件中记录当天及次一交易日的交易价格
import argparse
import json
import multiprocessing
import os.path
import time
import utils_date as DT
import utils as UTILS
import common_utils as CM

def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--path', type=str, default='', help='json文件路径')
    parser.add_argument('--price', type=str, default='day', help='价格类型[open,5m,day]')
    parser.add_argument('--ps', type=int, default=24, help='进程数')
    parser.add_argument('--DS', type=str, default='day106', help='数据集名称[day104,day106]')
    args = parser.parse_args()
    if args.DS == 'day1062':
        args.g_csv_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/106f_pad_csv_calendar"  # 过去80天数据读取路径
        args.scaler_file = f"{CM.G_ROOT_PATH}/yy_data/tushare_data/ts_260201_106f/scaler_info.txt"  # 数据集对应的字段字典表文件路径
        # 如果使用1062数据集回测,该数据集中的天数据是严格按照交易日历来生成的,没有padd节假日数据了
        # 下面这个全局变量设置了交易日历列表
        # 一旦被设置, 则所有与日期相关的计算(比如获取前一交易日/下一交易日)就依赖这个交易日历来进行了
        with open(DT.g_calendar_file, 'r', encoding='utf-8') as f:
            DT.g_calendar = json.load(f)
    else:
        raise ValueError(f"数据集参数错误{args.DS}")
    UTILS.g_csv_path = args.g_csv_path
    return args

def handel_one_file(path,pmfile,price):
    fwf = os.path.join(path, pmfile)  # 完整文件名
    total_weight = UTILS.read_pmfile(fwf)
    if total_weight is None:
        print(f"从文件{fwf}中读取权重失败")
        return 0
    tradeDay = pmfile[:8]  # 文件名的前8位为日期 YYYYMMDD
    total_weight.pop('CASH', 0)  # 删除现金项
    stock_code = list(total_weight.keys())  # 得到所有股票代码

    price_in = make_price_dict(stock_code, price, tradeDay)
    sellDay = DT.get_real_next_trade_day(tradeDay)
    price_out = make_price_dict(stock_code, price, sellDay)

    price_dict = {
        "in": price_in,
        "out": price_out
    }
    pfn = os.path.join(path, f"{pmfile}.p")
    with open(pfn, 'w', encoding="utf-8") as file:
        json.dump(price_dict, file)
    return 1


def make_price_dict(stock_code_list,price_type,tradeDay):
    price_dict = {}
    for sc in stock_code_list:
        p= UTILS.read_single_price(UTILS.g_csv_path, UTILS.g_field_index, sc, price_type, tradeDay)
        price_dict[sc]=p
    return price_dict


class FileReaderProcessor:
    def __init__(self, path, group_size, price):
        self.path = path
        self.price = price
        wflist= [f for f in os.listdir(path) if f.endswith('.json')]

        fl = len(wflist)
        gs = 1 + fl // group_size
        gs = max(gs, 2)  # 每个进程最少处理多少个文件
        idx = 0
        self.file_list_group = []
        while idx < fl:
            self.file_list_group.append(wflist[idx:idx + gs])
            idx += gs
        self.total = len(wflist)
        print(f"总文件数{self.total},进程数{len(self.file_list_group)}")
        self.data_results = 0

    def process_files(self, file_group, proc_no):
        handeled_count = 0
        success_count = 0
        try:
            for file_name in file_group:
                # 在这里编写对文件的处理逻辑
                handeled_count +=1
                success_count +=handel_one_file(self.path,file_name,self.price)
        except Exception as e:
            print(f"********进程#{proc_no} error: {e}")
        return success_count

    def collect_result(self, result):
        self.data_results +=result

    def run(self):
        pool = multiprocessing.Pool()
        proc_no = 0
        for file_group in self.file_list_group:
            # print(f"进程#{proc_no}处理{len(file_group)}个文件")
            pool.apply_async(self.process_files, args=(file_group, proc_no), callback=self.collect_result)
            proc_no += 1
        pool.close()
        pool.join()

    def get_result(self):
        return self.data_results

if __name__ == '__main__':
    args = get_args()
    if args.path == '' or (not os.path.exists(args.path) or (not os.path.isdir(args.path))):
        raise ValueError(f"指定的路径{args.path}不存在")

    # 先深度单线程,看看速度如何
    # wflist= [f for f in os.listdir(args.path) if f.endswith('.json')]
    # count =0
    # succ =0
    # print(f"文件总数:{len(wflist)}")
    # ts = time.time()
    # for wf in wflist:
    #     succ +=handel_one_file(args.path,wf,args.price)
    #     count +=1
    #     print(f"完成{succ}/{count},耗时:{time.time()-ts:.2f}秒")
    #     if count>=10:
    #         break
    # ts = time.time() - ts
    # print(f"完成{succ}/{count},耗时:{ts:.2f}秒")

    # 多进程模式
    # 读取所有文件
    ts = time.time()
    processer = FileReaderProcessor(args.path, args.ps, args.price)
    processer.run()
    succ = processer.get_result()
    print(f"总文件数{processer.total},成功数{succ},耗时:{time.time()-ts:.2f}秒")
