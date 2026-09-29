"""
一个临时工具文件,用于将指定路径下的csv文件截取最后若干条并保存到文件
如134的pad文件路径 :  /data/raw_generated_data/tushare_data/ts_260413_134f/tgt_pad_data

"""
import argparse
import os
import pandas as pd
import csv
from datetime import date
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Portfolio Parameter')

    parser.add_argument('--cmd', type=str, default='C', help='当前的任务是什么?')
    parser.add_argument('--bigpath', type=str, default='', help='权重数据json文件所在的路径')
    parser.add_argument('--line', type=int, default=12, help='读取数据条数')
    parser.add_argument('--addpath', type=str, default='./padcsv', help='保存路径')

    args = parser.parse_args()
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    return args

def read_and_save_last_lines(srcFile,tgtFile,lines):
    """
    从源文件读取最后若干行,保存到目标文件
    """

    # 读取全部数据进入内存
    df = pd.read_csv(srcFile, header=None)

    if df.shape[0] < lines:
        last_lines = df.shape[0]
    else:
        last_lines = lines

    # 取最后last_lines条数据
    last_data = df.tail(last_lines)

    last_data.to_csv(tgtFile,index=False,header=False)


def task_Copy_last(args):
    # 列出指定路径下的所有csv文件
    Total = 0
    for fn in os.listdir(args.bigpath):
        if not fn.endswith('.csv'):
            continue
        fullFN = os.path.join(args.bigpath,fn)
        addFN = os.path.join(args.addpath,fn)
        read_and_save_last_lines(fullFN,addFN,args.line)
        Total +=1
        if Total % 100 ==0:
            print(f"#{Total}---{fn}")
    print(f"#{Total}---DONE")

def task_Paste_last(args):
    Total = 0
    for fn in os.listdir(args.bigpath):
        if not fn.endswith('.csv'):
            continue
        fullFN = os.path.join(args.bigpath, fn)
        addFN = os.path.join(args.addpath, fn)
        read_and_save_last_lines(fullFN, addFN, args.line)
        Total += 1
        if Total % 100 == 0:
            print(f"#{Total}---{fn}")
    print(f"#{Total}---DONE")




def get_last_data_date(csv_path: Path):
    """
    获取 CSV 文件最后一行的第5/6/7列组成的日期。
    如果文件为空，返回 None。
    """
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        last_row = None
        for row in reader:
            last_row = row   # 直接迭代，不跳过任何行
        if last_row is None:
            return None
        # 第5/6/7列对应索引4,5,6（0-based）
        year = int(last_row[4])
        month = int(last_row[5])
        day = int(last_row[6])
        return date(year, month, day)

def filter_rows_by_date(a_path: Path, ref_date: date):
    """
    读取 a.csv 的所有行，返回日期 > ref_date 的行。
    """
    selected_rows = []
    with open(a_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue   # 列数不足，跳过（防御）
            year = int(row[4])
            month = int(row[5])
            day = int(row[6])
            row_date = date(year, month, day)
            if row_date > ref_date:
                selected_rows.append(row)
    return selected_rows

def read_all_rows(a_path: Path):
    """读取 a.csv 的所有行"""
    rows = []
    with open(a_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows

def append_to_b(b_path: Path, rows):
    """将行列表追加到 b.csv 末尾"""
    if not rows:
        return
    with open(b_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def append_a_to_b(a_file, b_file):
    a_path = Path(a_file)
    b_path = Path(b_file)

    # 1. 获取 b 文件最后一条数据的日期
    ref_date = get_last_data_date(b_path)

    # 2. 根据是否有参考日期决定从 a 中选取哪些行
    if ref_date is None:
        # b 文件为空 -> 将 a 全部追加
        print("b.csv 为空，将把 a.csv 的全部数据追加到 b.csv")
        rows_to_append = read_all_rows(a_path)
    else:
        # b 非空 -> 只追加日期大于 ref_date 的行
        rows_to_append = filter_rows_by_date(a_path, ref_date)

    # 3. 追加到 b
    if rows_to_append:
        append_to_b(b_path, rows_to_append)
        print(f"成功追加 {len(rows_to_append)} 行到 {b_file}")
    else:
        print("没有需要追加的数据。")


if __name__ == '__main__':
    args = get_args()
    if args.cmd.upper() == 'C':
        # C=Copy 这个指令在服务器上运行,用于将bigpath下所有文件最近的若干条数据(line参数指定)保存到一个单独的目录addpath下,
        # 这样的话,只有增量数据,文件大小很小,方便于下载到本地
        task_Copy_last(args)
    elif args.cmd.upper() == 'P':
        # P=Paste 这个指令在本地机器上运行,用于将指定目录
        # 目标是读取addpath下的所有数据文件,与bigpath下的对应文件做一一对照,如果有全新的文件,需要报警
        # 否则,判断新的增量文件中的数据,是否已经在原来的数据文件中, 只将更新的增量数据写入数据
        task_Paste_last(args)



