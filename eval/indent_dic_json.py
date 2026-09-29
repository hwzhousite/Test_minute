"""
将一个json文件格式重整理,增加缩进,方面人肉眼观察
"""
import argparse
import json
import os
import utils as UT


def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    # 下面这两个参数指明输入的每日权重数据所在的位置
    parser.add_argument('--file', type=str, default='', help='file name')
    parser.add_argument('--indent', type=int, default=1, help='indent')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    fileName = args.file
    if not os.path.exists(fileName):
        raise ValueError(f"{fileName}文件不存在")

    dic, enc = UT.load_json_dic(fileName)
    with open(fileName, 'w', encoding=enc) as file:
        json.dump(dic, file, indent=args.indent)
    print(f"DONE,enc={enc}")
