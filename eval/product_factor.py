import argparse
import json
import os.path

def get_args():
    parser = argparse.ArgumentParser(description='EvalPM')
    parser.add_argument('--path', type=str, default='D:/stock_data/describe.json', help='json文件所在的路径')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    assert os.path.exists(args.path),"指定因子数据文件路径"

    factor_style = {'size': '市值', 'beta': 'Beta', 'momentum': '动量', 'non_linear_size': '非线性市值', 'bp': '估值',
                    'earnings_yield': '盈利', 'growth': '成长', 'leverage': '杠杆', 'liquidity': '流动性',
                    'volatility': '波动'}
    factor_industry = {'ci500000': '交通运输', 'ci630000': '传媒', 'ci370000': '农林牧渔', 'ci350000': '医药',
                       'ci310000': '商贸零售',
                       'ci280000': '国防军工', 'ci220000': '基础化工', 'ci330000': '家电', 'ci240000': '建材',
                       'ci230000': '建筑', 'ci420000': '房地产', 'ci120000': '有色金属',
                       'ci260000': '机械', 'ci300000': '汽车', 'ci320000': '消费者服务', 'ci110000': '煤炭',
                       'ci200000': '电力及公用事业', 'ci270000': '电力设备及新能源',
                       'ci600000': '电子', 'ci100000': '石油石化', 'ci340000': '纺织服装', 'ci700000': '综合',
                       'ci430000': '综合金融', 'ci620000': '计算机',
                       'ci250000': '轻工制造', 'ci610000': '通信', 'ci210000': '钢铁', 'ci400000': '银行',
                       'ci410000': '非银行金融', 'ci360000': '食品饮料'}

    with open(args.path, 'r', encoding='utf-8') as file:
        dict_stock = json.load(file)
    print(dict_stock.keys())
    print("因子类型,名称,均值,标准差,最小值,最大值")
    for name,chname in factor_style.items():
        if not dict_stock.__contains__(name):
            continue
        fd = dict_stock[name]
        print(f"风格因子,{chname},{fd['mean']:.4f},{fd['std']:.4f},{fd['min']:.4f},{fd['max']:.4f}")
    for name,chname in factor_industry.items():
        if not dict_stock.__contains__(name):
            continue
        fd = dict_stock[name]
        print(f"行业因子,{chname},{fd['mean']:.4f},{fd['std']:.4f},{fd['min']:.4f},{fd['max']:.4f}")
    if dict_stock.__contains__('alpha'):
        fd = dict_stock['alpha']
        print(f"收益因子,Alpha,{fd['mean']:.6f},{fd['std']:.6f},{fd['min']:.6f},{fd['max']:.6f}")