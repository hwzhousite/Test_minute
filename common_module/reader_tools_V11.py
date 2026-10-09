"""
数据读取接口 V11

在 V5_calendar 基础上:
  - 分钟数据输出 [B, M, L, 48, C2]（按交易日 × 48 根 5 分钟）
  - 日线数据输出 [B, M, L, C1]
  - 日线决定样本与股票池，分钟数据按交易日历补充，缺失位置填零
  - 注册 xt_260527_14f 分钟数据集（14 列）
  - 双数据集 reader 支持 jiaoji_mode(stop_limit_mode 等) 与 zero_check_cols 过滤（与日线单数据集 reader 一致）
  - 支持 args.minute_offset_days: 分钟序列相对日线序列首日的偏移天数(按交易日历),
    用于只读取日线回看窗口末尾的若干天分钟数据
"""

from __future__ import annotations

import copy
import json
import math
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import common_module.reader_tools_V5_calendar as v5
from common_module.reader_tools_V5_calendar import (
    calendar,
    get_n_step_work_date,
    make_trade_date_list,
    read_one_stock_seq,
    timestr,
)

G_BASE_DATA = '/data'
MINUTE_BARS_PER_DAY = 48

# 复用 V5 工具函数与单数据集 reader 主体
DatasetConfig = v5.DatasetConfig
read_record_from_binary = v5.read_record_from_binary
read_record_from_binary_step = v5.read_record_from_binary_step


def reshape_minute_tensor(tensor: torch.Tensor, minute_before_num: int, minute_after_num: int,
                          m5_first_only: bool = False) -> torch.Tensor:
    """
    [M, L_day*48, C] -> [M, L_day, 48, C]
    m5_first_only 时 [M, L_day, C] -> [M, L_day, 1, C]
    """
    m, total_len, c = tensor.shape
    l_day = minute_before_num + minute_after_num
    if m5_first_only:
        if total_len != l_day:
            raise ValueError(f'm5_first_only 时期望长度 {l_day}, 实际 {total_len}')
        return tensor.view(m, l_day, 1, c)
    expected = l_day * MINUTE_BARS_PER_DAY
    if total_len != expected:
        raise ValueError(f'分钟序列长度应为 {expected} (= {l_day}*48), 实际 {total_len}')
    return tensor.view(m, l_day, MINUTE_BARS_PER_DAY, c)


# 注册 14f 分钟数据集
_orig_dataset_config_init = DatasetConfig.__init__


def _dataset_config_init_v11(self):
    _orig_dataset_config_init(self)
    self.config['xt_260527_14f'] = {
        'data_path_bin': f'{G_BASE_DATA}/yy_data/five_minute_data/xt_260527_14f/bin_data',
        'index_file': f'{G_BASE_DATA}/yy_data/five_minute_data/xt_260527_14f/index.json',
        'scaler_file': f'{G_BASE_DATA}/yy_data/five_minute_data/xt_260527_14f/scaler_info.txt',
        'channels': 14,
        'period': 'five_minute',
        'accuracy': 'd',
        'raw_data': 'Y',
    }


DatasetConfig.__init__ = _dataset_config_init_v11


class DailyWithMinuteDataset(torch.utils.data.Dataset):
    """日线样本决定股票池和日期；分钟覆盖不足的位置填零。"""

    def __init__(self, daily_dataset, reader):
        self.daily_dataset = daily_dataset
        self.reader = reader

    def __len__(self):
        return len(self.daily_dataset)

    def __getitem__(self, idx):
        day, group = self.daily_dataset.ds[idx][:2]
        stocks = self.daily_dataset.get_si_stock_list(day, group)
        daily = self.daily_dataset[idx]
        return daily, self.reader.read_minutes(stocks, day)


class DailyWithMinuteReader(v5.StockDataReaderON):
    """严格复用原版日线样本生成，不以分钟索引筛选日线样本。"""

    def __init__(self, args):
        self.ms_data_config = DatasetConfig().get_config(args.ms_data_name)
        with open(self.ms_data_config['index_file'], encoding='utf-8') as f:
            self.ms_index = v5.normalize_stock_index(json.load(f))
        self.ms_calendar = [int(d) for d in calendar]
        self.ms_calendar_index = {d: i for i, d in enumerate(self.ms_calendar)}
        super().__init__(args)

    def read_minutes(self, stocks, day):
        days = self.args.minute_before_num + self.args.minute_after_num
        channels = self.ms_data_config['channels']
        offset = getattr(self.args, 'minute_offset_days',
                         self.args.day_before_num - self.args.minute_before_num)
        if offset is None:
            offset = self.args.day_before_num - self.args.minute_before_num
        first = self.ms_calendar_index[int(day)] + offset
        if first < 0 or first + days > len(self.ms_calendar):
            raise ValueError('分钟窗口超出交易日历')
        if channels == 14 and self.ms_data_config.get('raw_data') == 'Y':
            # 原始14列文件可能缺日/缺bar，不能按index.json推算物理行号。
            dates = self.ms_calendar[first:first + days]
            first_only = getattr(self.args, 'm5_first_only', False)
            bars = 1 if first_only else MINUTE_BARS_PER_DAY
            rows = [v5.read_minute_calendar_seq(
                self.ms_data_config['data_path_bin'], stock, dates,
                accuracy=self.ms_data_config['accuracy'],
                m5_first_only=first_only,
                missing_policy=getattr(self.args, 'minute_missing_policy', 'warn'),
            ).float().reshape(days, bars, channels) for stock in stocks]
            return torch.stack(rows) if rows else torch.empty((0, days, bars, channels))
        out = np.zeros((len(stocks), days, 48, channels), dtype=np.float32)
        dtype = np.float64 if self.ms_data_config['accuracy'] == 'd' else np.float32
        for j, stock in enumerate(stocks):
            code = str(stock).split('.')[0].zfill(6)
            info = self.ms_index.get(code)
            if info is None:
                continue
            start, end = map(int, info[:2])
            source_start = self.ms_calendar_index[start]
            lo = max(first, source_start)
            hi = min(first + days, self.ms_calendar_index[end] + 1)
            if lo >= hi:
                continue
            path = os.path.join(self.ms_data_config['data_path_bin'], code + '.bin')
            if not os.path.isfile(path):
                continue
            count = (hi - lo) * 48 * channels
            data = np.fromfile(path, dtype=dtype, count=count,
                               offset=(lo - source_start) * 48 * channels * np.dtype(dtype).itemsize)
            # 短文件保留完整bar，其余填零；不改变日线样本。
            bars = data.size // channels
            dst = out[j].reshape(-1, channels)
            begin = (lo - first) * 48
            dst[begin:begin + bars] = data[:bars * channels].reshape(bars, channels)
        if getattr(self.args, 'm5_first_only', False):
            out = out[:, :, :1, :]
        return torch.from_numpy(out)

    def create_data_loader(self, start_time, end_time, refresh=False, stride_shift=False):
        loader = super().create_data_loader(start_time, end_time, refresh, stride_shift)
        return DataLoader(DailyWithMinuteDataset(loader.dataset, self),
                          batch_size=loader.batch_size, shuffle=self.args.shuffle,
                          num_workers=loader.num_workers, drop_last=loader.drop_last,
                          prefetch_factor=loader.prefetch_factor if loader.num_workers else None)


class DiskBinDatasetTW_V11(v5.DiskBinDatasetTW):
    def read_si_seq(self, stock_list, start_date_key, day_time_step, minute_time_step):
        """
        读取一组股票的日线序列 [S, L_day, C1] 与分钟序列 [S, L_min, 48, C2]
        分钟序列首日 = 日线序列首日 + minute_offset_days(交易日)
        """
        # 分钟序列相对日线首日的偏移; 未指定时沿用V5的定义(day_before_num - minute_before_num)
        offset = getattr(self.args, 'minute_offset_days', None)
        if offset is None:
            offset = self.args.day_before_num - self.args.minute_before_num
        ei = self.all_dates_index[start_date_key]  # 日线序列首日在全体日期列表中的索引
        ms_ei = ei + offset  # 分钟序列首日在全体日期列表中的索引(按交易日历, 不受节假日影响)

        seq = []
        ms_seq = []
        for stock_code in stock_list:
            si = self.all_dates_index[self.dict_stock[stock_code][0]]  # 日线文件首日索引
            stock_data = read_one_stock_seq(self.args.data_path_bin, stock_code, ei - si,
                                            self.args.channels, day_time_step, False,
                                            self.args.accuracy)  # [L, C1]

            config = self.DSD.ms_data_config
            if config['channels'] == 14 and config.get('raw_data') == 'Y':
                days = self.args.minute_before_num + self.args.minute_after_num
                dates = [int(d) for d in calendar]
                first = dates.index(int(start_date_key)) + offset
                if first < 0 or first + days > len(dates):
                    raise ValueError('分钟窗口超出交易日历')
                ms_stock_data = v5.read_minute_calendar_seq(
                    config['data_path_bin'], stock_code, dates[first:first + days],
                    accuracy=config['accuracy'], m5_first_only=self.args.m5_first_only,
                    missing_policy=getattr(self.args, 'minute_missing_policy', 'warn'))
            else:
                ms_si = self.all_dates_index[self.ms_dict_stock[stock_code][0]]  # 分钟文件首日索引
                ms_start_idx = ms_ei - ms_si
                assert ms_start_idx >= 0, f"分钟数据集起始索引必须>=0, {stock_code} ms_si-{ms_si} ms_ei-{ms_ei}"
                ms_stock_data = read_one_stock_seq(self.DSD.ms_data_config['data_path_bin'], stock_code,
                                                   ms_start_idx * MINUTE_BARS_PER_DAY,
                                                   self.DSD.ms_data_config['channels'],
                                                   minute_time_step, self.args.m5_first_only,
                                                   self.DSD.ms_data_config['accuracy'])  # [L*48, C2]

            seq.append(stock_data)
            ms_seq.append(ms_stock_data.float())  # 分钟数据量大, 以float32返回以节省内存
        seq = torch.stack(seq, dim=0)  # [S, L, C1]
        ms_seq = torch.stack(ms_seq, dim=0)  # [S, L*48, C2]
        ms_seq = reshape_minute_tensor(
            ms_seq,
            self.args.minute_before_num,
            self.args.minute_after_num,
            self.args.m5_first_only,
        )
        return seq, ms_seq


class StockDataReaderTW_V11(v5.StockDataReaderTW):
    """双数据集 reader，分钟输出 [M, L, 48, C]。"""

    # 样本股票筛选逻辑(jiaoji_mode / stop_limit / zero_check_cols)与日线单数据集reader完全一致
    build_zero_days_dict = v5.StockDataReaderON.build_zero_days_dict
    _filter_stocks_by_zero_days = v5.StockDataReaderON._filter_stocks_by_zero_days
    prepare_data_dict = v5.StockDataReaderON.prepare_data_dict

    def init(self):
        print(f"{timestr()}初始化数据加载器。。。")
        time_start = __import__('time').time()
        self.zero_days_dict = {}
        self.chengfen_data = {}
        if getattr(self.args, 'chengfen', ''):
            with open(self.args.chengfen, 'r', encoding='utf-8') as f:
                self.chengfen_data = {int(k): set(v) for k, v in json.load(f).items()}
        self.load_index_from_file()
        print(f"股票数量：{len(self.dict_stock)}")
        print(f"日期数量：{len(self.dict_date)}, 总数据量：{sum(len(v) for v in self.dict_date.values())}")

        # 按日线数据的zero_check_cols扫描全0数据(与日线reader一致)
        self.build_zero_days_dict()

        print(f"{timestr()}创建数据索引一级字典。。。")
        self.data_set_dict = self.prepare_data_dict()
        print(f"{timestr()}重新洗牌数据索引一级字典。。。")
        self.re_shuffle_data_dict()
        print(f"{timestr()}数据加载器初始化完成，耗时{__import__('time').time() - time_start:.2f}秒")

    def load_index_from_file(self):
        print(f"{timestr()}从文件{self.args.index_file}加载股票索引文件。。。")
        with open(self.args.index_file, 'r', encoding='utf-8') as file:
            dict_stock = json.load(file)
        with open(self.ms_data_config.get('index_file'), 'r', encoding='utf-8') as file:
            ms_dict_stock = json.load(file)

        dict_stock = {k: v for k, v in dict_stock.items() if k in ms_dict_stock}
        ms_dict_stock = {k: v for k, v in ms_dict_stock.items() if k in dict_stock}

        self.dict_stock = {str(k): v for k, v in dict_stock.items()}
        self.ms_dict_stock = {str(k): v for k, v in ms_dict_stock.items()}

        min_date = 99999999
        max_date = 0
        for stock_code, stock_index in self.dict_stock.items():
            ms_stock_index = self.ms_dict_stock[stock_code]
            min_date = min(min_date, stock_index[0], ms_stock_index[0])
            max_date = max(max_date, stock_index[1], ms_stock_index[1])

        self.all_dates_list = make_trade_date_list(min_date, max_date, calendar)
        self.dict_date = {}
        self.all_dates_index = {}
        for idx, date in enumerate(self.all_dates_list):
            self.dict_date[date] = []
            self.all_dates_index[date] = idx

        for stock_code, stock_index in self.dict_stock.items():
            ms_stock_index = self.ms_dict_stock[stock_code]
            si = max(self.all_dates_index[stock_index[0]], self.all_dates_index[ms_stock_index[0]])
            ei = min(self.all_dates_index[stock_index[1]], self.all_dates_index[ms_stock_index[1]])
            online_dates = self.all_dates_list[si:ei + 1]
            for day in online_dates:
                self.dict_date[day].append(stock_code)

        print(f"{timestr()}创建日期索引，DONE")

    def create_data_loader(self, start_time, end_time, refresh=False, stride_shift=False):
        if refresh:
            self.re_shuffle_data_dict()

        time_start = __import__('time').time()
        print(f"{timestr()}创建data_loader。。。")
        data_set = []
        data_set_keys = list(self.data_set_dict.keys())
        si = 0
        if stride_shift:
            self.stride_offset = (self.stride_offset + 1) % self.args.stride
            si = self.stride_offset

        actual_start_time = get_n_step_work_date(start_time, self.need_day_len * -1 + 1)
        for day_index in range(si, len(data_set_keys), self.args.stride):
            day = data_set_keys[day_index]
            if day < actual_start_time:
                continue
            di = self.all_dates_list.index(day)
            actual_end_day = self.all_dates_list[di + self.need_day_len - 1]
            if actual_end_day > end_time:
                continue
            stock_list = self.data_set_dict[day]
            group_count = (len(stock_list) + self.args.sicount - 1) // self.args.sicount
            for i in range(group_count):
                data_set.append([day, i])

        if not data_set:
            raise Exception('数据集为空')

        data_loader = DataLoader(
            DiskBinDatasetTW_V11(self, data_set),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            prefetch_factor=4 if self.args.num_workers > 0 else None,
        )
        print(f"{timestr()}创建data_loader完成，耗时{__import__('time').time() - time_start:.2f}秒")
        return data_loader


class DiskBinDatasetON_V11(v5.DiskBinDatasetON):
    def read_si_seq(self, stock_list, start_date_key, minute_idx, need_seq_len):
        seq = super().read_si_seq(stock_list, start_date_key, minute_idx, need_seq_len)
        if self.period == 'five_minute':
            m5_first = getattr(self.args, 'm5_first_only', False)
            if m5_first:
                seq = seq[:, ::MINUTE_BARS_PER_DAY, :]
            seq = reshape_minute_tensor(
                seq, self.args.minute_before_num, self.args.minute_after_num, m5_first)
        return seq


class StockDataReaderON_V11(v5.StockDataReaderON):
    def create_data_loader(self, start_time, end_time, refresh=False, stride_shift=False):
        if refresh:
            self.re_shuffle_data_dict()

        time_start = __import__('time').time()
        print(f"{timestr()}创建data_loader。。。")
        data_set = []
        data_set_keys = list(self.data_set_dict.keys())
        actual_start_time = get_n_step_work_date(start_time, self.need_day_len * -1 + 1)

        if self.args.period == 'day':
            si = 0
            if stride_shift:
                self.stride_offset = (self.stride_offset + 1) % self.args.stride
                si = self.stride_offset
            for day_index in range(si, len(data_set_keys), self.args.stride):
                day = data_set_keys[day_index]
                if day < actual_start_time:
                    continue
                di = self.all_dates_list.index(day)
                actual_end_day = self.all_dates_list[di + self.need_day_len - 1]
                if actual_end_day > end_time:
                    continue
                stock_list = self.data_set_dict[day]
                group_count = (len(stock_list) + self.args.sicount - 1) // self.args.sicount
                for i in range(group_count):
                    data_set.append([day, i])
        elif self.args.period == 'five_minute':
            for minute_index in range(0, len(data_set_keys) * MINUTE_BARS_PER_DAY, self.args.stride):
                day_index = minute_index // MINUTE_BARS_PER_DAY
                day = data_set_keys[day_index]
                if day < actual_start_time:
                    continue
                actual_end_day = self.get_next_date(day, self.need_day_len)
                if actual_end_day == -1:
                    break
                if actual_end_day > end_time:
                    continue
                day_minute_index = minute_index % MINUTE_BARS_PER_DAY
                stock_list = self.data_set_dict[day]
                group_count = (len(stock_list) + self.args.sicount - 1) // self.args.sicount
                for i in range(group_count):
                    data_set.append([day, i, day_minute_index])
        else:
            raise ValueError('period 仅支持 day / five_minute')

        if not data_set:
            raise Exception('数据集为空')

        data_loader = DataLoader(
            DiskBinDatasetON_V11(
                self, self.args, data_set, self.dict_stock, self.dict_date,
                self.data_set_dict, self.all_dates_index, self.args.period,
            ),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            prefetch_factor=4 if self.args.num_workers > 0 else None,
        )
        print(f"{timestr()}创建data_loader完成，耗时{__import__('time').time() - time_start:.2f}秒")
        return data_loader


class CheckDataV11(v5.CheckData):
    def __init__(self, **kwargs):
        self.args = copy.deepcopy(kwargs.get('args'))
        self.scaler_info = {}
        dc = DatasetConfig()
        data_config = dc.get_config(self.args.data_name)
        self.data_path_bin = data_config.get('data_path_bin')
        self.index_file = data_config.get('index_file')
        self.scaler_file = data_config.get('scaler_file')
        self.channels = data_config.get('channels')
        self.period = data_config.get('period')

        if self.scaler_file and __import__('os').path.isfile(self.scaler_file):
            with open(self.scaler_file, 'r', encoding='utf-8') as f:
                self.scaler_info = json.load(f)
            self.field_list = list(self.scaler_info.keys())
        else:
            from processing_raw_data.filed_idx import xt_5mk_14f_field_idx
            self.field_list = list(xt_5mk_14f_field_idx.keys())
            self.scaler_info = {k: ['NONE'] for k in self.field_list}

        torch.set_printoptions(sci_mode=False)

    def validation_data_minute(self, *args, **kwargs):
        print('开始验证分钟数据。。。')
        input_seq = args[0]
        start_time = kwargs.get('start_time', 0)
        end_time = kwargs.get('end_time', 99999999)
        look_back_days = kwargs.get('look_back_windows', 0)
        m5_first_only = kwargs.get('m5_first_only', False)

        # 与 dataloader 一致：允许窗口内最早一天早于传入的 start_time
        start_time = get_n_step_work_date(start_time, look_back_days * -1 + 1)

        if isinstance(input_seq, torch.Tensor):
            bars = 1 if m5_first_only else MINUTE_BARS_PER_DAY
            expect = [
                self.args.batch_size,
                self.args.sicount,
                look_back_days,
                bars,
                self.channels,
            ]
            if list(input_seq.shape) != expect:
                print(f'minute_seq 形状有误, 期望 {expect}, 实际 {list(input_seq.shape)}')

        self.validate_detail_minute_5d(input_seq, start_time, end_time)
        print('验证分钟数据完成')

    def validate_detail_minute_5d(self, tensor_seq: torch.Tensor, start_time: int, end_time: int):
        year_i = self.field_list.index('gen_year')
        month_i = self.field_list.index('gen_month')
        day_i = self.field_list.index('gen_day')
        minute_i = self.field_list.index('gen_minute')

        b, m, l_day, l_bar, _ = tensor_seq.shape
        for bi in range(b):
            start_dt = end_dt = None
            for si in range(m):
                sy = int(tensor_seq[bi, si, 0, 0, year_i])
                sm = int(tensor_seq[bi, si, 0, 0, month_i])
                sd = int(tensor_seq[bi, si, 0, 0, day_i])
                smin = int(tensor_seq[bi, si, 0, 0, minute_i])
                ey = int(tensor_seq[bi, si, -1, -1, year_i])
                em = int(tensor_seq[bi, si, -1, -1, month_i])
                ed = int(tensor_seq[bi, si, -1, -1, day_i])
                emin = int(tensor_seq[bi, si, -1, -1, minute_i])
                sdt = sy * 1000000 + sm * 10000 + sd * 100 + smin
                edt = ey * 1000000 + em * 10000 + ed * 100 + emin
                if start_dt is None:
                    start_dt, end_dt = sdt, edt
                elif sdt != start_dt or edt != end_dt:
                    raise Exception(
                        f'股票维度时间不一致: b={bi} s={si} {sdt}/{edt} vs {start_dt}/{end_dt}')
            assert (start_time * 100 + 1) <= start_dt and (end_time * 100 + MINUTE_BARS_PER_DAY) >= end_dt, (
                f'信息泄露: {start_dt}~{end_dt}, 窗口 {start_time}~{end_time}')


class DataReaderV11:
    @staticmethod
    def create_data_reader(args):
        dc = DatasetConfig()
        if args.data_name in dc.config and args.ms_data_name in dc.config:
            return DailyWithMinuteReader(args)
        if args.data_name in dc.config:
            return StockDataReaderON_V11(args)
        if args.ms_data_name in dc.config:
            m_args = copy.deepcopy(args)
            m_args.data_name = args.ms_data_name
            return StockDataReaderON_V11(m_args)
        raise ValueError('data_name 和 ms_data_name 至少有一个合法选项')


# 对外统一别名
DataReader = DataReaderV11
CheckData = CheckDataV11

