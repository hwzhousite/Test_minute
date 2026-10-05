"""
推理用5分钟数据读取

直接读取训练时使用的 bin 数据集(xt_260527_14f), 字段、顺序与训练完全一致:
  - {base}/bin_data/{6位代码}.bin: 每只股票一个文件, 从该股票首个交易日起, 每个交易日48根5分钟bar, 每根bar C个float64
  - {base}/index.json: {6位代码: [首日, 末日]}
  - {base}/scaler_info.txt: 字段字典(key的顺序即字段顺序)
行号按交易日历计算(与训练reader的all_dates_index一致): 第D天的首根bar = (D在日历中的序号 - 首日序号) * 48
分钟缺失或未更新的日期填零，默认退回日线；可显式设置min_cover启用覆盖率检查
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import utils_date as DT

MINUTE_BARS_PER_DAY = 48
DEFAULT_MS_DATA_PATH = '/data/yy_data/five_minute_data/xt_260527_14f'


class MinuteBinReader:
    def __init__(self, base_path=DEFAULT_MS_DATA_PATH, channels=14, accuracy='d', workers=16):
        self.bin_path = os.path.join(base_path, 'bin_data')
        with open(os.path.join(base_path, 'index.json'), 'r', encoding='utf-8') as f:
            self.index = {str(k).zfill(6): [int(v[0]), int(v[1])] for k, v in json.load(f).items()}
        with open(os.path.join(base_path, 'scaler_info.txt'), 'r', encoding='utf-8') as f:
            self.fields = list(json.load(f).keys())
        assert len(self.fields) == channels, f"分钟数据字段数{len(self.fields)}与channels={channels}不一致"
        with open(DT.g_calendar_file, 'r', encoding='utf-8-sig') as f:
            self.calendar = [int(d) for d in json.load(f)]
        self.cal_index = {d: i for i, d in enumerate(self.calendar)}
        self.channels = channels
        self.dtype = np.float64 if accuracy == 'd' else np.float32
        self.record_bytes = channels * np.dtype(self.dtype).itemsize
        self.workers = workers
        self.date_fields = [self.fields.index(f) for f in ('gen_year', 'gen_month', 'gen_day')] \
            if all(f in self.fields for f in ('gen_year', 'gen_month', 'gen_day')) else None
        self.data_last_day = max((v[1] for v in self.index.values()), default=0)
        print(f"分钟数据集{base_path}: 股票{len(self.index)}只, 数据最后日期{self.data_last_day}")

    def _read_one(self, code, first_day, last_day, days):
        """
        读取一只股票[first_day, last_day]区间的分钟数据, 返回 ([days,48,C] 或 None, 状态)
        新股(首日晚于first_day)只填充可用部分, 其余为0(模型会将0值bar视为无效)
        """
        info = self.index.get(code)
        if info is None:
            return None, 'missing'
        s, e = info
        read_last = min(e, last_day)
        if s > last_day:
            return None, 'missing'
        read_first = max(s, first_day)
        if read_first > read_last:
            return None, 'stale'
        start_row = (self.cal_index[read_first] - self.cal_index[s]) * MINUTE_BARS_PER_DAY
        n_days = self.cal_index[read_last] - self.cal_index[read_first] + 1
        count = n_days * MINUTE_BARS_PER_DAY * self.channels
        path = os.path.join(self.bin_path, f"{code}.bin")
        if not os.path.isfile(path):
            return None, 'missing'
        data = np.fromfile(path, dtype=self.dtype, count=count,
                           offset=start_row * self.record_bytes)
        out = np.zeros((days, MINUTE_BARS_PER_DAY, self.channels), dtype=np.float32)
        bars = data.size // self.channels
        begin = (self.cal_index[read_first] - self.cal_index[first_day]) * MINUTE_BARS_PER_DAY
        out.reshape(-1, self.channels)[begin:begin + bars] = data[:bars * self.channels].reshape(bars, self.channels)
        return out, ('partial' if n_days < days or data.size != count else 'ok')

    def read(self, stock_list, last_day, days, min_cover=0.0):
        """
        读取推理股票池截至last_day(含)最近days个交易日的分钟数据
        :param stock_list: 股票代码列表(如'000001.SZ'), 顺序与日线inputSeq的股票维一致
        :param last_day: 分钟窗口最后一天, 必须与日线回看窗口最后一天相同(即T-1)
        :param days: 天数(模型的ms_time_step)
        :param min_cover: 完整或部分分钟数据的股票占比下限，默认0允许日线回退
        :return: [1, M, days, 48, C] float32
        """
        last_day = int(last_day)
        if last_day not in self.cal_index:
            raise ValueError(f"分钟窗口最后一天{last_day}不是交易日")
        li = self.cal_index[last_day]
        if li - days + 1 < 0:
            raise ValueError(f"交易日历不足以回看{days}天")
        first_day = self.calendar[li - days + 1]

        codes = [str(c).split('.')[0].zfill(6) for c in stock_list]
        M = len(codes)
        out = np.zeros((M, days, MINUTE_BARS_PER_DAY, self.channels), dtype=np.float32)
        stat = {'ok': 0, 'partial': 0, 'missing': 0, 'stale': 0, 'short': 0}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(lambda c: self._read_one(c, first_day, last_day, days), codes))
        for j, (arr, st) in enumerate(results):
            stat[st] += 1
            if arr is not None:
                out[j] = arr

        cover = (stat['ok'] + stat['partial']) / max(M, 1)
        print(f"读取分钟数据[{first_day}~{last_day}]共{days}天, 股票{M}只: {stat}, 覆盖率{cover:.2%}")
        if cover < min_cover:
            raise ValueError(
                f"分钟数据覆盖率{cover:.2%}低于{min_cover:.0%}: 其中{stat['stale']}只未更新到{last_day}"
                f"(数据集最后日期{self.data_last_day}), {stat['missing']}只缺失. 请先更新5分钟bin数据")

        self._check_dates(out, first_day, days)
        return torch.from_numpy(out).unsqueeze(0)

    def _check_dates(self, out, first_day, days):
        """用分钟数据中的日期字段核对每天的日期是否与交易日历一致(全0的停牌/缺失日跳过)"""
        if self.date_fields is None:
            return
        yi, mi, di = self.date_fields
        bar0 = out[:, :, 0, :]  # [M, days, C]
        got = bar0[..., yi].round().astype(np.int64) * 10000 + bar0[..., mi].round().astype(np.int64) * 100 \
            + bar0[..., di].round().astype(np.int64)
        fi = self.cal_index[first_day]
        expect = np.array(self.calendar[fi:fi + days], dtype=np.int64)[None, :]
        valid = got > 0
        bad = valid & (got != expect)
        if bad.any():
            m, j = [int(v) for v in np.argwhere(bad)[0]]
            raise ValueError(f"分钟数据日期错位: 第{m}只股票第{j}天应为{expect[0, j]}, 实际为{got[m, j]}")
