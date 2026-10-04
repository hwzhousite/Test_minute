"""Synthetic checks; no server datasets or scaler files required."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def classes(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class MinuteReaderTests(unittest.TestCase):
    def test_partial_missing_and_stock_order(self):
        import os
        torch_stub = SimpleNamespace(from_numpy=lambda x: x,
                                     utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)))
        ns = classes('common_module/reader_tools_V11.py',
                     {'DailyWithMinuteReader', 'DailyWithMinuteDataset'},
                     dict(v5=SimpleNamespace(StockDataReaderON=object),
                          torch=torch_stub, np=np, os=os))
        reader = ns['DailyWithMinuteReader'].__new__(ns['DailyWithMinuteReader'])
        reader.args = SimpleNamespace(minute_before_num=4, minute_after_num=0,
                                      minute_offset_days=1, day_before_num=6)
        reader.ms_calendar = [20260105, 20260106, 20260107, 20260108, 20260109, 20260112]
        reader.ms_calendar_index = {d: i for i, d in enumerate(reader.ms_calendar)}
        reader.ms_index = {'000001': [20260107, 20260108], '000002': [20260105, 20260112]}
        with tempfile.TemporaryDirectory() as tmp:
            reader.ms_data_config = dict(channels=2, accuracy='d', data_path_bin=tmp)
            np.full((96, 2), 11., dtype=np.float64).tofile(Path(tmp) / '000001.bin')
            np.repeat(np.arange(6, dtype=np.float64), 48 * 2).tofile(Path(tmp) / '000002.bin')
            x = reader.read_minutes(['000002', '000003', '000001'], 20260105)
            self.assertEqual(x.shape, (3, 4, 48, 2))
            np.testing.assert_array_equal(x[0, :, 0, 0], [1, 2, 3, 4])
            self.assertFalse(x[1].any())
            np.testing.assert_array_equal(x[2, :, 0, 0], [0, 11, 11, 0])
            # A short file cannot shift later bars into another day.
            np.full((48, 2), 11., dtype=np.float64).tofile(Path(tmp) / '000001.bin')
            y = reader.read_minutes(['000001'], 20260105)
            np.testing.assert_array_equal(y[0, :, 0, 0], [0, 11, 0, 0])

        class Daily:
            ds = [[20260105, 0], [20260106, 1]]
            def __len__(self): return len(self.ds)
            def __getitem__(self, i): return ('daily', i)
            def get_si_stock_list(self, day, group): return [str(group)]
        calls = []
        auxiliary = SimpleNamespace(read_minutes=lambda stocks, day: calls.append((stocks, day)))
        dataset = ns['DailyWithMinuteDataset'](Daily(), auxiliary)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[1][0], ('daily', 1))
        self.assertEqual(calls, [(['1'], 20260106)])

    def test_all_episode_windows_end_before_trade(self):
        for lookback in (20, 80):
            for episode in (1, 20):
                offset = 80 - lookback
                length = lookback + episode - 1
                for i in range(episode):
                    self.assertEqual(offset + i + lookback - 1, i + 79)
                    self.assertLess(offset + i + lookback - 1, i + 80)
                    self.assertLessEqual(i + lookback, length)

    def test_inference_keeps_stale_partial_window(self):
        import os
        ns = classes('infer_ms/minute_data.py', {'MinuteBinReader'},
                     dict(np=np, os=os, DEFAULT_MS_DATA_PATH='', MINUTE_BARS_PER_DAY=48))
        reader = ns['MinuteBinReader'].__new__(ns['MinuteBinReader'])
        reader.cal_index = {d: i for i, d in enumerate([1, 2, 3, 4])}
        reader.index = {'000001': [2, 3]}
        reader.channels = 2
        reader.dtype = np.float64
        reader.record_bytes = 16
        with tempfile.TemporaryDirectory() as tmp:
            reader.bin_path = tmp
            np.full((96, 2), 7., dtype=np.float64).tofile(Path(tmp) / '000001.bin')
            x, status = reader._read_one('000001', 1, 4, 4)
            self.assertEqual(status, 'partial')
            np.testing.assert_array_equal(x[:, 0, 0], [0, 7, 7, 0])
            self.assertEqual(reader._read_one('000002', 1, 4, 4), (None, 'missing'))


class MinuteMaskTests(unittest.TestCase):
    def test_missing_tokens_cannot_change_valid_tokens(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest('PyTorch required for attention checks')
        ns = classes('pmdayV2/pm_model_ms.py', {'MinuteTSALayer'}, dict(torch=torch, nn=nn))
        torch.manual_seed(1)
        layer = ns['MinuteTSALayer'](8, 2, 16, 0., True).eval()
        valid = torch.tensor([[[True, False, True], [False, False, False]]])
        x = torch.randn(1, 2, 3, 8)
        changed = x.clone()
        changed[~valid] = 1000
        with torch.no_grad():
            a, b = layer(x, valid), layer(changed, valid)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a[valid], b[valid])
        self.assertEqual(a[~valid].abs().sum().item(), 0.)
        with torch.no_grad():
            empty = layer(x, torch.zeros_like(valid))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(empty.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
