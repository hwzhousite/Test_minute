"""Synthetic binary I/O tests; no production calendar or PyTorch installation required.

Only tensor wrapping is replaced by NumPy; production indexing and file I/O run
unchanged. Full DataLoader/GPU integration must be checked in the training env.
"""
import ast
import math
import os
import tempfile
import unittest
import warnings
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ArrayTensor(np.ndarray):
    def float(self):
        return self.astype(np.float32)

    def view(self, *shape):
        return self.reshape(*shape)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(ArrayTensor)


torch = SimpleNamespace(tensor=tensor, Tensor=ArrayTensor, float64=np.float64,
                        stack=lambda rows, dim=0: tensor(np.stack(rows, axis=dim)),
                        empty=lambda shape: tensor(np.empty(shape, np.float32)))
ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef)
                     and n.name == class_name)
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


class MinuteReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ns = dict(np=np, torch=torch, os=os, warnings=warnings, lru_cache=lru_cache)
        load_functions(ROOT / 'common_module/reader_tools_V5_calendar.py',
                       {'canonical_stock_code', 'normalize_stock_index', 'resolve_stock_bin',
                        '_minute_file_index', 'read_minute_calendar_seq'}, self.ns)
        self.read = self.ns['read_minute_calendar_seq']

    def write(self, specs, name='000001.SZ.bin'):
        rows = np.zeros((len(specs), 14), np.float64)
        for row, (day, bar, close) in zip(rows, specs):
            row[0] = 1
            row[2:5] = [day // 10000, day // 100 % 100, day % 100]
            row[6], row[10] = bar, close
        rows.tofile(Path(self.tmp.name) / name)

    def test_sparse_unsorted_rows_do_not_shift_days_or_bars(self):
        self.write([(20261009, 48, 30), (20261008, 2, 10), (20261009, 1, 20)])
        out = self.read(self.tmp.name, '1', [20261008, 20261009]).reshape(2, 48, 14)
        self.assertEqual(out[0, 1, 10], 10)
        self.assertEqual(out[1, 0, 10], 20)
        self.assertEqual(out[1, 47, 10], 30)
        self.assertEqual(out[0, 0, 13], 1)
        np.testing.assert_array_equal(out[0, 0, 2:5], [2026, 10, 8])

    def test_zero_close_is_padding(self):
        self.write([(20261009, 1, 0), (20261009, 2, 20)])
        out = self.read(self.tmp.name, '000001.SZ', [20261009])
        self.assertEqual(out[0, 13], 1)
        self.assertEqual(out[1, 13], 0)

    def test_missing_file_warns_or_errors(self):
        with self.assertWarns(RuntimeWarning):
            out = self.read(self.tmp.name, '1', [20261009])
        self.assertTrue((out[:, 13] == 1).all())
        with self.assertRaises(FileNotFoundError):
            self.read(self.tmp.name, '1', [20261009], missing_policy='error')

    def test_no_matching_window_errors_with_diagnostics(self):
        self.write([(20261009, 1, 10)])
        with self.assertRaisesRegex(ValueError, 'matched=0'):
            self.read(self.tmp.name, '1', [20261008], missing_policy='error')

    def test_duplicate_and_truncated_records_fail(self):
        self.write([(20261009, 1, 10), (20261009, 1, 20)])
        with self.assertRaisesRegex(ValueError, '重复'):
            self.read(self.tmp.name, '1', [20261009])
        (Path(self.tmp.name) / '000001.SZ.bin').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, '大小'):
            self.read(self.tmp.name, '1', [20261009])

    def test_file_change_invalidates_index(self):
        self.write([(20261008, 1, 10)])
        self.read(self.tmp.name, '1', [20261008])
        self.write([(20261008, 1, 10), (20261009, 1, 20)])
        self.assertEqual(self.read(self.tmp.name, '1', [20261009])[0, 10], 20)

    def test_active_reader_ignores_stale_index_and_preserves_offset(self):
        self.write([(20261009, 1, 20), (20261009, 48, 30)])
        ns = dict(np=np, torch=torch, v5=SimpleNamespace(**self.ns), MINUTE_BARS_PER_DAY=48)
        load_functions(ROOT / 'common_module/reader_tools_V11.py', {'read_minutes'}, ns,
                       'DailyWithMinuteReader')
        reader = SimpleNamespace(
            args=SimpleNamespace(minute_before_num=1, minute_after_num=0,
                                 day_before_num=2, minute_offset_days=1,
                                 m5_first_only=False, minute_missing_policy='error'),
            ms_data_config=dict(channels=14, raw_data='Y', accuracy='d', data_path_bin=self.tmp.name),
            ms_calendar=[20261008, 20261009], ms_calendar_index={20261008: 0, 20261009: 1},
            ms_index={})
        out = ns['read_minutes'](reader, ['1'], 20261008)
        self.assertEqual(out.shape, (1, 1, 48, 14))
        self.assertEqual(out[0, 0, 47, 10], 30)
        reader.args.m5_first_only = True
        out = ns['read_minutes'](reader, ['1'], 20261008)
        self.assertEqual(out.shape, (1, 1, 1, 14))
        self.assertEqual(out[0, 0, 0, 10], 20)
        reader.args.minute_offset_days = 2
        with self.assertRaisesRegex(ValueError, '日历'):
            ns['read_minutes'](reader, ['1'], 20261008)

    def test_stock_key_collision_fails(self):
        with self.assertRaises(ValueError):
            self.ns['normalize_stock_index']({'1': [], '000001.SZ': []})


if __name__ == '__main__':
    unittest.main()
