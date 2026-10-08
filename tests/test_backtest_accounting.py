import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, ns):
    nodes = [n for n in ast.parse((ROOT / path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)
    return ns


class AccountingTests(unittest.TestCase):
    def test_template_turnover_cases(self):
        tr = functions('infer_ms/utils.py', {'cal_tr'}, {})['cal_tr']
        cases = [({}, {}, 0.), ({}, {'CASH': .2, 'A': .8}, .8),
                 ({'CASH': .2, 'A': .8}, {}, .8),
                 ({'CASH': .2, 'A': .8}, {'CASH': .4, 'A': .6}, .1),
                 ({'CASH': 0., 'A': .5, 'B': .5},
                  {'CASH': 0., 'A': .5, 'B': .5}, 0.)]
        for prev, nxt, expected in cases:
            self.assertAlmostEqual(tr(prev, nxt), expected)

    def test_template_execution_defaults(self):
        tree = ast.parse((ROOT / 'infer_ms/batch_test_ms.py').read_text())
        defaults = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                for keyword in node.keywords:
                    if keyword.arg == 'default' and isinstance(keyword.value, ast.Constant):
                        defaults[node.args[0].value] = keyword.value.value
        self.assertEqual(defaults['--price_target'], 'close_hfq')
        self.assertEqual(defaults['--close_limit_filter'], 1)
        period = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'period_test')
        call = next(n for n in ast.walk(period) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'cal_tr')
        self.assertEqual([n.id for n in call.args], ['prev_w', 'next_w'])

    def test_initial_buy_and_rebalance_cost(self):
        ns = functions('infer_ms/utils.py', {'cal_tr', 'drift_weights'}, {})
        tr = ns['cal_tr']
        # Preserve Template's initial double-sided fee convention.
        self.assertAlmostEqual(2 * tr({}, {'CASH': 0., 'A': 1.}) * .0005, .001)
        self.assertEqual(tr({'CASH': 0., 'A': 1.}, {'CASH': 0., 'B': 1.}), 1.)
        w = ns['drift_weights']({'CASH': 0., 'A': .5, 'B': .5},
                               {'A': 10., 'B': 10.}, {'A': 20., 'B': 10.})
        self.assertAlmostEqual(w['A'], 2 / 3)
        self.assertAlmostEqual(sum(w.values()), 1.)
        self.assertAlmostEqual(tr(w, {'CASH': 0., 'A': .5, 'B': .5}), 1 / 6)

    def test_template_summary_preserves_other_dates(self):
        ns = functions('infer_ms/batch_test_ms.py', {'make_summery_info', 'merge_dict'},
                       dict(json=json, os=os, UTILS=SimpleNamespace(make_output_flag=lambda a: 'experiment')))
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                Path('output').mkdir()
                fn = ns['make_summery_info']
                args = SimpleNamespace(tradeCost=.0005)
                fn(['20250101'], [1.01], [{'20250101': {'alpha': 1}}])
                fn(['20260101'], [1.02], [{'20260101': {'alpha': 2}}])
                x = json.loads(Path('output/pd_list.json').read_text())
                self.assertEqual(set(x), {'20250101', '20260101'})
            finally:
                os.chdir(old)

    def test_resume_matches_template_skip_behavior(self):
        saved = {'CASH': .25, 'A': .75}
        received = []
        args = SimpleNamespace(TD=1, keepWeightDays=1, withBJ=False, shuffle=0)
        # Stop after new inference; this isolates state recovery from market reporting.
        class Done(Exception): pass
        def infer(*a, **kw):
            received.append(kw['last_weight'])
            raise Done()
        ns = functions('infer_ms/batch_test_ms.py', {'period_test'},
                       dict(os=SimpleNamespace(path=SimpleNamespace(join=lambda *a: '/'.join(a),
                                                                     exists=lambda p: p.endswith('1.json'))),
                            DT=SimpleNamespace(trade_date_list=lambda *a: ['1', '2'],
                                               get_rid_of_none_trading_days=lambda x: x, timestr=lambda: ''),
                            UTILS=SimpleNamespace(make_output_json_name=lambda a, d: d + '.json',
                                                  read_pmfile=lambda p: saved),
                            time=SimpleNamespace(time=lambda: 0), model_args=None,
                            InferTool=SimpleNamespace(infer_one_day=infer)))
        with self.assertRaises(Done):
            ns['period_test'](args, [], '1', '2')
        self.assertEqual(received, [None])


if __name__ == '__main__':
    unittest.main()
