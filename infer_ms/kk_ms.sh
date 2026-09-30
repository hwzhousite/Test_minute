#!/bin/bash
# 5分钟模型逐日推理回测 (保留旧的位置参数用法, 实际调用 ../backtest.sh, 更多选项见 ../backtest.sh -h)
# 用法: ./kk_ms.sh [模型路径] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint] [其他batch_test_ms.py参数...]
# 例:   ./kk_ms.sh ../pmdayV2/output/Ex27_epoch2.pth 20260101 20260722 0 zz1000 0
if [ $# -lt 6 ]; then
    echo "kk_ms.sh [模型路径] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint] [其他参数...]"
    exit 1
fi
exec "$(dirname "$0")/../backtest.sh" -k ms -m "$1" -s "$2" -e "$3" -g "$4" -i "$5" -- --Factor_constraint="$6" "${@:7}"
