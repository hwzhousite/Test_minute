#!/bin/bash
# 2018-2026 分年并行回测 (保留旧的位置参数用法, 实际调用 ../backtest.sh, 更多选项见 ../backtest.sh -h)
# 结束日期默认 20260722, 可用环境变量 END 覆盖, 例: END=20260930 ./kk18_26.sh ...
if [ $# -lt 7 ]; then
    echo "xxx.sh [数据集名称] [模型路径] [1/0是否含北交所] [模型类型] [mask_mf] [indexName] [t/task] [其他参数...]"
    exit 1
fi
exec "$(dirname "$0")/../backtest.sh" -k day -d "$1" -m "$2" -b "$3" -t "$4" -x "$5" -i "$6" \
    -s 20180101 -e "${END:-20260722}" -g 0,1,2,3,4,5,6,7 -y -- --t="$7" "${@:8}"
