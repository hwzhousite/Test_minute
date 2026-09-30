#!/bin/bash
# 5分钟模型(日线395f+5分钟) 分年并行回测 + 汇总, 参数顺序与之前日频的 batch_test_X.py 脚本一致
# 实际调用 ../backtest.sh -k ms -y (固定 --mask_mf=602 --MODELTYPE=A), 更多选项见 ../backtest.sh -h
# 默认区间 20180101-20251231, 每年一个进程, GPU 0-7 轮流分配; 可用环境变量覆盖:
#   START=20190101 END=20260722 GPUS=0,1,2,3 ./kk_ms_year.sh ...
# 例: ./kk_ms_year.sh wd395 ../pmdayV2/output/Ex27_epoch2.pth 1 0 0 zz1000 0 0.0005 zz1000
if [ $# -lt 9 ]; then
    echo "kk_ms_year.sh [数据集名称] [模型路径] [1/0是否含北交所] [indexpull] [multitask] [indexName] [Factor_constraint] [minw] [indexTarget] [其他参数...]"
    exit 1
fi
exec "$(dirname "$0")/../backtest.sh" -k ms -d "$1" -m "$2" -b "$3" -i "$6" \
    -s "${START:-20180101}" -e "${END:-20251231}" -g "${GPUS:-0,1,2,3,4,5,6,7}" -y -- \
    --indexpull="$4" --multitask="$5" --Factor_constraint="$7" --minw="$8" --indexTarget="$9" "${@:10}"
