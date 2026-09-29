#!/bin/bash
if [ $# -ne 7 ]; then
    echo "xxx.sh [数据集名称] [模型路径] [1/0是否含北交所] [模型类型字A/B/C] [mask_mf] [indexName] [t/task]"
    exit 1
fi
python -W ignore -u ./batch_test_L.py --SUM=1 --t=$7 --mask_mf=$5 --indexName=$6 --MODELTYPE=$4 --DS=$1 --model=$2 --withBJ=$3 --GPU=7 --period=week --cfd=20260101 --cld=20260722


