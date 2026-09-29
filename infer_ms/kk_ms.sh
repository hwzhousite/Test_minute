#!/bin/bash
# 5分钟模型逐日推理回测
# 用法: ./kk_ms.sh [模型路径] [日线csv目录] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint]
# 例:   ./kk_ms.sh ../pmdayV2/output/T1V2ms_epoch12.pth /swdata/yy_data/wd395/csv 20260101 20260722 0 zz1000 0
if [ $# -ne 7 ]; then
    echo "kk_ms.sh [模型路径] [日线csv目录] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint]"
    exit 1
fi
python -W ignore -u ./batch_test_ms.py --SUM=1 --DS=wd395 --mask_mf=602 --multitask=0 --MODELTYPE=A \
    --model=$1 --csv_path=$2 --cfd=$3 --cld=$4 --GPU=$5 --indexName=$6 --indexTarget=$6 --Factor_constraint=$7 \
    --withBJ=1 --period=week
