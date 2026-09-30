#!/bin/bash
# 5分钟模型逐日推理回测
# 日线数据默认使用 /data/yy_data/qd/train/wd_395f_0819/260824/pads (见batch_test_ms.py DS=wd395), 如需更换请直接给batch_test_ms.py传--csv_path
# 用法: ./kk_ms.sh [模型路径] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint]
# 例:   ./kk_ms.sh ../pmdayV2/output/T1V2ms_epoch12.pth 20260101 20260722 0 zz1000 0
if [ $# -ne 6 ]; then
    echo "kk_ms.sh [模型路径] [起始日期] [结束日期] [GPU] [indexName] [Factor_constraint]"
    exit 1
fi
python -W ignore -u ./batch_test_ms.py --SUM=1 --DS=wd395 --mask_mf=602 --multitask=0 --MODELTYPE=A \
    --model=$1 --cfd=$2 --cld=$3 --GPU=$4 --indexName=$5 --indexTarget=$5 --Factor_constraint=$6 \
    --withBJ=1 --period=week
