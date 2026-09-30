#!/bin/bash
# 通用回测脚本
#   -k day : infer/batch_test_L.py   (pm_model 日线模型)
#   -k ms  : infer_ms/batch_test_ms.py (pm_model_ms 日线+5分钟模型)
# 可在任意目录下运行, 脚本会自动切换到对应的推理目录; 相对模型路径按当前目录解析

usage() {
    cat <<'EOF'
用法: backtest.sh -m 模型路径 [选项] [-- 透传给batch_test的其他参数]

选项:
  -k day|ms     回测类型, 默认 ms
  -m PATH       模型文件(.pth), 必填
  -s YYYYMMDD   起始日期, 默认 20180101
  -e YYYYMMDD   结束日期, 默认 今天
  -g GPUS       GPU列表, 逗号分隔, 默认 0; 分年模式下各年份轮流分配
  -i NAME       指数名称 indexName, 默认 zz1000 (ms模式同时作为 indexTarget)
  -d DS         数据集, 默认 ms=wd395, day=wd311
  -t TYPE       MODELTYPE, 默认 ms=A, day=RONWIND
  -x N          mask_mf, 默认 ms=602, day=503
  -b 0|1        是否包含北交所, 默认 1
  -y            按年份拆分并行回测, 全部完成后再汇总
  -j N          分年模式下最大并行进程数, 默认 0 = 不限制
  -c            回测前先清理日志和 output/ (等同于 ./cls)
  -h            显示帮助

例:
  ./backtest.sh -m pmdayV2/output/Ex27_epoch2.pth -s 20260101 -e 20260722
  ./backtest.sh -m pmdayV2/output/Ex27_epoch2.pth -y -g 0,1,2,3 -- --Factor_constraint=2
  ./backtest.sh -k day -m pmdayV2/output/T1V2.pth -y -g 0,1,2,3,4,5,6,7 -- --t=3
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python}

KIND=ms MODEL='' START=20180101 END=$(date +%Y%m%d) GPUS=0 INDEX=zz1000
DS='' MODELTYPE='' MASK_MF='' BJ=1 SPLIT=0 JOBS=0 CLEAN=0

while getopts "k:m:s:e:g:i:d:t:x:b:j:ych" opt; do
    case $opt in
        k) KIND=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        s) START=$OPTARG ;;
        e) END=$OPTARG ;;
        g) GPUS=$OPTARG ;;
        i) INDEX=$OPTARG ;;
        d) DS=$OPTARG ;;
        t) MODELTYPE=$OPTARG ;;
        x) MASK_MF=$OPTARG ;;
        b) BJ=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        y) SPLIT=1 ;;
        c) CLEAN=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")  # 透传参数放在最后, argparse 以最后出现的为准, 可覆盖下面的默认值

if [ -z "$MODEL" ]; then echo "错误: 必须用 -m 指定模型路径"; usage; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "错误: 模型文件不存在: $MODEL"; exit 1; fi
MODEL=$(realpath "$MODEL")
if [[ ! $START =~ ^[0-9]{8}$ || ! $END =~ ^[0-9]{8}$ || $START > $END ]]; then
    echo "错误: 日期区间不合法: $START - $END"; exit 1
fi

case $KIND in
    ms)
        WORKDIR=$ROOT/infer_ms SCRIPT=batch_test_ms.py
        DS=${DS:-wd395} MODELTYPE=${MODELTYPE:-A} MASK_MF=${MASK_MF:-602}
        KIND_ARGS=(--multitask=0 --indexTarget="$INDEX")
        ;;
    day)
        WORKDIR=$ROOT/infer SCRIPT=batch_test_L.py
        DS=${DS:-wd311} MODELTYPE=${MODELTYPE:-RONWIND} MASK_MF=${MASK_MF:-503}
        KIND_ARGS=()
        ;;
    *) echo "错误: -k 只能是 day 或 ms"; exit 1 ;;
esac

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
COMMON=(--DS="$DS" --MODELTYPE="$MODELTYPE" --mask_mf="$MASK_MF" --model="$MODEL"
        --indexName="$INDEX" --withBJ="$BJ" --period=week "${KIND_ARGS[@]}")

cd "$WORKDIR" || exit 1
if [ $CLEAN -eq 1 ]; then rm -f ./*log* && rm -rf output/*; fi
mkdir -p output

echo "[$(date +%H:%M:%S)] 回测: $SCRIPT 模型=$(basename "$MODEL") 区间=[$START, $END] DS=$DS MODELTYPE=$MODELTYPE mask_mf=$MASK_MF 指数=$INDEX GPU=$GPUS"

if [ $SPLIT -eq 0 ]; then
    # 单进程: 直接跑完整区间并汇总
    $PY -W ignore -u "./$SCRIPT" --SUM=1 "${COMMON[@]}" --GPU="${GPU_LIST[0]}" \
        --cfd="$START" --cld="$END" "${EXTRA[@]}" 2>&1 | tee "bt_log_${START}_${END}.log"
    exit "${PIPESTATUS[0]}"
fi

# 分年并行: 每个年份一个进程, GPU 轮流分配; 已存在的每日json会被跳过, 所以中断后可以直接重跑续算
declare -A PID_YEAR
n=0
for ((y = ${START:0:4}; y <= ${END:0:4}; y++)); do
    s=$(( y == ${START:0:4} ? START : y * 10000 + 101 ))
    e=$(( y == ${END:0:4} ? END : y * 10000 + 1231 ))
    gpu=${GPU_LIST[$((n % ${#GPU_LIST[@]}))]}
    if [ "$JOBS" -gt 0 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    fi
    $PY -W ignore -u "./$SCRIPT" "${COMMON[@]}" --GPU="$gpu" --cfd="$s" --cld="$e" "${EXTRA[@]}" \
        > "bt_log_${y}.log" 2>&1 &
    PID_YEAR[$!]=$y
    echo "[$(date +%H:%M:%S)] 启动 $y 年 [$s, $e] GPU=$gpu 日志=bt_log_${y}.log"
    n=$((n + 1))
done

failed=()
for pid in "${!PID_YEAR[@]}"; do
    wait "$pid" || failed+=("${PID_YEAR[$pid]}")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "[$(date +%H:%M:%S)] 以下年份回测失败, 请查看对应的 bt_log_<年份>.log: ${failed[*]}"
    exit 1
fi

echo "[$(date +%H:%M:%S)] 所有分年回测都已完成, 生成汇总信息..."
$PY -W ignore -u "./$SCRIPT" --SUM=1 "${COMMON[@]}" --GPU="${GPU_LIST[0]}" \
    --cfd="$START" --cld="$END" "${EXTRA[@]}" 2>&1 | tee "bt_log_sum_${START}_${END}.log"
exit "${PIPESTATUS[0]}"
