# 获取上一级目录
dir=$(dirname $(pwd))
project_name=$(basename $dir)

#使用_对project_name进行分割，并获取第一个字段
project_name=${project_name%%_*}

send $dir g6 g8 


# Ex26_2日频框架 + 5分钟分支 (M须为600, 与IndexTargetGenerator一致; 其余参数请与Ex26_2的启动参数保持一致)
post ./yy_train_pm_ms.py g7 g6 g8 "--mark ${project_name} --M 600 --use_minute 1 --ms_time_step 20 --ms_patch_size 12"