import copy
import os
import platform
import json

import torch
from common_module import yy_common as CM
import argparse


def get_args(run_mode='train', episode_len=20, model='PM_SMV2', loadcheck='', mark='', M=200):
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--num_node', type=int, default=1, help='lt训练时使用的节点数')
    parser.add_argument('--gpu_num', type=int, default=0, help='使用的卡数')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--agb', type=int, default=1, help='梯度累积')
    parser.add_argument('--model', type=str, default=model, help='当前训练的模型')
    parser.add_argument('--price_target', type=str, default='5m', help='价格目标[open,935,5m,60m,day]')
    parser.add_argument('--RUN_MODE', type=str, default=run_mode, help='当前训练模式[train,check]')
    parser.add_argument('--mark', type=str, default=mark, help='name str')
    parser.add_argument('--loadcheck', type=str, default=loadcheck, help='load check point')
    parser.add_argument('--save_path', type=str, default="./output", help='vali data file')
    parser.add_argument('--epno', type=int, default=0, help='训练起始轮数，用于中断后接续训练')

    # forecasting task
    parser.add_argument('--time_step', type=int, default=20, help='input sequence length')
    parser.add_argument('--ffv_day', type=int, default=5, help='未来价值日,它必须小于等于patch_size')
    parser.add_argument('--ffv_discount', type=int, default=0, help='未来价值是否折扣,默认为0')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction length 当前要求它必须与patch_size完全相等!!!')
    parser.add_argument('--episode_len', type=int, default=episode_len, help='trade days per episode')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--stride_val', type=int, default=1, help='stride')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--lossfunc', type=str, default='ir', help='损失函数类型')
    parser.add_argument('--vf', type=float, default=0, help="计算value时v(t1,t2)的系数")
    parser.add_argument('--M', type=int, default=M, help='资产组合中的股票数量')
    parser.add_argument('--val_M', type=int, default=2500, help='验证时每个样本中的股票数')
    parser.add_argument('--val_B', type=int, default=1, help='验证时每个样本中的股票数')
    parser.add_argument('--minw', type=float, default=0.001, help='最小权重,默认0.001,推理时模型外使用,用于smooth')
    parser.add_argument('--layers', type=int, default=3, help='cross layers')

    # Add by ZenithHorizon
    parser.add_argument('--valid_episode_len', type=int, default= 20, help='当前训练的模型')
    parser.add_argument('--valid_stride', type=int, default=5, help='valid stride')
    parser.add_argument('--stride_shift', type=bool, default=True, help='shift stride')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.985, help='学习率衰减率')
    parser.add_argument('--lr_patience', type=int, default=3, help='连续几个epoch未优化时，学习率衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--GPU', type=str, default='A',
                        help='[a|A|0,1,2..], A表示全部，a表示除最后一个卡之外的卡，0,1,2...表示具体卡号')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--kfold', type=int, default=1, help='k-fold training, k为序号, 默认为0表示不使用k-fold')

    args = parser.parse_args()
    assert 2 <= args.ffv_day <= args.pred_len, "未来价值日,它必须大于1, 小于等于patch_size"
    if args.gpu_num == 7:
        args.GPU = 'a'
    elif args.gpu_num == 8:
        args.GPU = 'A'
    if args.stride_val == 0:
        args.stride_val = args.stride

    main_path = os.path.dirname(os.path.abspath(__file__))
    if args.loadcheck.startswith('./'):
        args.loadcheck = os.path.join(main_path, args.loadcheck[2:])
    if args.save_path.startswith('./'):
        args.save_path = os.path.join(main_path, args.save_path[2:])

    if args.loadcheck == '' and args.RUN_MODE == 'train' and args.mark != '' and args.epno > 0:
        args.loadcheck = os.path.join(str(args.save_path), f"{args.mark}.pth")

    # args = try_load_mark_args(args, args.loadcheck)

    args.dataset_name = 'wd_260616_311f'
    args.scaler_info_path = '/data/yy_data/qd/train/wd_311f_hs/0616/scaler_info.json'
    args.ms_dataset_name = 'xt_260527_14f'
    args.ms_scaler_info_path = '/data/yy_data/five_minute_data/xt_260527_14f/scaler_info.txt'

    k_fold_dates = [
        [[20100104, 20171231], [20180101, 20231231], [20190401, 20201231]],
        [[20100104, 20191231], [20201001, 20241231], [20220401, 20231231]],
        [[20100104, 20211231], [20221001, 20251231], [20241001, 20250630]],
        [[20100104, 20231231], [20240101, 20251231], [20241001, 20250630]],
        [[20100104, 20251231], [20260101, 20260531], [20241001, 20250630]],
    ]
    if args.RUN_MODE == 'train' and args.kfold > 0:
        assert args.kfold <= len(k_fold_dates)
        args.train_date_area, args.valid_date_area, args.valid_date_area_2 = k_fold_dates[args.kfold - 1]
        print(f"K-fold trainging, K=#{args.kfold}")

    args.train_data_param = {args.dataset_name: ''}
    args.valid_data_param = {args.dataset_name: ''}
    args.test_data_param={args.dataset_name: ''}
    args.jiaoji_mode='stop_limit_mode'
    args.stop_limit=20

    args.src_field_list = ['f_d_code_int','f_d_code_id','f_d_exchange_id','f_d_market_id','f_d_industry_id','f_d_year','f_d_month','f_d_day','f_d_week',
                           'f_s_is_dividend','f_p_open','f_p_high','f_p_low','f_p_close','f_p_pre_close','f_i_change','f_i_pct_chg',
                           'f_v_volume','f_v_amount','f_p_open_adj','f_p_high_adj','f_p_low_adj','f_p_close_adj','f_p_pre_close_adj',
                           'f_p_adj_factor','f_p_avg','f_p_limit_up','f_p_limit_down','f_s_trade_status',
                           'f_i_mv_total','f_i_mv_circ','f_i_shares_total','f_i_shares_float','f_i_shares_free',
                           'f_p_high_52w','f_p_low_52w','f_p_high_52w_adj','f_p_low_52w_adj',
                           'f_i_pe','f_i_pb','f_i_pe_ttm','f_i_pcf_ocf','f_i_pcf_ocf_ttm','f_i_pcf_ncf','f_i_pcf_ncf_ttm','f_i_ps','f_i_ps_ttm',
                           'f_i_turnover','f_i_turnover_free','f_i_price_div_dps','f_i_net_profit_ttm','f_i_net_profit_lyr',
                           'f_i_net_assets','f_i_cash_flow_oper_ttm','f_i_cash_flow_oper_lyr','f_i_oper_rev_ttm','f_i_oper_rev_lyr',
                           'f_i_cash_flow_total_ttm','f_i_cash_flow_total_lyr','f_s_limit_status','f_i_swing','f_v_avg_3M','f_v_trades_total',
                           'f_p_call_open','f_v_call_open_volume','f_v_call_open_trades','f_p_call_close','f_v_call_close_volume','f_v_call_close_trades',
                           'f_v_buy_exlarge_amount','f_v_sell_exlarge_amount','f_v_buy_large_amount','f_v_sell_large_amount',
                           'f_v_buy_med_amount','f_v_sell_med_amount','f_v_buy_small_amount','f_v_sell_small_amount',
                           'f_v_buy_exlarge_volume','f_v_sell_exlarge_volume','f_v_buy_large_volume','f_v_sell_large_volume',
                           'f_v_buy_med_volume','f_v_sell_med_volume','f_v_buy_small_volume','f_v_sell_small_volume','f_v_trades',
                           'f_v_buy_trades_exlarge','f_v_sell_trades_exlarge','f_v_buy_trades_large','f_v_sell_trades_large','f_v_buy_trades_med','f_v_sell_trades_med',
                           'f_v_buy_trades_small','f_v_sell_trades_small','f_v_diff_small_volume','f_v_diff_small_volume_act','f_v_diff_med_volume','f_v_diff_med_volume_act',
                           'f_v_diff_large_volume','f_v_diff_large_volume_act','f_v_diff_institute_volume','f_v_diff_institute_volume_act','f_v_diff_small_amount','f_v_diff_small_amount_act',
                           'f_v_diff_med_amount','f_v_diff_med_amount_act','f_v_diff_large_amount','f_v_diff_large_amount_act','f_v_diff_institute_amount','f_v_diff_institute_amount_act',
                           'f_v_net_inflow_volume','f_v_net_inflow_rate_volume','f_v_inflow_open_volume','f_v_inflow_open_rate_volume','f_v_inflow_close_volume','f_v_inflow_close_rate_volume',
                           'f_v_net_inflow','f_v_net_inflow_rate','f_v_inflow_open','f_v_inflow_open_rate','f_v_inflow_close','f_v_inflow_close_rate',
                           'f_i_mf_pct_volume','f_i_mf_pct_open_volume','f_i_mf_pct_close_volume','f_i_mf_pct_value','f_i_mf_pct_open_value','f_i_mf_pct_close_value',
                           'f_v_inflow_large_volume','f_v_inflow_large_rate_volume','f_v_inflow_large','f_v_inflow_large_rate','f_i_mf_pct_volume_large','f_i_mf_pct_value_large',
                           'f_v_inflow_large_open_volume','f_v_inflow_large_open_rate_volume','f_v_inflow_large_open','f_v_inflow_large_open_rate','f_i_mf_pct_large_open_volume','f_i_mf_pct_large_open_value',
                           'f_v_inflow_large_close_volume','f_v_inflow_large_close_rate_volume','f_v_inflow_large_close','f_v_inflow_large_close_rate','f_i_mf_pct_large_close_volume','f_i_mf_pct_large_close_value',
                           'f_v_buy_exlarge_amount_act','f_v_sell_exlarge_amount_act','f_v_buy_large_amount_act','f_v_sell_large_amount_act','f_v_buy_med_amount_act','f_v_sell_med_amount_act',
                           'f_v_buy_small_amount_act','f_v_sell_small_amount_act','f_v_buy_exlarge_volume_act','f_v_sell_exlarge_volume_act','f_v_buy_large_volume_act','f_v_sell_large_volume_act','f_v_buy_med_volume_act','f_v_sell_med_volume_act','f_v_buy_small_volume_act','f_v_sell_small_volume_act','f_i_boll_mid',
                           'f_i_boll_upper','f_i_boll_lower','f_i_bbiboll_bbi','f_i_bbiboll_upr','f_i_bbiboll_dwn','f_i_cdp','f_i_cdp_ah','f_i_cdp_al','f_i_cdp_nh','f_i_cdp_nl','f_i_env_upper','f_i_env_lower',
                           'f_i_mike_wr','f_i_mike_mr','f_i_mike_sr','f_i_mike_ws','f_i_mike_ms','f_i_mike_ss','f_i_obv','f_i_obv_obv','f_i_pvt','f_i_wvad_wvad','f_i_wvad_mawvad',
                           'f_i_arbr_ar','f_i_arbr_br','f_i_cr','f_i_psy','f_i_psyma','f_i_wad','f_i_mawad','f_i_market','f_i_strength','f_i_weakness','f_i_bottoming_b','f_i_bottoming_d',
                           'f_i_dmi_pdi','f_i_dmi_mdi','f_i_dmi_adx','f_i_dmi_adxr','f_i_expma','f_i_ma_5','f_i_ma_10','f_i_ma_20','f_i_ma_30','f_i_ma_60','f_i_ma_120','f_i_ma_250',
                           'f_i_macd_diff','f_i_macd_dea','f_i_macd','f_i_bbi','f_i_dma_ddd','f_i_dma_ama','f_i_mtm','f_i_mtm_mtmma','f_i_priceosc','f_i_trix','f_i_trma',
                           'f_i_sar','f_i_buy_rate','f_i_buy_amount','f_i_buy_volume','f_i_sell_rate','f_i_sell_amount','f_i_sell_volume','f_i_large_buy_rate','f_i_large_buy_amount','f_i_large_buy_volume','f_i_large_sell_rate','f_i_large_sell_amount','f_i_large_sell_volume',
                           'f_i_margin_balance','f_i_margin_buy','f_i_margin_repay','f_i_seclending_balance','f_i_seclending_volume','f_i_seclending_sell','f_i_seclending_repay','f_i_margin_total','f_i_margin_balance_volume','f_i_seclending_sell_amount','f_i_seclending_repay_amount',
                           'f_i_rc_50d','f_i_mi_a12d','f_i_mi_mi12d','f_i_srmi_9d','f_i_atr_tr14d','f_i_atr','f_i_mass','f_i_vhf','f_i_cvlt','f_i_adtm','f_i_adtm_adtmma',
                           'f_i_bias','f_i_kdj_k','f_i_kdj_d','f_i_kdj_j','f_i_rsi_6','f_i_cci','f_i_dpo','f_i_dpo_madpo','f_i_roc','f_i_roc_rocma','f_i_si','f_i_slowkd_k','f_i_slowkd_d','f_i_wr',
                           'f_i_bias_36','f_i_bias_612','f_i_volume_ratio','f_i_vma_1M','f_i_vma_5d','f_i_vma_22d','f_i_vma_60d','f_i_vmacd','f_i_vmacd_dea','f_i_vmacd_macd',
                           'f_i_vosc','f_i_tapi_16d','f_i_tapi_6d','f_i_vstd_10d','f_i_vrsi_6d','f_i_vroc_12d','f_i_sobv','f_i_vr_26d',
                           'f_i_xin9','f_i_hsi','f_i_hkah','f_i_dji','f_i_spx','f_i_ixic',
                           'f_p_open5min_close','f_p_open5min_avg','f_p_open1h_avg','f_p_avg_1d',
                           'f_i_news_micro','f_i_news_macro',
                           'f_s_is_suspend','f_s_is_st',
                           'f_w_hs300','f_w_csi500','f_w_csi1000','f_w_gz2000','f_w_csi2000','f_w_csi800','f_w_csi_all',
                           'f_s_limit_mark','f_s_padding_flag'
                           ]  # 311 channels
    args.open_price_index = CM.find_str_in_list(args.src_field_list, 'f_p_open')  # '开盘价的列索引'
    args.close_price_index = CM.find_str_in_list(args.src_field_list, 'f_p_close')  # '收盘价的列索引'
    args.priceRelatedField = ['f_p_open', 'f_p_high', 'f_p_low', 'f_p_close', 'f_p_pre_close', 'f_p_avg',
                              'f_p_open_adj', 'f_p_high_adj', 'f_p_low_adj', 'f_p_close_adj', 'f_p_pre_close_adj',
                              'f_p_limit_up', 'f_p_limit_down', 'f_p_high_52w', 'f_p_low_52w', 'f_p_high_52w_adj', 'f_p_low_52w_adj',
                              'f_p_call_open', 'f_p_call_close',
                              'f_p_open5min_close', 'f_p_open5min_avg', 'f_p_open1h_avg', 'f_p_avg_1d']
    args.mask_channel = ['f_d_code_int','f_d_code_id','f_i_news_micro','f_i_news_macro','f_w_hs300','f_w_csi500','f_w_csi1000','f_w_gz2000','f_w_csi2000','f_w_csi800','f_w_csi_all']

    if args.price_target == '935':  # 9:35分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open5min_close')
    elif args.price_target == '5m':  # 前5分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open5min_avg')
    elif args.price_target == '60m':  # 前60分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open1h_avg')
    elif args.price_target == 'day':  # 全天均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_avg')
    else:  # 默认使用开盘价
        args.price_target_index = args.open_price_index


    args.sm_stock_code = None  # 资产组合中的股票代码
    args.shuffle = True  # 是否shuffle数据集
    args.fast_running_its = 0  # 设置为0是表示正常训练，设置大于0时，则表示快速测试多少个batch/epoch, 用于快速跑通代码流程
    args.useGPU = False
    args.device = 'cpu'
    args.multi_device = False
    args.device_list = []
    if torch.cuda.is_available():
        args.useGPU = True
        args.device_list = CM.make_device_list(args.GPU)
        args.multi_device = len(args.device_list) > 1
        args.device = 'cuda:' + str(args.device_list[0])
        if args.model.startswith('PP_') and args.RUN_MODE == 'check':
            args.device_list = [0]
            args.multi_device = False
            args.device = 'cuda:0'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.mark == '':
        args.mark = f"{args.model}_{CM.timestr(1)}"
    if args.RUN_MODE == 'train':
        CM.save_base_args(args)

    return args


# def try_load_mark_args(args, load_model_file):
#     model_base_args = ["model", "time_step", "d_model",
#                        "price_target", "DS", "cross_stock_attn",
#                        "global_stock_space", "firstWeight", "ci",
#                        "layers"]
#     if load_model_file != '':
#         # 加载模型基础参数
#         base_args = CM.load_base_args(load_model_file)
#         if base_args is not None:
#             for base_arg_name in base_args.keys():
#                 if base_arg_name in model_base_args:
#                     setattr(args, base_arg_name, base_args[base_arg_name])
#     return args


def adjust_learning_rate_dynamic(optimizer, args, printout=True, rank=0, g_single_gpu=0):
    """
    动态调整学习率，根据当前epoch调整学习率,按指定衰减率衰减
    :param optimizer: 优化器
    :param args: 参数
    :param printout: 是否打印信息
    :param rank: 进程id
    :param g_single_gpu: 单卡训练时
    """

    current_lr = optimizer.param_groups[0]['lr']
    new_lr = current_lr * args.lr_decay  # 将学习率调整为原来的75%
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    if printout and rank == g_single_gpu:
        print(f'{CM.timestr()}Updating learning rate to {new_lr:.7f}')
