import copy
import json
import multiprocessing
import os
import time
import pandas as pd
import numpy as np
import ast
import common_utils as CM
import torch
import utils_date as DT

g_csv_path = ""  # 过去80天数据读取路径
g_scaler_info = ""  # 数据集对应的字段字典表文件路径
g_stock_info_path = f"{CM.G_ROOT_PATH}/raw_generated_data/tushare_data/cfg/stock_info.json"  # 股票信息文件路径
g_xt_open_price_path = f"{CM.G_ROOT_PATH}/raw_generated_data/realtime_data/xt_open_price"  # 讯投的开盘价数据路径
g_local_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件路径
g_llm_bad_info_path = f"{CM.G_ROOT_PATH}/raw_generated_data/realtime_data/exclude_stock/"
g_quit_info_path = f"{CM.G_ROOT_PATH}/raw_generated_data/delist_stock_date_mapping.json"
g_open_t1_path = g_xt_open_price_path  # 开盘价数据路径
g_stock_pool = []

g_source_field = {
    'wd311': ['f_d_code_int', 'f_d_code_id', 'f_d_exchange_id', 'f_d_market_id', 'f_d_industry_id', 'f_d_year',
              'f_d_month',
              'f_d_day', 'f_d_week',
              'gen_cq', 'f_p_open', 'f_p_high', 'f_p_low', 'f_p_close', 'f_p_pre_close', 'f_i_change', 'f_i_pct_chg',
              'f_v_volume', 'f_v_amount', 'f_p_open_adj', 'f_p_high_adj', 'f_p_low_adj', 'f_p_close_adj',
              'f_p_pre_close_adj',
              'f_p_adj_factor', 'f_p_avg', 'f_p_limit_up', 'f_p_limit_down', 'f_s_trade_status',
              'f_i_mv_total', 'f_i_mv_circ', 'f_i_shares_total', 'f_i_shares_float', 'f_i_shares_free',
              'f_p_high_52w', 'f_p_low_52w', 'f_p_high_52w_adj', 'f_p_low_52w_adj',
              'f_i_pe', 'f_i_pb', 'f_i_pe_ttm', 'f_i_pcf_ocf', 'f_i_pcf_ocf_ttm', 'f_i_pcf_ncf', 'f_i_pcf_ncf_ttm',
              'f_i_ps', 'f_i_ps_ttm',
              'f_i_turnover', 'f_i_turnover_free', 'f_i_price_div_dps', 'f_i_net_profit_ttm', 'f_i_net_profit_lyr',
              'f_i_net_assets', 'f_i_cash_flow_oper_ttm', 'f_i_cash_flow_oper_lyr', 'f_i_oper_rev_ttm',
              'f_i_oper_rev_lyr',
              'f_i_cash_flow_total_ttm', 'f_i_cash_flow_total_lyr', 'f_s_limit_status', 'f_i_swing', 'f_v_avg_3M',
              'f_v_trades_total',
              'f_p_call_open', 'f_v_call_open_volume', 'f_v_call_open_trades', 'f_p_call_close',
              'f_v_call_close_volume',
              'f_v_call_close_trades',
              'f_v_buy_exlarge_amount', 'f_v_sell_exlarge_amount', 'f_v_buy_large_amount', 'f_v_sell_large_amount',
              'f_v_buy_med_amount', 'f_v_sell_med_amount', 'f_v_buy_small_amount', 'f_v_sell_small_amount',
              'f_v_buy_exlarge_volume', 'f_v_sell_exlarge_volume', 'f_v_buy_large_volume', 'f_v_sell_large_volume',
              'f_v_buy_med_volume', 'f_v_sell_med_volume', 'f_v_buy_small_volume', 'f_v_sell_small_volume',
              'f_v_trades',
              'f_v_buy_trades_exlarge', 'f_v_sell_trades_exlarge', 'f_v_buy_trades_large', 'f_v_sell_trades_large',
              'f_v_buy_trades_med', 'f_v_sell_trades_med',
              'f_v_buy_trades_small', 'f_v_sell_trades_small', 'f_v_diff_small_volume', 'f_v_diff_small_volume_act',
              'f_v_diff_med_volume', 'f_v_diff_med_volume_act',
              'f_v_diff_large_volume', 'f_v_diff_large_volume_act', 'f_v_diff_institute_volume',
              'f_v_diff_institute_volume_act', 'f_v_diff_small_amount', 'f_v_diff_small_amount_act',
              'f_v_diff_med_amount', 'f_v_diff_med_amount_act', 'f_v_diff_large_amount', 'f_v_diff_large_amount_act',
              'f_v_diff_institute_amount', 'f_v_diff_institute_amount_act',
              'f_v_net_inflow_volume', 'f_v_net_inflow_rate_volume', 'f_v_inflow_open_volume',
              'f_v_inflow_open_rate_volume', 'f_v_inflow_close_volume', 'f_v_inflow_close_rate_volume',
              'f_v_net_inflow', 'f_v_net_inflow_rate', 'f_v_inflow_open', 'f_v_inflow_open_rate', 'f_v_inflow_close',
              'f_v_inflow_close_rate',
              'f_i_mf_pct_volume', 'f_i_mf_pct_open_volume', 'f_i_mf_pct_close_volume', 'f_i_mf_pct_value',
              'f_i_mf_pct_open_value', 'f_i_mf_pct_close_value',
              'f_v_inflow_large_volume', 'f_v_inflow_large_rate_volume', 'f_v_inflow_large', 'f_v_inflow_large_rate',
              'f_i_mf_pct_volume_large', 'f_i_mf_pct_value_large',
              'f_v_inflow_large_open_volume', 'f_v_inflow_large_open_rate_volume', 'f_v_inflow_large_open',
              'f_v_inflow_large_open_rate', 'f_i_mf_pct_large_open_volume', 'f_i_mf_pct_large_open_value',
              'f_v_inflow_large_close_volume', 'f_v_inflow_large_close_rate_volume', 'f_v_inflow_large_close',
              'f_v_inflow_large_close_rate', 'f_i_mf_pct_large_close_volume', 'f_i_mf_pct_large_close_value',
              'f_v_buy_exlarge_amount_act', 'f_v_sell_exlarge_amount_act', 'f_v_buy_large_amount_act',
              'f_v_sell_large_amount_act', 'f_v_buy_med_amount_act', 'f_v_sell_med_amount_act',
              'f_v_buy_small_amount_act', 'f_v_sell_small_amount_act', 'f_v_buy_exlarge_volume_act',
              'f_v_sell_exlarge_volume_act', 'f_v_buy_large_volume_act', 'f_v_sell_large_volume_act',
              'f_v_buy_med_volume_act', 'f_v_sell_med_volume_act', 'f_v_buy_small_volume_act',
              'f_v_sell_small_volume_act',
              'f_i_boll_mid',
              'f_i_boll_upper', 'f_i_boll_lower', 'f_i_bbiboll_bbi', 'f_i_bbiboll_upr', 'f_i_bbiboll_dwn', 'f_i_cdp',
              'f_i_cdp_ah', 'f_i_cdp_al', 'f_i_cdp_nh', 'f_i_cdp_nl', 'f_i_env_upper', 'f_i_env_lower',
              'f_i_mike_wr', 'f_i_mike_mr', 'f_i_mike_sr', 'f_i_mike_ws', 'f_i_mike_ms', 'f_i_mike_ss', 'f_i_obv',
              'f_i_obv_obv', 'f_i_pvt', 'f_i_wvad_wvad', 'f_i_wvad_mawvad',
              'f_i_arbr_ar', 'f_i_arbr_br', 'f_i_cr', 'f_i_psy', 'f_i_psyma', 'f_i_wad', 'f_i_mawad', 'f_i_market',
              'f_i_strength', 'f_i_weakness', 'f_i_bottoming_b', 'f_i_bottoming_d',
              'f_i_dmi_pdi', 'f_i_dmi_mdi', 'f_i_dmi_adx', 'f_i_dmi_adxr', 'f_i_expma', 'f_i_ma_5', 'f_i_ma_10',
              'f_i_ma_20', 'f_i_ma_30', 'f_i_ma_60', 'f_i_ma_120', 'f_i_ma_250',
              'f_i_macd_diff', 'f_i_macd_dea', 'f_i_macd', 'f_i_bbi', 'f_i_dma_ddd', 'f_i_dma_ama', 'f_i_mtm',
              'f_i_mtm_mtmma', 'f_i_priceosc', 'f_i_trix', 'f_i_trma',
              'f_i_sar', 'f_i_buy_rate', 'f_i_buy_amount', 'f_i_buy_volume', 'f_i_sell_rate', 'f_i_sell_amount',
              'f_i_sell_volume', 'f_i_large_buy_rate', 'f_i_large_buy_amount', 'f_i_large_buy_volume',
              'f_i_large_sell_rate', 'f_i_large_sell_amount', 'f_i_large_sell_volume',
              'f_i_margin_balance', 'f_i_margin_buy', 'f_i_margin_repay', 'f_i_seclending_balance',
              'f_i_seclending_volume',
              'f_i_seclending_sell', 'f_i_seclending_repay', 'f_i_margin_total', 'f_i_margin_balance_volume',
              'f_i_seclending_sell_amount', 'f_i_seclending_repay_amount',
              'f_i_rc_50d', 'f_i_mi_a12d', 'f_i_mi_mi12d', 'f_i_srmi_9d', 'f_i_atr_tr14d', 'f_i_atr', 'f_i_mass',
              'f_i_vhf',
              'f_i_cvlt', 'f_i_adtm', 'f_i_adtm_adtmma',
              'f_i_bias', 'f_i_kdj_k', 'f_i_kdj_d', 'f_i_kdj_j', 'f_i_rsi_6', 'f_i_cci', 'f_i_dpo', 'f_i_dpo_madpo',
              'f_i_roc', 'f_i_roc_rocma', 'f_i_si', 'f_i_slowkd_k', 'f_i_slowkd_d', 'f_i_wr',
              'f_i_bias_36', 'f_i_bias_612', 'f_i_volume_ratio', 'f_i_vma_1M', 'f_i_vma_5d', 'f_i_vma_22d',
              'f_i_vma_60d',
              'f_i_vmacd', 'f_i_vmacd_dea', 'f_i_vmacd_macd',
              'f_i_vosc', 'f_i_tapi_16d', 'f_i_tapi_6d', 'f_i_vstd_10d', 'f_i_vrsi_6d', 'f_i_vroc_12d', 'f_i_sobv',
              'f_i_vr_26d',
              'f_i_xin9', 'f_i_hsi', 'f_i_hkah', 'f_i_dji', 'f_i_spx', 'f_i_ixic',
              'f_p_open5min_close', 'f_p_open5min_avg', 'f_p_open1h_avg', 'f_p_avg_1d',
              'f_i_news_micro', 'f_i_news_macro',
              'is_suspend', 'st',
              'f_w_hs300', 'f_w_csi500', 'f_w_csi1000', 'f_w_gz2000', 'f_w_csi2000', 'f_w_csi800', 'f_w_csi_all',
              'limit_mark', 'gen_padding_flag'
              ],
    'wd395': ["f_d_code_int", "f_d_code_id", "f_d_exchange_id", "f_d_market_id", "f_d_industry_id", "f_d_year",
                                "f_d_month", "f_d_day", "f_d_week", "gen_cq", "f_p_open", "f_p_high", "f_p_low", "f_p_close",
                                "f_p_pre_close", "f_i_change", "f_i_pct_chg", "f_v_volume", "f_v_amount", "f_p_open_adj", "f_p_high_adj",
                                "f_p_low_adj", "f_p_close_adj", "f_p_pre_close_adj", "f_p_adj_factor", "f_p_avg", "f_p_limit_up",
                                "f_p_limit_down", "f_s_trade_status", "f_i_mv_total", "f_i_mv_circ", "f_i_shares_total", "f_i_shares_float",
                                "f_i_shares_free", "f_p_high_52w", "f_p_low_52w", "f_p_high_52w_adj", "f_p_low_52w_adj", "f_i_pe", "f_i_pb",
                                "f_i_pe_ttm", "f_i_pcf_ocf", "f_i_pcf_ocf_ttm", "f_i_pcf_ncf", "f_i_pcf_ncf_ttm", "f_i_ps", "f_i_ps_ttm",
                                "f_i_turnover", "f_i_turnover_free", "f_i_price_div_dps", "f_i_net_profit_ttm", "f_i_net_profit_lyr",
                                "f_i_net_assets", "f_i_cash_flow_oper_ttm", "f_i_cash_flow_oper_lyr", "f_i_oper_rev_ttm", "f_i_oper_rev_lyr",
                                "f_i_cash_flow_total_ttm", "f_i_cash_flow_total_lyr", "f_s_limit_status", "f_i_swing", "f_v_avg_3M",
                                "f_v_trades_total", "f_p_call_open", "f_v_call_open_volume", "f_v_call_open_trades", "f_p_call_close",
                                "f_v_call_close_volume", "f_v_call_close_trades", "f_v_buy_exlarge_amount", "f_v_sell_exlarge_amount",
                                "f_v_buy_large_amount", "f_v_sell_large_amount", "f_v_buy_med_amount", "f_v_sell_med_amount",
                                "f_v_buy_small_amount", "f_v_sell_small_amount", "f_v_buy_exlarge_volume", "f_v_sell_exlarge_volume",
                                "f_v_buy_large_volume", "f_v_sell_large_volume", "f_v_buy_med_volume", "f_v_sell_med_volume",
                                "f_v_buy_small_volume", "f_v_sell_small_volume", "f_v_trades", "f_v_buy_trades_exlarge",
                                "f_v_sell_trades_exlarge", "f_v_buy_trades_large", "f_v_sell_trades_large", "f_v_buy_trades_med",
                                "f_v_sell_trades_med", "f_v_buy_trades_small", "f_v_sell_trades_small", "f_v_diff_small_volume",
                                "f_v_diff_small_volume_act", "f_v_diff_med_volume", "f_v_diff_med_volume_act", "f_v_diff_large_volume",
                                "f_v_diff_large_volume_act", "f_v_diff_institute_volume", "f_v_diff_institute_volume_act",
                                "f_v_diff_small_amount", "f_v_diff_small_amount_act", "f_v_diff_med_amount", "f_v_diff_med_amount_act",
                                "f_v_diff_large_amount", "f_v_diff_large_amount_act", "f_v_diff_institute_amount",
                                "f_v_diff_institute_amount_act", "f_v_net_inflow_volume", "f_v_net_inflow_rate_volume", "f_v_inflow_open_volume",
                                "f_v_inflow_open_rate_volume", "f_v_inflow_close_volume", "f_v_inflow_close_rate_volume", "f_v_net_inflow",
                                "f_v_net_inflow_rate", "f_v_inflow_open", "f_v_inflow_open_rate", "f_v_inflow_close", "f_v_inflow_close_rate",
                                "f_i_mf_pct_volume", "f_i_mf_pct_open_volume", "f_i_mf_pct_close_volume", "f_i_mf_pct_value",
                                "f_i_mf_pct_open_value", "f_i_mf_pct_close_value", "f_v_inflow_large_volume", "f_v_inflow_large_rate_volume",
                                "f_v_inflow_large", "f_v_inflow_large_rate", "f_i_mf_pct_volume_large", "f_i_mf_pct_value_large",
                                "f_v_inflow_large_open_volume", "f_v_inflow_large_open_rate_volume", "f_v_inflow_large_open",
                                "f_v_inflow_large_open_rate", "f_i_mf_pct_large_open_volume", "f_i_mf_pct_large_open_value",
                                "f_v_inflow_large_close_volume", "f_v_inflow_large_close_rate_volume", "f_v_inflow_large_close",
                                "f_v_inflow_large_close_rate", "f_i_mf_pct_large_close_volume", "f_i_mf_pct_large_close_value",
                                "f_v_buy_exlarge_amount_act", "f_v_sell_exlarge_amount_act", "f_v_buy_large_amount_act",
                                "f_v_sell_large_amount_act", "f_v_buy_med_amount_act", "f_v_sell_med_amount_act", "f_v_buy_small_amount_act",
                                "f_v_sell_small_amount_act", "f_v_buy_exlarge_volume_act", "f_v_sell_exlarge_volume_act",
                                "f_v_buy_large_volume_act", "f_v_sell_large_volume_act", "f_v_buy_med_volume_act", "f_v_sell_med_volume_act",
                                "f_v_buy_small_volume_act", "f_v_sell_small_volume_act", "f_i_boll_mid", "f_i_boll_upper", "f_i_boll_lower",
                                "f_i_bbiboll_bbi", "f_i_bbiboll_upr", "f_i_bbiboll_dwn", "f_i_cdp", "f_i_cdp_ah", "f_i_cdp_al", "f_i_cdp_nh",
                                "f_i_cdp_nl", "f_i_env_upper", "f_i_env_lower", "f_i_mike_wr", "f_i_mike_mr", "f_i_mike_sr", "f_i_mike_ws",
                                "f_i_mike_ms", "f_i_mike_ss", "f_i_obv", "f_i_obv_obv", "f_i_pvt", "f_i_wvad_wvad", "f_i_wvad_mawvad",
                                "f_i_arbr_ar", "f_i_arbr_br", "f_i_cr", "f_i_psy", "f_i_psyma", "f_i_wad", "f_i_mawad", "f_i_market",
                                "f_i_strength", "f_i_weakness", "f_i_bottoming_b", "f_i_bottoming_d", "f_i_dmi_pdi", "f_i_dmi_mdi",
                                "f_i_dmi_adx", "f_i_dmi_adxr", "f_i_expma", "f_i_ma_5", "f_i_ma_10", "f_i_ma_20", "f_i_ma_30", "f_i_ma_60",
                                "f_i_ma_120", "f_i_ma_250", "f_i_macd_diff", "f_i_macd_dea", "f_i_macd", "f_i_bbi", "f_i_dma_ddd", "f_i_dma_ama",
                                "f_i_mtm", "f_i_mtm_mtmma", "f_i_priceosc", "f_i_trix", "f_i_trma", "f_i_sar", "f_i_buy_rate", "f_i_buy_amount",
                                "f_i_buy_volume", "f_i_sell_rate", "f_i_sell_amount", "f_i_sell_volume", "f_i_large_buy_rate",
                                "f_i_large_buy_amount", "f_i_large_buy_volume", "f_i_large_sell_rate", "f_i_large_sell_amount",
                                "f_i_large_sell_volume", "f_i_margin_balance", "f_i_margin_buy", "f_i_margin_repay", "f_i_seclending_balance",
                                "f_i_seclending_volume", "f_i_seclending_sell", "f_i_seclending_repay", "f_i_margin_total",
                                "f_i_margin_balance_volume", "f_i_seclending_sell_amount", "f_i_seclending_repay_amount", "f_i_rc_50d",
                                "f_i_mi_a12d", "f_i_mi_mi12d", "f_i_srmi_9d", "f_i_atr_tr14d", "f_i_atr", "f_i_mass", "f_i_vhf", "f_i_cvlt",
                                "f_i_adtm", "f_i_adtm_adtmma", "f_i_bias", "f_i_kdj_k", "f_i_kdj_d", "f_i_kdj_j", "f_i_rsi_6", "f_i_cci",
                                "f_i_dpo", "f_i_dpo_madpo", "f_i_roc", "f_i_roc_rocma", "f_i_si", "f_i_slowkd_k", "f_i_slowkd_d", "f_i_wr",
                                "f_i_bias_36", "f_i_bias_612", "f_i_volume_ratio", "f_i_vma_1M", "f_i_vma_5d", "f_i_vma_22d", "f_i_vma_60d",
                                "f_i_vmacd", "f_i_vmacd_dea", "f_i_vmacd_macd", "f_i_vosc", "f_i_tapi_16d", "f_i_tapi_6d", "f_i_vstd_10d",
                                "f_i_vrsi_6d", "f_i_vroc_12d", "f_i_sobv", "f_i_vr_26d", "f_i_xin9", "f_i_hsi", "f_i_hkah", "f_i_dji", "f_i_spx",
                                "f_i_ixic", "f_p_open5min_close", "f_p_open5min_avg", "f_p_open1h_avg", "f_p_avg_1d", "f_i_news_micro",
                                "f_i_news_macro", "is_suspend", "st", "f_w_hs300", "f_w_csi500", "f_w_csi1000", "f_w_gz2000",
                                "f_w_csi2000", "f_w_csi800", "f_w_csi_all", 'limit_mark', 'gen_padding_flag', "f_i_exp_size",
                                "f_i_exp_non_linear_size", "f_i_exp_momentum", "f_i_exp_liquidity", "f_i_exp_book_to_price", "f_i_exp_leverage",
                                "f_i_exp_growth", "f_i_exp_earnings_yield", "f_i_exp_beta", "f_i_exp_residual_volatility", "f_i_exp_comovement",
                                "f_i_exp_gangtie", "f_i_exp_feiyinjinrong", "f_i_exp_qiche", "f_i_exp_jiaotongyunshu", "f_i_exp_jianzhucailiao",
                                "f_i_exp_shipinyinliao", "f_i_exp_gongyongshiye", "f_i_exp_yousejinshu", "f_i_exp_zonghe", "f_i_exp_tongxin",
                                "f_i_exp_fangdichan", "f_i_exp_fangzhifushi", "f_i_exp_jixieshebei", "f_i_exp_qinggongzhizao",
                                "f_i_exp_shangmaolingshou", "f_i_exp_nonglinmuyu", "f_i_exp_chuanmei", "f_i_exp_huanbao", "f_i_exp_jichuhuagong",
                                "f_i_exp_dianlishebei", "f_i_exp_dianzi", "f_i_exp_guofangjungong", "f_i_exp_yinhang", "f_i_exp_shiyoushihua",
                                "f_i_exp_jisuanji", "f_i_exp_jianzhuzhuangshi", "f_i_exp_shehuifuwu", "f_i_exp_meironghuli",
                                "f_i_exp_jiayongdianqi", "f_i_exp_yiyaoshengwu", "f_i_exp_meitan", "f_i_ret_size", "f_i_ret_non_linear_size",
                                "f_i_ret_momentum", "f_i_ret_liquidity", "f_i_ret_book_to_price", "f_i_ret_leverage", "f_i_ret_growth",
                                "f_i_ret_earnings_yield", "f_i_ret_beta", "f_i_ret_residual_volatility", "f_i_ret_comovement", "f_i_ret_gangtie",
                                "f_i_ret_feiyinjinrong", "f_i_ret_qiche", "f_i_ret_jiaotongyunshu", "f_i_ret_jianzhucailiao",
                                "f_i_ret_shipinyinliao", "f_i_ret_gongyongshiye", "f_i_ret_yousejinshu", "f_i_ret_zonghe", "f_i_ret_tongxin",
                                "f_i_ret_fangdichan", "f_i_ret_fangzhifushi", "f_i_ret_jixieshebei", "f_i_ret_qinggongzhizao",
                                "f_i_ret_shangmaolingshou", "f_i_ret_nonglinmuyu", "f_i_ret_chuanmei", "f_i_ret_huanbao", "f_i_ret_jichuhuagong",
                                "f_i_ret_dianlishebei", "f_i_ret_dianzi", "f_i_ret_guofangjungong", "f_i_ret_yinhang", "f_i_ret_shiyoushihua",
                                "f_i_ret_jisuanji", "f_i_ret_jianzhuzhuangshi", "f_i_ret_shehuifuwu", "f_i_ret_meironghuli",
                                "f_i_ret_jiayongdianqi", "f_i_ret_yiyaoshengwu", "f_i_ret_meitan"
              ],
    'day1062': ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
                'gen_year', 'gen_month', 'gen_day', 'gen_week', 'gen_cq',
                'open', 'high', 'low', 'close', 'pre_close',
                'change', 'pct_chg', 'vol', 'amount', 'turnover_rate',
                'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb',
                'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share',
                'float_share', 'free_share', 'total_mv', 'circ_mv', 'buy_sm_vol',
                'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount',
                'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol',
                'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
                'net_mf_vol', 'net_mf_amount', 'up_limit', 'down_limit', 'call_auction_close',
                'call_auction_open', 'call_auction_high', 'call_auction_low', 'call_auction_vol',
                'call_auction_amount',
                'call_auction_vwap', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
                'adj_factor', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 'kdj_bfq',
                'kdj_d_bfq', 'kdj_k_bfq', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250',
                'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq',
                'macd_dea_bfq', 'macd_dif_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq',
                'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 'wr_bfq', 'wr1_bfq',
                'XIN9', 'HSI', 'HKAH', 'DJI', 'SPX',
                'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'news_micro_score',
                'news_macro_score', 'is_suspend', 'st',
                'limit_mark', 'gen_padding_flag'],
    'day134': ['ts_code', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry', 'gen_year', 'gen_month',
               'gen_day',
               'gen_week', 'gen_cq', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'volume',
               'amount',
               'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
               'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'buy_sm_vol',
               'buy_sm_amount',
               'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
               'buy_lg_vol',
               'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol',
               'sell_elg_amount', 'net_mf_vol', 'net_mf_amount', 'up_limit', 'down_limit', 'call_auction_close',
               'call_auction_open', 'call_auction_high', 'call_auction_low', 'call_auction_vol', 'call_auction_amount',
               'call_auction_vwap', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'adj_factor', 'boll_lower_bfq',
               'boll_mid_bfq', 'boll_upper_bfq', 'kdj_bfq', 'kdj_d_bfq', 'kdj_k_bfq', 'ma_bfq_10', 'ma_bfq_20',
               'ma_bfq_250',
               'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq', 'macd_dea_bfq', 'macd_dif_bfq', 'mtm_bfq',
               'mtmma_bfq', 'obv_bfq', 'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 'wr_bfq', 'wr1_bfq', 'XIN9', 'HSI',
               'HKAH',
               'DJI', 'SPX', 'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'news_micro_score',
               'news_macro_score', 'hs300_weight', 'zz500_weight', 'zz1000_weight', 'gz2000_weight', 'zz2000_weight',
               'zz800_weight', 'exp_size', 'exp_non_linear_size', 'exp_momentum', 'exp_liquidity', 'exp_book_to_price',
               'exp_leverage', 'exp_growth', 'exp_earnings_yield', 'exp_beta', 'exp_residual_volatility',
               'exp_comovement',
               'ret_size', 'ret_non_linear_size', 'ret_momentum', 'ret_liquidity', 'ret_book_to_price', 'ret_leverage',
               'ret_growth', 'ret_earnings_yield', 'ret_beta', 'ret_residual_volatility', 'ret_comovement',
               'is_suspend',
               'st', 'limit_mark', 'gen_padding_flag'],
    'day196': ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
               'gen_year', 'gen_month', 'gen_day', 'gen_week', 'gen_cq',
               'open', 'high', 'low', 'close', 'pre_close',
               'change', 'pct_chg', 'vol', 'amount', 'turnover_rate',
               'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb',
               'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share',
               'float_share', 'free_share', 'total_mv', 'circ_mv', 'buy_sm_vol',
               'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount',
               'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol',
               'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
               'net_mf_vol', 'net_mf_amount', 'up_limit', 'down_limit', 'call_auction_close',
               'call_auction_open', 'call_auction_high', 'call_auction_low', 'call_auction_vol',
               'call_auction_amount',
               'call_auction_vwap', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq',
               'adj_factor', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 'kdj_bfq',
               'kdj_d_bfq', 'kdj_k_bfq', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250',
               'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq',
               'macd_dea_bfq', 'macd_dif_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq',
               'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 'wr_bfq', 'wr1_bfq',
               'XIN9', 'HSI', 'HKAH', 'DJI', 'SPX',
               'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'news_micro_score',
               'news_macro_score', "hs300_weight", "zz500_weight", "zz1000_weight", "gz2000_weight", "zz2000_weight",
               "zz800_weight", "exp_size", "exp_non_linear_size", "exp_momentum", "exp_liquidity", "exp_book_to_price",
               "exp_leverage", "exp_growth", "exp_earnings_yield", "exp_beta", "exp_residual_volatility",
               "exp_comovement", 'exp_gangtie', 'exp_feiyinjinrong', 'exp_qiche', 'exp_jiaotongyunshu',
               'exp_jianzhucailiao', 'exp_shipinyinliao', 'exp_gongyongshiye', 'exp_yousejinshu', 'exp_zonghe',
               'exp_tongxin', 'exp_fangdichan', 'exp_fangzhifushi', 'exp_jixieshebei', 'exp_qinggongzhizao',
               'exp_shangmaolingshou', 'exp_nonglinmuyu', 'exp_chuanmei', 'exp_huanbao', 'exp_jichuhuagong',
               'exp_dianlishebei', 'exp_dianzi', 'exp_guofangjungong', 'exp_yinhang', 'exp_shiyoushihua',
               'exp_jisuanji', 'exp_jianzhuzhuangshi', 'exp_shehuifuwu', 'exp_meironghuli', 'exp_jiayongdianqi',
               'exp_yiyaoshengwu', 'exp_meitan', 'ret_size', 'ret_non_linear_size', 'ret_momentum', 'ret_liquidity',
               'ret_book_to_price', 'ret_leverage', 'ret_growth', 'ret_earnings_yield', 'ret_beta',
               'ret_residual_volatility', 'ret_comovement', 'ret_gangtie', 'ret_feiyinjinrong', 'ret_qiche',
               'ret_jiaotongyunshu', 'ret_jianzhucailiao', 'ret_shipinyinliao', 'ret_gongyongshiye', 'ret_yousejinshu',
               'ret_zonghe', 'ret_tongxin', 'ret_fangdichan', 'ret_fangzhifushi', 'ret_jixieshebei',
               'ret_qinggongzhizao', 'ret_shangmaolingshou', 'ret_nonglinmuyu', 'ret_chuanmei', 'ret_huanbao',
               'ret_jichuhuagong', 'ret_dianlishebei', 'ret_dianzi', 'ret_guofangjungong', 'ret_yinhang',
               'ret_shiyoushihua', 'ret_jisuanji', 'ret_jianzhuzhuangshi', 'ret_shehuifuwu', 'ret_meironghuli',
               'ret_jiayongdianqi', 'ret_yiyaoshengwu', 'ret_meitan', 'is_suspend', 'st', 'limit_mark',
               'gen_padding_flag'],
}

# 有一些常量(比如"open",在我们的回测代码体系中,它指的是开盘价, 对于不同的数据集, 我们有一些索引值是在创建UTILS对象时,就预先生成好的,根据它在当前所使用的数据集字段列表中的位置确定)
# 但是, 不同的数据集中, 字段序列名称可能有差异, 表示开盘价的字段有可能叫做"open", 也有可能叫别的名称,比如说:"f_open", 这样会导致我们无法通过一套简单的查询代码来定位索引
# 现在, 我们把所有涉及需要使用的常量都列出来, 对应它在每个数据集中的真实名称也罗列出来, 以减少代码修改量, 以后如有新的数据集产生, 只需要改动下面这一处映射表即可
'''
CONST_FIELD_NAME = {
    "open": {'wd311': 'f_p_open', 'def': 'open'},
    # 表示open字段在wd311数据集中的真实名称为f_p_open, 其它的都走'def'的设定(不设定def,则表明字段名与key完全相同)
    "high": {'wd311': 'f_p_high'},
    "low": {'wd311': 'f_p_low'},
    "close": {'wd311': 'f_p_close'},
    "pre_close": {'wd311': 'f_p_pre_close'},
    "5m": {'wd311': 'f_p_open5min_avg', 'def': 'avg_price'},
    "935": {'wd311': 'f_p_open5min_close', 'def': '35_close'},
    "60m": {'wd311': 'f_p_open1h_avg', 'def': '60m_avg_price'},
    "day": {'wd311': 'f_p_avg', 'def': 'daily_avg_price'},
    "gen_exchange": {'wd311': 'f_d_exchange_id'},
    "gen_market": {'wd311': 'f_d_market_id'},
    "gen_industry": {'wd311': 'f_d_industry_id'},
    "cq": {'def': 'gen_cq'},
    "st": {},  # 在所有数据集中,都是原名称,所以不用指定
    "mv": {'wd311': 'f_i_mv_circ', 'def': 'circ_mv'},
    "total_mv": {'wd311': 'f_i_mv_total'},
    "stop": {'def': 'is_suspend'},
    "pad": {'def': 'gen_padding_flag'},
    "open_hfq": {'wd311': 'f_p_open_adj'},
    "high_hfq": {'wd311': 'f_p_high_adj'},
    "low_hfq": {'wd311': 'f_p_low_adj'},
    "close_hfq": {'wd311': 'f_p_close_adj'},
    "amount": {'wd311': 'f_v_amount'},
    "limit_mark": {},
    "adj_factor": {'wd311': 'f_p_adj_factor'},
    "year": {'wd311': 'f_d_year', 'def': 'gen_year'}
}'''

CONST_FIELD_NAME = {
    "open": {'wd395': 'f_p_open', 'def': 'open'},
    # 表示open字段在wd395数据集中的真实名称为f_p_open, 其它的都走'def'的设定(不设定def,则表明字段名与key完全相同)
    "high": {'wd395': 'f_p_high'},
    "low": {'wd395': 'f_p_low'},
    "close": {'wd395': 'f_p_close'},
    "pre_close": {'wd395': 'f_p_pre_close'},
    "5m": {'wd395': 'f_p_open5min_avg', 'def': 'avg_price'},
    "935": {'wd395': 'f_p_open5min_close', 'def': '35_close'},
    "60m": {'wd395': 'f_p_open1h_avg', 'def': '60m_avg_price'},
    "day": {'wd395': 'f_p_avg', 'def': 'daily_avg_price'},
    "gen_exchange": {'wd395': 'f_d_exchange_id'},
    "gen_market": {'wd395': 'f_d_market_id'},
    "gen_industry": {'wd395': 'f_d_industry_id'},
    "cq": {'def': 'gen_cq'},
    "st": {},  # 在所有数据集中,都是原名称,所以不用指定
    "mv": {'wd395': 'f_i_mv_circ', 'def': 'circ_mv'},
    "total_mv": {'wd395': 'f_i_mv_total'},
    "stop": {'def': 'is_suspend'},
    "pad": {'def': 'gen_padding_flag'},
    "open_hfq": {'wd395': 'f_p_open_adj'},
    "high_hfq": {'wd395': 'f_p_high_adj'},
    "low_hfq": {'wd395': 'f_p_low_adj'},
    "close_hfq": {'wd395': 'f_p_close_adj'},
    "amount": {'wd395': 'f_v_amount'},
    "limit_mark": {},
    "adj_factor": {'wd395': 'f_p_adj_factor'},
    "year": {'wd395': 'f_d_year', 'def': 'gen_year'}
}

def get_index_of_field(CF, DS, field_list: list):
    """
    当出现新的数据集时,只需要改此一处即可
    CF 是当前要定位其索引的常量字段(被外界直接以某个约定名称来调用的),如: open 表示开盘价  st表示是否被ST了
    DS 当前的数据集名称
    field_list 是当前数据集所对应的全部字段名称
    """
    assert CONST_FIELD_NAME.__contains__(CF), f"要找的常量字段{CF}信息不存在,请检查"
    fd = CONST_FIELD_NAME[CF]
    if fd.__contains__(DS):
        # 如果当前数据集下有特别指定的映射名称
        real_name = fd[DS]
    elif fd.__contains__('def'):
        # 如果当前数据集下有默认的名称映射关系
        real_name = fd['def']
    else:
        # 如果以上二者都未设定,则表明真实字段名称与CF完全相同
        real_name = CF
    fid = field_list.index(real_name)
    assert fid >= 0, f"字段索引位置出错,请检查,CF={CF},DS={DS}"
    return fid


def remove_quit_stock(all_stocklist, tradeDay):
    """
    排除退市股票
    """
    global g_quit_info_path
    if not os.path.exists(g_quit_info_path):
        return all_stocklist, 0
    # 从文件中加载数据
    with open(g_quit_info_path, 'r', encoding='utf-8') as file:
        quitDict = json.load(file)
    tradeDay = f'{tradeDay}'
    if not quitDict.__contains__(tradeDay):
        return all_stocklist, 0
    quitStockList = quitDict[tradeDay]
    retStockList = []
    rmv = 0
    for code in all_stocklist:
        if code in quitStockList:
            rmv += 1
        else:
            retStockList.append(code)
    return retStockList, rmv


def remove_llm_bad(all_stocklist, tradeDay):
    fn = os.path.join(g_llm_bad_info_path, f"{tradeDay}.json")
    if not os.path.exists(fn):
        return all_stocklist, 0
    # 从文件中加载数据
    with open(fn, 'r', encoding='utf-8') as file:
        content = file.read().replace('\n', '')
    if content is None or content == '':
        return all_stocklist, 0

    # 安全解析字符串
    my_list = ast.literal_eval(content)
    new_list = []
    rmv = 0
    for x in all_stocklist:
        if x in my_list:
            rmv += 1
        else:
            new_list.append(x)
    return new_list, rmv


class FileReaderBufferProcessor:
    def __init__(self, file_list, group_size, first_day, total_days, last_day):
        self.first_day = first_day
        self.total_days = total_days
        self.last_day = last_day
        fl = len(file_list)
        gs = 1 + fl // group_size
        gs = max(gs, 5)  # 每个进程最少处理8个文件
        idx = 0
        self.code_list_group = []
        while idx < fl:
            self.code_list_group.append(file_list[idx:idx + gs])
            idx += gs
        self.data_results = {}

    def process_files(self, file_group, proc_no):
        data_res = {}
        for f in file_group:
            stock_code, data = get_someline_from_csv(f, self.first_day, self.total_days, self.last_day)
            if data is not None:
                data_res[stock_code] = data
        if len(data_res) == 0:
            print(f"{proc_no}--read-data-error")
        return data_res

    def collect_result(self, result):
        if result is None:
            return
        self.data_results.update(result)

    def run(self):
        pool = multiprocessing.Pool()
        proc_no = 0
        for code_group in self.code_list_group:
            pool.apply_async(self.process_files, args=(code_group, proc_no), callback=self.collect_result)
            proc_no += 1
        pool.close()
        pool.join()

    def get_result(self):
        return self.data_results


g_field_index = {}


def set_g_csv_path(DS, csv_path):
    global g_csv_path
    global g_field_index
    g_csv_path = csv_path
    if g_csv_path == '' or DS == '':
        return
    DS = str(DS).lower()
    assert g_source_field.__contains__(DS), f"指定的数据集不存在,当前支持的数据集有{list(g_source_field.keys())}"
    src_field_list = g_source_field[DS]

    g_field_index = {}
    for CF in list(CONST_FIELD_NAME.keys()):
        g_field_index[CF] = get_index_of_field(CF, DS, src_field_list)


class StockData:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.all_data = {}  # 这个是所有的缓存数据  key: 股票代码xxxxxx.SH  value: 一个numpy数组,某只股票连续n天的数据
        self.all_index = {}  # 这个是一个索引字典  key: 股票代码xxxxxx.SH value: 是另一个字典[key 为日期, vlaue是一个序号表示这天的数据在第几行]
        # 读取数据方法:
        # (1)根据股票代码s从 all_data[s]得到该股票的全部缓存数据data,
        # (2)从all_index[s]得到该股票的[日期,序号]字典sd,
        # (3)根据当前所需要的日期td, 从sd[td]得到行号i
        # (4)从data[i] 就可以读取这只股票在td这一天的数据
        self.stop_trade_dic = {}  # 20250919 增加一个缓存数据项,缓存数据区间内所有日期内的停牌股票, 结构为[key 为股票代码, value为一个list[停牌日期]]

        # 本对象在当前类被import时会自动创建一个初始对象,此时尚未指定当前实际需要使用的数据集,暂时给第一个
        default_DS = list(g_source_field.keys())[0]  # 默认第一个数据集
        self.DS = default_DS

        self.full_field_list = g_source_field[self.DS]

        # 常用价格类字段的下标
        self.field_index = {}

        self.zz_index = {}
        self.set_pad_open_zero = True

    def reset(self, DS, csv_path, indexName='', set_pad_open_zero=True):
        DS = str(DS).lower()
        set_g_csv_path(DS, csv_path)
        self.DS = DS
        self.csv_path = csv_path
        self.all_data = {}
        self.all_index = {}
        self.stop_trade_dic = {}
        self.set_pad_open_zero = set_pad_open_zero
        self.full_field_list = g_source_field[DS]

        # 常用价格类字段的下标
        self.field_index = copy.deepcopy(g_field_index)

        zzlist = [indexName]
        for zz in zzlist:
            if zz == '':
                continue
            local_file = os.path.join(f"{CM.G_ROOT_PATH}/indexData/", f"{zz}.json")
            if os.path.exists(local_file):
                # 读取本地缓存文件
                with open(local_file, 'r', encoding='utf-8-sig') as f:
                    self.zz_index[zz] = json.load(f)
            else:
                raise ValueError(f"指定的指数数据文件{local_file}不存在,请检查.")

    def is_buffer_data_loaded(self):
        return len(self.all_data) > 0

    def is_stock_st_or_suspend(self, stock_data):
        # 判断一只股票是否曾经ST或者停牌  stock_data [L,C]
        ever_st = stock_data[:, self.field_index['st']]  # [L]
        ever_st = ever_st > 0.00001
        if torch.sum(ever_st) > 0:
            return True

        ever_sus = stock_data[:, self.field_index['stop']]
        ever_sus = ever_sus > 0.00001
        if torch.sum(ever_sus) > 0:
            return True

        return False

    def get_zz_value(self, indexName, t1Day, t2Day):
        """
        根据交易日,获取中证1000指数的相对价值
        :param indexName
        :param t1Day # t1日期 [B]
        :param t2Day # t2日期 [B]
        """

        if not self.zz_index.__contains__(indexName):
            return 1.
        zzindex = self.zz_index[indexName]

        # self.zzindex 是一个字典表key为str(日期),value为当日的中证1000指数
        B = t1Day.shape[0]
        fv = torch.zeros(B).to(t1Day.device)  # 构造全1张量 [B]
        for i in range(B):
            s = str(int(t1Day[i].item()))
            t = str(int(t2Day[i].item()))
            if not zzindex.__contains__(s):
                # 如果当前日期在字典中并不存在,则表明这个日期是一个非交易日,
                # 相应的大盘价值直接设置为1, 下同
                fv[i] = 1.
            elif not zzindex.__contains__(t):
                fv[i] = 1.
            else:
                szz = zzindex[s]
                tzz = zzindex[t]
                fv[i] = tzz / szz
        return fv

    def read_index(self, indexName, tradeDay):
        # 先尝试从本地缓存读取
        if self.zz_index.__contains__(indexName):
            # 已经缓存了本指数
            zs = self.zz_index[indexName]
            if zs.__contains__(tradeDay):
                return zs[tradeDay]
        print(f"ERROR--{indexName}--{tradeDay}")
        return None

    def read_stock_price(self, stock_code, price_type, tradeDay, fuquan=False):
        """
        读取股票价格
        :param stock_code: 股票代码
        :param price_type: 价格类型
        :param tradeDay: 交易日
        :param fuquan: 是否要对价格进行复权操作之后再返回
        :return: 股票价格
        """

        # 股票代码可能带有.SH或.SZ后缀，需要去掉
        if '.' in stock_code:
            stock_code = stock_code.split('.')[0]
        if stock_code not in self.all_data:
            return None

        index = self.field_index

        stock_data = self.all_data[stock_code]
        # slen = len(stock_data)

        # 将targetLastDate转换为int类型,并解析出年月日
        tradeDay = int(tradeDay)

        dd = self.all_index[stock_code]
        if not dd.__contains__(tradeDay):
            return None
        i = dd[tradeDay]

        if price_type == 'stcq':
            if stock_data[i, self.field_index['cq']] > 0.99 or stock_data[i, self.field_index['st']] > 0.99:
                return 1.
            else:
                return 0.
        elif price_type == 'mv':
            return round(float(stock_data[i, self.field_index['mv']]) / 10000., 2)  # 流通市值单位本为万,这里转换为亿,保留2位小数
        else:
            pi = index[price_type]
            if price_type == 'open' and stock_data[i, self.field_index['pad']] > 0.99 and self.set_pad_open_zero:
                # 从缓存中读取股票的开盘价时, 如果这一天刚好是padding的数据,则将开盘价返回为0
                # 这样, 在回测时,这只股票就会被剔除出推理的股票池了, 这是个临时的补救措施!!! 20250917
                return 0.
            pv = float(stock_data[i, pi])
            if fuquan:
                fq_i = index['adj_factor']
                fq_factor = float(stock_data[i, fq_i])
                pv = pv * fq_factor
            return round(pv, 2)

    def make_data_index(self, stock_data):
        dd = {}
        slen = len(stock_data)
        YI = self.field_index['year']
        for i in range(slen):
            bd = int(stock_data[i][YI]) * 10000 + int(stock_data[i][YI + 1]) * 100 + int(stock_data[i][YI + 2])
            dd[bd] = i
        return dd

    def make_stop_index(self, stock_data):
        # stock_data 是某一只股票的全部缓存数据
        # 这里遍历全部数据, 找到停牌的日期, 将所有停牌日期放入一个list并返回
        slen = len(stock_data)
        stop_day_list = []
        YI = self.field_index['year']
        for i in range(slen):
            stop_flag = stock_data[i][self.field_index['stop']]
            if stop_flag > 0.99:
                # 停牌标识位为1
                bd = int(stock_data[i][YI]) * 10000 + int(stock_data[i][YI + 1]) * 100 + int(stock_data[i][YI + 2])
                stop_day_list.append(bd)
        return stop_day_list

    def read_stock_data(self, stock_code, last_day, time_step):
        """
        读取股票数据
        :param stock_code: 股票代码
        :param last_day: 最后日期
        :param time_step: 时间步长
        :return: 股票数据np.array
        """

        if stock_code not in self.all_data:
            return None

        stock_data = self.all_data[stock_code]
        slen = len(stock_data)
        if slen < time_step:
            return None

        # 将targetLastDate转换为int类型,并解析出年月日
        last_day = int(last_day)
        dd = self.all_index[stock_code]
        if dd.__contains__(last_day):
            i = dd[last_day]
            if i < time_step - 1:
                return None
            return stock_data[i - time_step + 1:i + 1]
        return None

    def init_buffer_data(self, DS, firstTradeDay, lastTradeDay):
        self.reset(DS, g_csv_path)
        # 获取所有csv文件名
        csv_files = [f for f in os.listdir(self.csv_path) if f.endswith('.csv')]
        # 将文件名补全为完整路径
        full_csv_files = [os.path.join(self.csv_path, f) for f in csv_files]

        # first_day = get_next_trade_day(firstTradeDay, -81)  # 首个交易日向历史方向偏移80天
        # first_day = get_real_next_trade_day(first_day,next_day=False)
        # lastTradeDay = get_real_next_trade_day(lastTradeDay)  # 最后交易日向未来方向偏移1天
        total_days = DT.count_trade_days(firstTradeDay, lastTradeDay)  # 总天数

        start_time = time.time()
        print(f"多进程并行读取所有股票数据自{firstTradeDay}开始的连续{total_days}天数据...")
        print(f"数据路径:{self.csv_path}")
        processer = FileReaderBufferProcessor(full_csv_files, 32, firstTradeDay, total_days, lastTradeDay)
        processer.run()
        self.all_data = processer.get_result()
        # 构造日期索引---对每只股票
        self.all_index = {}
        self.stop_trade_dic = {}
        for stock_code, stock_data in self.all_data.items():
            self.all_index[stock_code] = self.make_data_index(stock_data)  # 构造日期索引
            stop_dic = self.make_stop_index(stock_data)  # 构造停牌数据字典
            if len(stop_dic) > 0:
                self.stop_trade_dic[stock_code] = stop_dic
        print(f"读取{len(self.all_data)}个文件, 总耗时{time.time() - start_time:.2f}s")
        return len(self.all_data)


# 定义一个全局数据对象
G_ALL_STOCK_DATA = StockData(g_csv_path)


def get_stop_stocks(stock_list, tradeDay):
    stop_list = []
    if not G_ALL_STOCK_DATA.is_buffer_data_loaded():
        return stop_list

    # 获取缓存的停牌数据字典
    stop_dic = G_ALL_STOCK_DATA.stop_trade_dic
    for stock in stock_list:
        if not stop_dic.__contains__(stock):
            continue
        stop_days = stop_dic[stock]  # 该股票的所有停牌日期列表
        if tradeDay in stop_days:
            # 当前指定的交易日刚好在停牌日期列表中
            stop_list.append(stock)
    return stop_list


def get_someline_from_csv(csv_path, first_day, total_days, last_day):
    """
    从一个CSV文件中读取最后若干条数据
    :param csv_path:  文件路径
    :param first_day: 起始日期
    :param total_days: 总天数
    :param last_day: lastday
    :return: 数据np.array
    """

    global g_field_index

    # 从文件名中获取股票代码
    stock_code = os.path.basename(csv_path).split('.')[0]

    # 读取全部数据进入内存
    df = pd.read_csv(csv_path, header=None)

    # 将targetLastDate转换为int类型,并解析出年月日
    first_day = int(first_day)
    ty = first_day // 10000
    tm = (first_day % 10000) // 100
    td = first_day % 100
    # 查找符合条件的行（假设年、月、日分别在 0、1、2 列）
    YI = g_field_index['year']
    mask = (df[YI] == ty) & (df[YI + 1] == tm) & (df[YI + 2] == td)
    matching_rows = df[mask]
    if matching_rows.shape[0] == 0:
        # 未找到起始日期为first_day的数据行
        # 尝试查找最末尾天的数据

        last_day = int(last_day)
        ty = last_day // 10000
        tm = (last_day % 10000) // 100
        td = last_day % 100
        # 查找符合条件的行（假设年、月、日分别在 0、1、2 列）
        mask = (df[YI] == ty) & (df[YI + 1] == tm) & (df[YI + 2] == td)
        matching_rows = df[mask]
        if matching_rows.shape[0] == 0:
            # 末尾这一天也不存在, 则返回空数据
            return None, None
        else:
            # 否则返回整个数据集中从第1条开始直到该行的全部数据, 该股票大概是在firstday-lastday期间上市的
            r = matching_rows.index[0]
            somelines = df.iloc[0:r + 1].values
            return stock_code, somelines
    else:
        # 找到日期为first_day这一条了
        r = matching_rows.index[0]
        somelines = df.iloc[r:r + total_days].values

        return stock_code, somelines


def get_lastline_from_csv(csv_path, last_lines):
    """
    从一个CSV文件中读取最后若干条数据
    :param csv_path:  文件路径
    :param last_lines:  最后若干条数据的数量
    :return: 最后若干条数据np.array
    """

    # 从文件名中获取股票代码
    stock_code = os.path.basename(csv_path).split('.')[0]

    # 读取全部数据进入内存
    df = pd.read_csv(csv_path, header=None)

    if df.shape[0] < last_lines:
        last_lines = df.shape[0]

    # 取最后last_lines条数据
    last_data = df.tail(last_lines).values

    return stock_code, last_data


def read_pmfile(pmfile):
    try:
        with open(pmfile, 'r') as f:
            ret = json.load(f)
        return ret
    except Exception as e:
        print(e, pmfile)
        return None


def make_output_flag(args):
    md = os.path.basename(args.model)
    md = os.path.splitext(md)[0]
    # if args.G > 1:
    #     md = md + f"_G{args.G}"
    if not args.withBJ:
        md = md + "_noBJ"
    if args.BJMAX > 0:
        md = f"{md}_BJMAX{args.BJMAX}"
    if args.t > 0.0001:
        md = f"{md}_T{args.t}"
    md = f"{md}_m{args.minw}_m{args.max_sw}"
    md += (f"_D{args.TD}_FC{args.Factor_constraint}_IX{args.indexTarget}"
           f"_IP{args.indexpull}_PX{args.price_target}_FQ{args.fuquan}"
           f"_CL{getattr(args, 'close_limit_filter', 1)}_TemplateV3")
    return md


def make_output_json_name(args, tradeDay):
    md = make_output_flag(args)
    targetFile = f"{tradeDay}_{md}.json"
    return targetFile


def make_output_pool_name(args, tradeDay):
    md = os.path.basename(args.model)
    md = os.path.splitext(md)[0]
    if args.G > 1:
        md = md + f"_G{args.G}"
    if not args.withBJ:
        md = md + "_noBJ"
    if args.BJMAX > 0:
        md = f"{md}_BJMAX{args.BJMAX}"
    targetFile = f"{tradeDay}_{md}_s{args.skipLoss}_m{args.minw}_SF{args.shuffle}.pool"
    return targetFile


def cal_tr(prev_w, next_w):
    """
    计算TR值
    :param prev_w: 上一个权重字典
    :param next_w: 下一个权重字典
    :return: TR值
    """

    if len(prev_w) == 0 and len(next_w) == 0:
        return 0.
    if len(prev_w) == 0:
        return 1.0 - next_w['CASH']
    if len(next_w) == 0:
        return 1.0 - prev_w['CASH']
    ka = set(prev_w.keys())
    kb = set(next_w.keys())
    k = ka.union(kb)
    tr = 0.
    for key in k:
        if key == 'CASH':
            continue
        if key in prev_w and key in next_w:
            tr += abs(prev_w[key] - next_w[key])
        elif key in prev_w:
            tr += prev_w[key]
        elif key in next_w:
            tr += next_w[key]
    return tr / 2.


def drift_weights(weights, previous_prices, current_prices):
    """成交价变化后的交易前权重，仅用于费用核算，不向模型注入未来价格。"""
    values = {}
    for code, weight in weights.items():
        ratio = 1.
        if code != 'CASH':
            p0, p1 = previous_prices.get(code), current_prices.get(code)
            if p0 is not None and p1 is not None and p0 > 0 and p1 > 0:
                ratio = p1 / p0
        values[code] = weight * ratio
    total = sum(values.values())
    if total <= 0:
        raise ValueError('交易前组合价值必须大于0')
    return {code: value / total for code, value in values.items()}


def make_link_url(stock_code):
    """
    生成股票链接(指向东方财富网)
    :param stock_code: 股票代码
    :return: 股票链接
    """

    stocktype = stock_code[-2:].upper()
    stock_simple_code = stock_code[:-3]
    if stocktype == 'SH':
        return f"http://quote.eastmoney.com/sh{stock_simple_code}.html"
    elif stocktype == 'SZ':
        return f"http://quote.eastmoney.com/sz{stock_simple_code}.html"
    elif stocktype == 'BJ':
        return f"http://quote.eastmoney.com/bj/{stock_simple_code}.html"
    else:
        return ""


def get_last_timesteps_from_path(csv_path, timesteps, last_date, stocklist=None):
    """
    从一个指定的路径下读取所有csv文件，并取最后若干条数据
    :param csv_path:  文件路径
    :param timesteps:  最后若干条数据的数量
    :param last_date:  最后一条数据的日期
    :param stocklist:  指定股票列表，如果为None，则读取全部股票
    :return: 最后若干条数据np.array
    """

    # 获取所有csv文件名
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]

    if stocklist is not None:
        target_list = []
        for stock in stocklist:
            # 这里的stock是完整的股票代码如'600000.SH', 先把尾巴去掉
            stock = stock.split('.')[0]
            target_list.append(f"{stock}.csv")
        # 取出两个list的交集
        csv_files = list(set(csv_files) & set(target_list))  # 取交集

    # 将文件名补全为完整路径
    csv_files = [os.path.join(csv_path, f) for f in csv_files]

    if G_ALL_STOCK_DATA.is_buffer_data_loaded():
        # 如果已经有数据缓存对象,则直接从内存中读取
        ret_data = []
        for f in csv_files:
            stock_code = os.path.basename(f).split('.')[0]
            sd = G_ALL_STOCK_DATA.read_stock_data(stock_code, last_date, timesteps)
            if sd is not None:
                ret_data.append(sd)
            # else:
            #     print(f"tryLoad:{stock_code}-{last_date}-{timesteps}--FAILED")
        if len(ret_data) == 0:
            return None
        return np.array(ret_data)
    else:
        # 读取所有文件
        print(f"{DT.timestr()}尝试多进程读取{len(csv_files)}个数据文件...")
        processer = FileReaderProcessor(csv_files, 48, timesteps, int(last_date))
        processer.run()
        data_list = processer.get_result()
        # 将list转换为numpy数组
        if data_list is None:
            return None
        return np.array(data_list)


def read_single_price(g_csv_path, g_field_index, code, price_type, tradeDay, fuquan=False):
    """
    读取单个股票的价格数据
    :param g_csv_path
    :param g_field_index
    :param code: 股票代码
    :param price_type: 价格类型[open, 5m, 935, 60m, day]
    :param tradeDay: 交易日
    :param fuquan: 是否需要乘上复权因子
    :return: 价格数据
    """

    stock_file = os.path.join(g_csv_path, f"{code.split('.')[0]}.csv")
    if not os.path.exists(stock_file):
        print(f"{DT.timestr()}股票{code}文件{stock_file}不存在...")
        return 0.

    index = g_field_index

    # 读取全部数据进入内存
    df = pd.read_csv(stock_file, header=None)
    # 第5,6,7列分别为年月日
    tradeDay = str(tradeDay)
    ty = int(tradeDay[:4])
    tm = int(tradeDay[4:6])
    td = int(tradeDay[6:])
    # 取出指定日期的数据
    YI = g_field_index['year']
    df = df[df[YI] == ty]
    df = df[df[YI + 1] == tm]
    df = df[df[YI + 2] == td]
    if df.shape[0] == 0:
        # print(f"{DT.timestr()}股票{code}日期{tradeDay}数据不存在...")
        return 0.
    # 预防错误: 如果文件中同一日期的数据有多条,取第1条即可
    df = df.iloc[0]
    # 取出指定价格类型的数据
    if price_type == 'stcq':
        cq_i = index['cq']
        st_i = index['st']
        if df[cq_i] > 0.99 or df[st_i] > 0.99:
            return 1.
        else:
            return 0.
    elif price_type == 'mv':
        return round(float(df[index['mv']]) / 10000., 2)
    else:
        pv = float(df[index[price_type]])
        if fuquan:
            # 需要对价格字段乘以当天的复权因子后再返回
            fq_i = index['adj_factor']
            fq_factor = float(df[fq_i])
            pv = pv * fq_factor
        return round(pv, 2)


def read_group_price(code_list, price_type, tradeDay, force_read_file=False, fuquan=False):
    if len(code_list) == 0:
        return {}
    # 多进程方式读取多个股票的价格数据
    # 读取所有文件
    if G_ALL_STOCK_DATA.is_buffer_data_loaded() and (not force_read_file):
        # 如果已经有数据缓存对象,则直接从内存中读取
        ret_data = {}
        for code in code_list:
            priceValue = G_ALL_STOCK_DATA.read_stock_price(code, price_type, tradeDay, fuquan=fuquan)
            if priceValue is None:
                # raise ValueError(f"股票{code}价格数据不存在于缓存数据中")
                continue
            ret_data[code] = priceValue
        if len(ret_data) > 0:
            return ret_data
        else:
            print(f"**********缓存未命中,需要读取文件*********{code_list[0]}-{price_type}-{tradeDay}")

    processer = FilePriceProcessor(g_csv_path, g_field_index, code_list, 24, price_type, tradeDay, fuquan=fuquan)
    processer.run()
    price_dict = processer.get_result()
    return price_dict


class FilePriceProcessor:
    def __init__(self, g_csv_path, g_field_index, codelist, group_size, priceType, readDay, fuquan=False):
        """
        初始化文件处理器
        :param codelist: 待处理文件列表
        :param group_size: 需要分成多少组进行并行处理（即进程数)
        """
        self.g_csv_path = g_csv_path
        self.g_field_index = g_field_index
        self.priceType = priceType
        self.readDay = readDay
        self.fuquan = fuquan
        fl = len(codelist)
        gs = 1 + fl // group_size
        gs = max(gs, 5)  # 每个进程最少处理8个文件
        idx = 0
        self.code_list_group = []
        while idx < fl:
            self.code_list_group.append(codelist[idx:idx + gs])
            idx += gs
        self.data_results = {}

    def process_files(self, code_group, proc_no):
        data_res = {}
        for code in code_group:
            # 在这里编写对文件的处理逻辑
            priceValue = read_single_price(self.g_csv_path, self.g_field_index, code, self.priceType, self.readDay,
                                           fuquan=self.fuquan)
            data_res[code] = priceValue
        if len(data_res) == 0:
            print(f"********{DT.timestr()}进程#{proc_no}----结束{len(code_group)},{len(data_res)}")
        return data_res

    def collect_result(self, result):
        if result is None:
            return
        self.data_results.update(result)

    def run(self):
        pool = multiprocessing.Pool()
        proc_no = 0
        for code_group in self.code_list_group:
            pool.apply_async(self.process_files, args=(code_group, proc_no), callback=self.collect_result)
            proc_no += 1
        pool.close()
        pool.join()

    def get_result(self):
        return self.data_results


def get_last_timesteps_from_csv(csv_path, timesteps, targetLastDate):
    """
    从一个CSV文件中读取最后若干条数据
    :param csv_path:  文件路径
    :param timesteps:  最后若干条数据的数量
    :param targetLastDate:  最后一条数据的日期
    :return: 最后若干条数据np.array, 最后一条数据的日期
    """

    global g_field_index

    # 读取全部数据进入内存
    df = pd.read_csv(csv_path, header=None)

    # 判断总数据条数是否大于timesteps
    if df.shape[0] < timesteps:
        return None, None

    # 将targetLastDate转换为int类型,并解析出年月日
    targetLastDate = int(targetLastDate)
    ty = targetLastDate // 10000
    tm = (targetLastDate % 10000) // 100
    td = targetLastDate % 100
    # 查找符合条件的行（假设年、月、日分别在 0、1、2 列）
    YI = g_field_index['year']
    mask = (df[YI] == ty) & (df[YI + 1] == tm) & (df[YI + 2] == td)
    matching_rows = df[mask]
    if matching_rows.shape[0] == 0:
        return None, None

    # 找到这一条了
    r = matching_rows.index[0]
    if r - timesteps + 1 < 0:
        # 虽然找到了, 但是总数据行数不够多
        return None, None
    last_timesteps = df.iloc[r - timesteps + 1:r + 1].values
    return last_timesteps, targetLastDate


class FileReaderProcessor:
    def __init__(self, filelist, group_size, timesteps, lastDate):
        """
        初始化文件处理器
        :param filelist: 待处理文件列表
        :param group_size: 需要分成多少组进行并行处理（即进程数)
        """

        self.timesteps = timesteps
        self.lastDate = lastDate

        fl = len(filelist)
        gs = 1 + fl // group_size
        gs = max(gs, 50)  # 每个进程最少处理50个文件
        idx = 0
        self.file_list_group = []
        while idx < fl:
            self.file_list_group.append(filelist[idx:idx + gs])
            idx += gs
        self.data_results = []

    def process_files(self, file_group, proc_no):
        # print(f"{DT.timestr()}进程#{proc_no}----开始{len(file_group)}")
        handeled_count = 0
        success_count = 0
        data_res = []
        try:
            for file_name in file_group:
                # 在这里编写对文件的处理逻辑
                last_timesteps, last_date = get_last_timesteps_from_csv(file_name, self.timesteps, self.lastDate)
                if last_date == self.lastDate:
                    data_res.append(last_timesteps)
                    success_count += 1
                # else:
                #     print(f"{DT.timestr()}进程#{proc_no}文件{file_name}读取失败 last_date={last_date},{self.lastDate}")
                handeled_count += 1
                # if handeled_count % 200 == 0:
                #     print(f"{DT.timestr()}进程#{proc_no}读取{handeled_count}个文件,成功{success_count}个")
            # print(f"{DT.timestr()}进程#{proc_no}----结束{handeled_count},成功{success_count}")
        except Exception as e:
            print(f"********进程#{proc_no} error: {e}")
            return None

        # 合并数据,增加一个维度
        try:
            # print(f"********{DT.timestr()}进程#{proc_no}----结束")
            data_res = [data.reshape(1, *data.shape) for data in data_res]
            combined_array = np.concatenate(data_res, axis=0)
            # print(f"{DT.timestr()}进程#{proc_no}----结束,共读取{handeled_count}个文件,成功{success_count}个,数据维度{combined_array.shape}")
            return combined_array
        except Exception as e:
            print(f"********进程#{proc_no} error: {e}")
            return None

    def collect_result(self, result):
        if result is None:
            return
        self.data_results.append(result)

    def run(self):
        pool = multiprocessing.Pool()
        proc_no = 0
        for file_group in self.file_list_group:
            # print(f"进程#{proc_no}处理{len(file_group)}个文件")
            pool.apply_async(self.process_files, args=(file_group, proc_no), callback=self.collect_result)
            proc_no += 1
        pool.close()
        pool.join()

    def get_result(self):
        if len(self.data_results) == 0:
            return None
        data_array = np.concatenate(self.data_results, axis=0)
        return data_array


def restore_stock_code(stock_code):
    """
    还原六位股票编码到带后缀的完整股票编码
    :param stock_code: 六位股票编码
    :return: 完整股票编码
    """
    try:
        # 处理非数字输入的异常
        code = str(int(stock_code)).zfill(6)
    except (ValueError, TypeError):
        raise ValueError(f"无效的股票代码：{stock_code}")

    if code[:2] in ('00', '30'):
        return code + '.SZ'  # 深市主板（00）、创业板（30）
    elif code[:1] == '6':
        return code + '.SH'  # 沪市（60开头主板、688开头科创板）
    elif code[:1] in ('8', '9'):
        return code + '.BJ'  # 北交所（8、9开头）
    else:
        # 处理未覆盖的代码（如港股通、其他特殊代码）
        raise ValueError(f"无法识别的股票代码：{code}")


def get_stock_list(stock_info_path, withBJ=True):
    """
    从股票信息文件中获取股票列表
    :param stock_info_path: 股票信息文件路径
    :param withBJ: 是否包含北交所股票,1表示包含,0表示不包含
    :return: 股票列表
    """
    with open(stock_info_path, 'r', encoding='utf-8-sig') as file:
        g_stock_dict = json.load(file)
        stock_list = list(g_stock_dict.keys())
        abs_all = stock_list.copy()
        if not withBJ:
            new_stock_list = []
            for stock in stock_list:
                if not 'BJ' in stock.upper():
                    new_stock_list.append(stock)
            stock_list = new_stock_list
    return stock_list, abs_all


def read_all_stock_chname():
    stock_dict = {}

    today = DT.timestr(1)
    opent1_file = os.path.join(g_open_t1_path, f"open_price_{today}.csv")
    if not os.path.exists(opent1_file):
        today = DT.get_next_trade_day(today, -1)
        opent1_file = os.path.join(g_open_t1_path, f"open_price_{today}.csv")
        if not os.path.exists(opent1_file):
            return stock_dict

    try:
        df = pd.read_csv(opent1_file, header=None)
        df.columns = ['ts_code', 'open', 'chname']
        for index, row in df.iterrows():
            stock_code = row['ts_code']
            # open_price = row['open']
            chname = row['chname']
            stock_dict[stock_code] = chname
    except Exception as e:
        print(e)
        return stock_dict
    return stock_dict


def read_price_from_buffer(refer_stocklist, price_type, tradeDay):
    today = DT.timestr(1)
    if int(today) == int(tradeDay):
        return None, 0
    open_dict = {}
    open_price = read_group_price(refer_stocklist, price_type, tradeDay)
    for stock, price in open_price.items():
        open_dict[stock] = [price, '']
    return open_dict, len(open_dict)


def read_open_price_from_csv(refer_stocklist, opent1_file):
    """
    从csv文件中读取开盘价
    :param refer_stocklist: 参考股票列表(完整股票名称)
    :param opent1_file: 开盘价文件名
    :return: 开盘价字典, 成功读取合法的开盘价的股票数量
    """
    open_dict = {}
    success_count = 0
    for stock in refer_stocklist:
        open_dict[stock] = [0.0, '']
    try:
        df = pd.read_csv(opent1_file, header=None)
        # 获取df中的列数
        col_count = df.shape[1]
        if col_count == 2:
            df.columns = ['ts_code', 'open']
        else:
            df.columns = ['ts_code', 'open', 'chname']
        for index, row in df.iterrows():
            stock_code = row['ts_code']
            open_price = row['open']
            chname = stock_code
            if col_count > 2:
                chname = row['chname']
            if open_dict.__contains__(stock_code) and open_price > 0:
                open_dict[stock_code] = [open_price, chname]
                success_count += 1
        return open_dict, success_count
    except Exception as e:
        print(f"read_open_price_from_csv error003: {opent1_file},{e}")
        return open_dict, success_count


def make_date_from_tensor(x, yidx, midx, didx):
    year = int(x[yidx] * 40 + 1990)
    month = int(x[midx] * 12 + 1)
    day = int(x[didx] * 31 + 1)
    return year * 10000 + month * 100 + day


def make_futuer_seq(x, days, knownChannels, yidx, midx, didx, widx):
    """
    根据已知的序列数据制作未来序列
    :param x: 输入序列 [B, M, L, C]
    :param days: 需要构造的未来天数
    :param knownChannels: 已知的通道数
    :param yidx: 年分字段的索引
    :param midx: 月份字段的索引
    :param didx: 日期字段的索引
    :param widx: 星期字段的索引
    :return: 未来序列 [B, M, days, C]
    """

    B, M, L, C = x.shape
    future = x[:, :, -1:, :].clone()  # [B, M, 1, C]
    future = future.repeat(1, 1, days, 1)  # [B, M, days, C]
    future[:, :, :, knownChannels:] = 0.  # 除了已知的通道，其他通道设置为0

    for b in range(B):
        last_date = make_date_from_tensor(x[b, 0, -1], yidx, midx, didx)
        for i in range(days):
            last_date = DT.get_next_trade_day(str(last_date), 1)
            y, m, d, w = DT.parse_date(last_date)
            future[b, :, i, yidx] = (y - 1990) / 40.  # 年份字段归一化
            future[b, :, i, midx] = (m - 1) / 12.  # 月份字段归一化
            future[b, :, i, didx] = (d - 1) / 31.  # 日期字段归一化
            future[b, :, i, widx] = w / 6.  # 星期字段归一化
    return future


##############################################


def find_str_in_list(input_list, target_str):
    """
    找到列表中字符串的索引
    :param input_list: 输入列表
    :param target_str: 目标字符串
    :return: 字符串的索引
    """
    for i, item in enumerate(input_list):
        if item == target_str:
            return i
    raise ValueError(f"{target_str} not found in input_list")


def make_price_relative_v2(prices, price_channels: list, t1_open):
    """
    生成价格变化向量, 将所有价格类字段都除以T1的开盘价
    :param prices: [B, M, L, C]  M个资产的价格序列
    :param price_channels: 价格序列的通道数
    :param t1_open: t1的开盘价[B,M]
    :return: [B,M, L, C]  M个资产的价格变化序列
    """

    B, M, L, C = prices.shape

    # 扩展维度
    t1_open = t1_open.view(B, M, 1).clone()

    # 计算价格占比
    for pos in price_channels:
        prices[:, :, :, pos] = prices[:, :, :, pos] / t1_open  # [B, M, L, C]
    return prices


def normalize_data(args, timeStepsData):
    """
    归一化数据
    :param args: 模型参数
    :param timeStepsData: 输入数据 [B, M, L, C]
    :return: 归一化后的数据, 归一化参数
    """

    # B, M, L, C = timeStepsData.shape

    # 读取归一化参数
    scaler_info = {}
    if os.path.exists(args.scaler_info_path):
        with open(args.scaler_info_path, 'r') as f:
            scaler_info = json.load(f)

    B, M, L, C = timeStepsData.shape
    assert C == len(args.input_channels), f"输入数据通道数{C}与模型输入通道数{len(args.input_channels)}不一致"
    for i in range(C):
        field_name = args.input_channels[i]
        if not scaler_info.__contains__(field_name):
            raise ValueError(f"未找到{field_name}的归一化参数")
        field_scaler = scaler_info[
            field_name]  # ["MMS-PRICE", 0.01, 5000.0, -3.755361433133479e-07, 0.00024440950582090806]

        field_type = field_scaler[0]  # 字段类型
        if field_type == 'MM-YY':
            # 年份字段只进行归一化(这里相当于min-max归一化, 1990-2030之间的值都归一化到0-1之间)
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - 1990) / 40
        elif field_type == 'MM-MM':
            # 月份字段只进行归一化（这里相当于min-max归一化, 1-12之间的值都归一化到0-1之间）
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - 1) / 12
        elif field_type == 'MM-DD':
            # 日期字段只进行归一化(这里相当于min-max归一化, 1-31之间的值都归一化到0-1之间)
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - 1) / 31
        elif field_type == 'MM-WW':
            # 星期字段只进行归一化(这里相当于min-max归一化, 0~6之间的值都归一化到0-1之间)
            timeStepsData[:, :, :, i] = timeStepsData[:, :, :, i] / 6
        elif field_type == 'MM-EXCHANGE':
            timeStepsData[:, :, :, i] = timeStepsData[:, :, :, i] / 10
        elif field_type == 'MM-MARKET':
            timeStepsData[:, :, :, i] = timeStepsData[:, :, :, i] / 20
        elif field_type == 'MM-INDUSTRY':
            timeStepsData[:, :, :, i] = timeStepsData[:, :, :, i] / 200
        elif field_type == 'MMS-PRICE':
            # 价格字段需要进行标准化
            _, fmin, fmax, fmean, fstd = field_scaler
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - fmin) / (fmax - fmin + 1e-8)
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - fmean) / (fstd + 1e-8)
        elif field_type == 'MMS':
            # 非价格字段需要进行标准化
            _, fmin, fmax, fmean, fstd = field_scaler
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - fmin) / (fmax - fmin + 1e-8)
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - fmean) / (fstd + 1e-8)
        elif field_type == 'MM-LIMIT':
            timeStepsData[:, :, :, i] = (timeStepsData[:, :, :, i] - 1) / 5
        else:
            continue

    return timeStepsData


def get_stock_list_by_mark(logfile, markstr):
    """
    从日志文件中获取股票列表
    :param logfile: 日志文件路径
    :param markstr: 股票名称关键词
    :return: 股票列表
    """
    with open(logfile, 'r') as f:
        for line in f.readlines():
            if line is None:
                break
            line = line.strip()
            if line.startswith(markstr):
                # xxx:['837592.BJ', '300117.SZ', '832171.BJ', '836422.BJ', '834407.BJ', '834770.BJ', '430300.BJ', '871642.BJ', '837046.BJ', '873223.BJ']
                stock_code = line.split(':')[1]
                stock_code = ast.literal_eval(stock_code)
                return stock_code
    return []


def get_stock_list_by_date(logfile, datestr):
    m_s_dict = {}
    with open(logfile, 'r') as f:
        for line in f.readlines():
            if line is None:
                break
            line = line.strip()
            if line.startswith('HS'):
                ms = line.split(':')
                if len(ms) != 2:
                    continue
                mark, stock = ms[0], ms[1]
                mark_date = f"2025{mark[2:6]}"
                if mark_date == datestr:
                    m_s_dict[mark] = ast.literal_eval(stock)
    return m_s_dict


def get_mark_date(mark):
    # HS02211029_B1_MW0_MM
    month = int(mark[2:4])
    day = int(mark[4:6])
    mdate = 2025 * 10000 + month * 100 + day
    return mdate


def get_next_trade_date_of_mark(mark, days=1):
    mdate = get_mark_date(mark)
    return DT.get_next_trade_day(str(mdate), days)


def max_drawdown(v):
    if len(v) == 0:
        return 0.0  # 空序列返回0

    peak = v[0]  # 初始化历史峰值
    max_dd = 0.0  # 初始化最大回撤

    for i in range(1, len(v)):
        if v[i] > peak:
            peak = v[i]  # 更新峰值
        else:
            drawdown = (peak - v[i]) / peak
            if drawdown > max_dd:
                max_dd = drawdown  # 更新最大回撤
    return max_dd


def keep_input_pool_stock(stock_list, args, last_trade_day):
    poolFile = make_output_pool_name(args, last_trade_day)
    poolFile = f"./output/{poolFile}"
    if not os.path.exists(poolFile):
        print(f"file_not_exist: {poolFile}")
        return stock_list
    poolStock = read_pmfile(poolFile)
    if poolStock is None:
        return stock_list
    # 这里读出来的poolStock是一个list, 但是带有.SH后缀
    ps = [f.split('.')[0] for f in poolStock]
    target = list(set(stock_list) & set(ps))
    return target


def cal_market_value(args, tradeDayList, force_read_file=False, indexName='FOOL'):
    """
    计算大盘涨跌幅
    :param args 运行参数
    :param tradeDayList: 交易日列表
    :param force_read_file: 强制要求从文件中读取(跳过缓存, 因为缓存有可能是不完整的)
    :param indexName : 大盘指数的名称,默认为FOOL表示平均指数
    :return: 市值涨跌幅
    """

    # 找到上一个交易日
    # prev_trade_day = get_real_next_trade_day(tradeDayList[0], False)
    # last_trade_day = tradeDayList[-1]

    prev_trade_day = tradeDayList[0]
    last_trade_day = DT.get_real_next_trade_day(tradeDayList[-1], True)

    if indexName != 'FOOL' and indexName != 'WEIGHT':
        # 指定了大盘指数, 则用对应的大盘指数来计算大盘的价值
        # 注意: 这里的某些大盘有可能是经过加权的!!!!
        vs = G_ALL_STOCK_DATA.read_index(indexName, prev_trade_day)
        ve = G_ALL_STOCK_DATA.read_index(indexName, last_trade_day)
        return round(ve / vs, 4)

    # 先构造所有股票列表
    stock_list = []
    # 获取所有csv文件名
    csv_files = [f for f in os.listdir(g_csv_path) if f.endswith('.csv')]
    for f in csv_files:
        stock_code = f.split('.')[0]
        stock_list.append(stock_code)

    # ADD 20251127 只计算在末尾这个交易日中进入推理的股票池中所有股票在这两个日期之间的等权价值变化
    # 来作为市场大盘的价值
    # SL1 = len(stock_list)
    stock_list = keep_input_pool_stock(stock_list, args, last_trade_day)
    # SL2 = len(stock_list)
    # if SL1 != SL2:
    #     print(f"以区间末尾交易日的推理池股票为基准计算大盘收益,总股票{SL1},推理池股票{SL2}")

    # 读取所有股票的在上个交易日的收盘价
    close_price_t1 = read_group_price(stock_list, 'close', prev_trade_day, force_read_file=force_read_file)
    # print(f"{prev_trade_day}---{len(close_price_t1)}")

    # 读取所有股票在最后一个交易日的收盘价
    close_price_t2 = read_group_price(stock_list, 'close', last_trade_day, force_read_file=force_read_file)
    # print(f"{tradeDayList[-1]}---{len(close_price_t2)}")

    open_stock_list = set(close_price_t1.keys())
    close_stock_list = set(close_price_t2.keys())

    common_stock_list = open_stock_list & close_stock_list
    if len(common_stock_list) == 0:
        raise ValueError("没有找到共同的股票--market value计算失败")
    if indexName == 'FOOL':
        v = 0.
        count = 0
        for stock in common_stock_list:
            if close_price_t1[stock] <= 0.0001 or close_price_t2[stock] <= 0.0001:
                continue
            count += 1
            sv = close_price_t2[stock] / close_price_t1[stock]
            v += sv
        if count == 0:
            raise ValueError("没有找到有效的股票--market value计算失败")
        return round(v / count, 4)
    else:
        # indexName = 'WEIGHT' 加权大盘价值
        dp_stock = {}  # 参与表示大盘的所有股票数据, key=股票代码,value=涨跌幅
        for stock in common_stock_list:
            if close_price_t1[stock] <= 0.0001 or close_price_t2[stock] <= 0.0001:
                continue
            sv = close_price_t2[stock] / close_price_t1[stock] - 1.  # 该股票的涨跌幅
            dp_stock[stock] = sv
        if len(dp_stock) == 0:
            raise ValueError("没有找到有效的股票--market value计算失败")
        dp_stock_list = list(dp_stock.keys())
        mv = read_group_price(dp_stock_list, 'mv', last_trade_day)  # 得到最后一天所有股票的流通市值
        total_value = 0.

        total_mv = sum(mv.values())  # 总的大盘市值
        for stock, svalue in dp_stock.items():
            smv = mv[stock]
            total_value = total_value + svalue * (smv / total_mv)
        return round(total_value, 4) + 1.
