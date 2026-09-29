import os
import torch
from common_module import yy_common as CM
import argparse
from enum import Enum
import copy


class Period(Enum):
    FIVE_MINUTE = 'five_minute'
    DAY = 'day'


class TaskType(Enum):
    PM = 'PM'
    Classification = 'classification'


def get_args(run_mode='train', episode_len=20, model='PM_SMV2', loadcheck='', mark='', M=20):
    parser = argparse.ArgumentParser(description='Portfolio Parameter')
    parser.add_argument('--num_node', type=int, default=1, help='lt训练时使用的节点数')
    parser.add_argument('--gpu_num', type=int, default=0, help='使用的卡数')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--model', type=str, default=model, help='当前训练的模型')
    parser.add_argument('--price_target', type=str, default='day', help='价格目标[open,935,5m,60m,day,day_hfq]')
    parser.add_argument('--price_layer', type=int, default=0, help='')
    parser.add_argument('--double_val', type=int, default=0, help='是否使用双验证')
    parser.add_argument('--use_prev_w', type=int, default=1, help='资产组合模型中是否使用前一个W')
    parser.add_argument('--w_factor', type=float, default=1, help='softmax的缩放系数')
    parser.add_argument('--firstWeight', type=str, default='RAND', help='权重初始化方法[AVG,CASH,RAND]')
    parser.add_argument('--RUN_MODE', type=str, default=run_mode, help='当前训练模式[train,check]')
    parser.add_argument('--mark', type=str, default=mark, help='name str')
    parser.add_argument('--loadcheck', type=str, default=loadcheck, help='load check point')
    parser.add_argument('--save_path', type=str, default="./output", help='vali data file')
    parser.add_argument('--epno', type=int, default=0, help='训练起始轮数，用于中断后接续训练')
    parser.add_argument('--use_w_loss', type=int, default=0, help='是否使用w_loss')
    parser.add_argument('--w_loss_factor', type=float, default=5., help='w_loss的系数')
    parser.add_argument('--w_sharp_factor', type=int, default=50, help='w_loss的sharpness')
    parser.add_argument('--dnv', type=int, default=0, help='数据归一处理版本0/1')
    parser.add_argument('--ad_info', type=int, default=0, help='是否使用AD信息')
    
    # forecasting task
    parser.add_argument('--time_step', type=int, default=80, help='input sequence length')
    parser.add_argument('--patch_size', type=int, default=5, help='patch size')
    parser.add_argument('--patch_stride', type=int, default=5, help='patch stride')
    parser.add_argument('--pred_len', type=int, default=10, help='prediction length')
    parser.add_argument('--episode_len', type=int, default=episode_len, help='trade days per episode')
    parser.add_argument('--decay_type', type=int, default=0, help='0:无衰减,1:单步预测未来多日收益衰减,2:多步预测未来多日衰减')
    parser.add_argument('--decay_factor_1', type=float, default=0.9, help='价值衰减率')
    parser.add_argument('--decay_factor_2', type=float, default=0.95, help='价值衰减率')
    parser.add_argument('--tr_days', type=int, default=1, help='调仓周期天数')
    parser.add_argument('--stride', type=int, default=20, help='stride')
    parser.add_argument('--stride_val', type=int, default=3, help='stride')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--ci', type=int, default=1, help='是否使用各通道独立编码[0,1]')
    parser.add_argument('--DS', type=str, default='day395', help="训练数据集名称")
    parser.add_argument('--data_norm', type=int, default=1, help="是否对数据进行归一化")
    parser.add_argument('--task', type=str, default='PM', help='任务名称')
    parser.add_argument('--pf', type=float, default=0.98, help='涨跌停判定系数')
    parser.add_argument('--risk_factor', type=float, default=0.01, help='风险系数')
    parser.add_argument('--cost_factor', type=float, default=0.001, help='交易成本训练参数')
    parser.add_argument('--tc1', type=float, default=0.001, help='交易成本参数1-测试用')
    parser.add_argument('--tc2', type=float, default=0.003, help='交易成本参数1-测试用')
    parser.add_argument('--vf', type=float, default=0.55, help="计算value时v(t1,t2)的系数")
    parser.add_argument('--count_cost', type=int, default=0, help='推理时是否计算交易成本')
    parser.add_argument('--M', type=int, default=M, help='资产组合中的股票数量')
    parser.add_argument('--maxw', type=int, default=0, help='最大权重,默认0表示不限制--模型内使用')
    parser.add_argument('--minw', type=float, default=0.001, help='最小权重,默认0.001,推理时模型外使用,用于smooth')
    parser.add_argument('--G', type=int, default=0, help='推理时使用分组方法,G为分组数量')
    parser.add_argument('--K', type=int, default=1, help='推理时使用分组方法,每组取TOP-K只股票')
    parser.add_argument('--lm', type=int, default=0, help='推理时使用,是否用省内存推理')
    parser.add_argument('--foolbuy', type=int, default=0, help='推理时使用,是否进行平均交易策略')
    parser.add_argument('--noise', type=int, default=0, help='推理时使用,是否用省内存推理')
    parser.add_argument('--noise_std', type=float, default=0.1, help='推理时使用,是否用省内存推理')

    parser.add_argument('--global_stock_space', type=int, default=6000, help='SM模型-全局股票数')
    parser.add_argument('--sm_stock', type=int, default=1, help='SM模型-全局股票数')
    parser.add_argument('--sm_token', type=int, default=16, help='SM模型-全局token数量')
    parser.add_argument('--sm_info_usage', type=str, default='ADD', help='[ADD,MSA]')

    # 与crossformer结构相关的几个参数
    parser.add_argument('--layers', type=int, default=3, help='cross layers')
    parser.add_argument('--blocks', type=int, default=1, help='cross layers')
    parser.add_argument('--router_factor', type=int, default=10, help='cross router')
    parser.add_argument('--perceiver_depth', type=int, default=3, help='perceiver layers')
    parser.add_argument('--big', type=float, default=0.02,
                        help='测试PM模型(单次测试)时使用，统计每天持仓比例在big以上的数量')
    parser.add_argument('--ftd', type=str, default='',
                        help='测试PM模型时使用，设定起始交易日--20240902,这个参数优先权最高，如有设定则后面两个参数根据ftd和time_step计算')
    parser.add_argument('--cfd', type=str, default='20241001', help='测试PM模型时使用，设定测试数据集的范围起点')
    parser.add_argument('--cld', type=str, default='20250630', help='测试PM模型时使用，设定测试数据集的范围终点')
    parser.add_argument('--checktimes', type=int, default=50, help='测试样本数')
    parser.add_argument('--show_trade_list', type=int, default=0, help='只用于详细测试时,输出交易日历')
    parser.add_argument('--transfer_xls', type=int, default=1, help='只用于详细测试时,是否要转换为excel工作表')
    parser.add_argument('--pp_adjust_method', type=str, default='stock', help='PP调整方法[none,cash,stock]')
    parser.add_argument('--dvplus', type=float, default=0.03,
                        help='输出调仓纪录时使用,日收益率大于该值的日期,将输出具体收益来源')
    parser.add_argument('--buffdays', type=int, default=20,
                        help='按月测试时为保证有足够样本量,将时间窗口后移buffdays天')

    # optimization
    parser.add_argument('--fvtype', type=str, default='FOOL', help='定义用什么做大盘收益，FOOL， ZZ1000， weight')
    parser.add_argument('--train_with_noise', type=int, default=1, help='带噪音的训练')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.985, help='学习率衰减率')
    parser.add_argument('--lr_patience', type=int, default=3, help='连续几个epoch未优化时，学习率衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--GPU', type=str, default='A',
                        help='[a|A|0,1,2..], A表示全部，a表示除最后一个卡之外的卡，0,1,2...表示具体卡号')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--kfold', type=int, default=0, help='k-fold training, k为序号, 默认为0表示不使用k-fold')

    # 5分钟数据分支
    parser.add_argument('--use_minute', type=int, default=1, help='是否加入5分钟数据分支[0,1]')
    parser.add_argument('--ms_time_step', type=int, default=20, help='5分钟数据回看天数(取日线回看窗口的最后ms_time_step天), 须<=time_step')
    parser.add_argument('--ms_patch_size', type=int, default=12, help='分钟patch大小(bar数), 12=1天4个token(每小时1个, 可看到日内结构), 48=1天1个token')
    parser.add_argument('--ms_d_model', type=int, default=64, help='分钟分支隐藏层维度')
    parser.add_argument('--ms_layers', type=int, default=2, help='分钟分支编码层数')
    parser.add_argument('--ms_stock_attn', type=int, default=1, help='分钟分支是否做股票方向自注意力[0,1]')
    parser.add_argument('--ms_drop_calendar', type=int, default=1, help='分钟数据是否剔除gen_year/month/day等日历字段[0,1]')
    parser.add_argument('--debug_rep', type=int, default=20, help='每N次模型前向打印一次日线/分钟表征统计(约每个episode一次), 0为关闭')

    args = parser.parse_args()
    args.double_val = args.double_val > 0
    # if args.use_w_loss == 1:
    #     args.use_prev_w = 1
    if args.gpu_num == 7:
        args.GPU = 'a'
    elif args.gpu_num == 8:
        args.GPU = 'A'
    if args.model == 'PP_SM':
        args.task = 'classification'
    if args.stride_val == 0:
        args.stride_val = args.stride

    main_path = os.path.dirname(os.path.abspath(__file__))
    if args.loadcheck.startswith('./'):
        args.loadcheck = os.path.join(main_path, args.loadcheck[2:])
    if args.save_path.startswith('./'):
        args.save_path = os.path.join(main_path, args.save_path[2:])

    if args.loadcheck == '' and args.RUN_MODE == 'train' and args.mark != '' and args.epno > 0:
        args.loadcheck = os.path.join(str(args.save_path), f"{args.mark}.pth")

    # 部分模型参数可以保存到mark同名文件中，方便未来在加载模型时使用
    # 注意，下面的参数无论是做断点续训还是finetune或者test，都是需要与被加载的模型参数保持一致的，才进行保存和加载
    # 而其他参数可能随着任务不同，需要从命令行参数传入
    args = try_load_mark_args(args, args.loadcheck)

    args.m_stock_list = None
    # args.num_class_value = [-0.0289, -0.0159, -0.0082, -0.002, 0.0, 0.0058, 0.0127, 0.0236, 0.0554]  # 只看1天收盘价区间点位
    # args.num_class_value = [-0.0261, -0.0076, 0.0, 0.0075, 0.0146, 0.0235, 0.0359, 0.0561, 0.1048]
    args.num_class_value = [-0.02, -0.01, 0.0, 0.01, 0.02]
    args.num_class = len(args.num_class_value) + 1
    args.expect_class_value = make_expect_class_value(args.num_class_value)

    if args.model.startswith('PP_'):
        args.task = 'classification'

    args.limit_mark_type = 10  # 股票涨跌停限制类别数
    args.limit_mark_base = 1  # 股票涨跌停限制类别数的基数

    # 5分钟数据集
    args.ms_dataset_name = 'xt_260527_14f'
    args.ms_scaler_info_path = '/data/yy_data/five_minute_data/xt_260527_14f/scaler_info.txt'
    if args.use_minute:
        assert 0 < args.ms_time_step <= args.time_step, "ms_time_step 必须在 (0, time_step] 内"
        assert (args.ms_time_step * 48) % args.ms_patch_size == 0, "ms_time_step*48 必须能被 ms_patch_size 整除"
    # args.train_date_area = [20100101, 20221231]
    # args.valid_date_area = [20230101, 20240923]

    if args.model.startswith('PM_') and args.RUN_MODE == 'check' and args.ftd != '':
        # 测试PM模型时，如果有设定起始交易日
        args.cfd = CM.make_pre_date(args.ftd, args.time_step + 1)
        sample_per_day = int(5000 // args.M) + 1  # 假定每天有5000只股票在活跃,按M只股票来分组,则每天大约有这么多组
        # 为了保证在做批量测试时有足够多的样本,需要把结束日期向后推若干天,以保证样本量, 假定最少需要checktimes个样本
        buffer_days = args.checktimes // sample_per_day
        if args.model == 'PM_SMV2':
            buffer_days = max(buffer_days, 4)  # 至少要预测1天
        args.cld = CM.get_next_trade_day(args.ftd, args.episode_len + buffer_days)
    args.firstChannelIsID = True
    args.test_date_area = [int(args.cfd), int(args.cld)]
    args.train_date_area = [20100101, 20171231]
    args.valid_date_area = [20180101, 20221231]

    if args.kfold > 0:
        args.double_val = True
        k_fold_dates = [
            [[20100101, 20151231], [20151001, 20171231], [20171001, 20191231]],
            [[20151001, 20171231], [20171001, 20191231], [20191001, 20211231]],
            [[20171001, 20191231], [20191001, 20211231], [20211001, 20231231]],
            [[20191001, 20211231], [20211001, 20231231], [20231001, 20250630]],
            [[20211001, 20231231], [20231001, 20250630], [20231001, 20250630]],
            [[20221001, 20241231], [20231001, 20250630], [20231001, 20250630]]  # no use
        ]
        assert args.kfold <= len(k_fold_dates)
        args.train_date_area, args.valid_date_area, args.test_date_area = k_fold_dates[args.kfold - 1]
        print(f"K-fold trainging, K=#{args.kfold}")

    args.pre_known_future = 8  # 不算ts_code, 未来数据的前多少列是已知的
    if args.DS == 'day111':
        args.DS_flag = 'ts_orig_111f_zh'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'news_macro_score','gz2000_iter', 'gz2000_one', 'hs300_iter', 'hs300_one' , 'avg_price_hfq' , 'is_suspend', 'st',
                               'limit_mark', 'gen_padding_flag']  # 106 channels
        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        args.input_channels.remove('ts_code ')
        args.mv_index = CM.find_str_in_list(args.input_channels, 'circ_mv')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq'),
                               CM.find_str_in_list(args.input_channels, 'avg_price_hfq')
                               ]  # '价格类字段的列索引'

    elif args.DS == 'day106':
        args.DS_flag = 'ts_260201_106f'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'limit_mark', 'gen_padding_flag']  # 106 channels
        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        args.input_channels.remove('ts_code ')
        args.mv_index = CM.find_str_in_list(args.input_channels, 'circ_mv')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    elif args.DS == 'day112':
        args.DS_flag = 'ts_260327_112f'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'news_macro_score',  "hs300_weight", "zz500_weight", "zz1000_weight", "gz2000_weight", "zz2000_weight", "zz800_weight", 'is_suspend', 'st',
                               'limit_mark', 'gen_padding_flag']  # 106 channels
        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        args.input_channels.remove('ts_code ')
        args.mv_index = CM.find_str_in_list(args.input_channels, 'circ_mv')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    elif args.DS == 'day134':
        args.DS_flag = 'ts_260413_134f'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'news_macro_score',  "hs300_weight", "zz500_weight", "zz1000_weight", "gz2000_weight", "zz2000_weight", "zz800_weight", "exp_size", "exp_non_linear_size", "exp_momentum", "exp_liquidity", "exp_book_to_price", "exp_leverage", "exp_growth", "exp_earnings_yield", "exp_beta", "exp_residual_volatility", "exp_comovement", "ret_size", "ret_non_linear_size", "ret_momentum", "ret_liquidity", "ret_book_to_price", "ret_leverage", "ret_growth", "ret_earnings_yield", "ret_beta", "ret_residual_volatility", "ret_comovement",'is_suspend', 'st',
                               'limit_mark', 'gen_padding_flag']  # 106 channels
        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        args.input_channels.remove('ts_code ')
        args.mv_index = CM.find_str_in_list(args.input_channels, 'circ_mv')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    elif args.DS == 'day196':
        args.DS_flag = 'ts_260525_196f'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'news_macro_score',  "hs300_weight", "zz500_weight", "zz1000_weight", "gz2000_weight", "zz2000_weight", "zz800_weight", "exp_size", "exp_non_linear_size", "exp_momentum", "exp_liquidity", "exp_book_to_price", "exp_leverage", "exp_growth", "exp_earnings_yield", "exp_beta", "exp_residual_volatility", "exp_comovement", 'exp_gangtie', 'exp_feiyinjinrong', 'exp_qiche', 'exp_jiaotongyunshu', 'exp_jianzhucailiao', 'exp_shipinyinliao', 'exp_gongyongshiye', 'exp_yousejinshu', 'exp_zonghe', 'exp_tongxin', 'exp_fangdichan', 'exp_fangzhifushi', 'exp_jixieshebei', 'exp_qinggongzhizao', 'exp_shangmaolingshou', 'exp_nonglinmuyu', 'exp_chuanmei', 'exp_huanbao', 'exp_jichuhuagong', 'exp_dianlishebei', 'exp_dianzi', 'exp_guofangjungong', 'exp_yinhang', 'exp_shiyoushihua', 'exp_jisuanji', 'exp_jianzhuzhuangshi', 'exp_shehuifuwu', 'exp_meironghuli', 'exp_jiayongdianqi', 'exp_yiyaoshengwu', 'exp_meitan', 'ret_size', 'ret_non_linear_size', 'ret_momentum', 'ret_liquidity', 'ret_book_to_price', 'ret_leverage', 'ret_growth', 'ret_earnings_yield', 'ret_beta', 'ret_residual_volatility', 'ret_comovement', 'ret_gangtie', 'ret_feiyinjinrong', 'ret_qiche', 'ret_jiaotongyunshu', 'ret_jianzhucailiao', 'ret_shipinyinliao', 'ret_gongyongshiye', 'ret_yousejinshu', 'ret_zonghe', 'ret_tongxin', 'ret_fangdichan', 'ret_fangzhifushi', 'ret_jixieshebei', 'ret_qinggongzhizao', 'ret_shangmaolingshou', 'ret_nonglinmuyu', 'ret_chuanmei', 'ret_huanbao', 'ret_jichuhuagong', 'ret_dianlishebei', 'ret_dianzi', 'ret_guofangjungong', 'ret_yinhang', 'ret_shiyoushihua', 'ret_jisuanji', 'ret_jianzhuzhuangshi', 'ret_shehuifuwu', 'ret_meironghuli', 'ret_jiayongdianqi', 'ret_yiyaoshengwu', 'ret_meitan', 'is_suspend', 'st', 'limit_mark', 'gen_padding_flag']  # 196 channels


        #['ts_code', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry', 'gen_year', 'gen_month', 'gen_day', 'gen_week', 'gen_cq', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'volume', 'amount', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount', 'net_mf_vol', 'net_mf_amount', 'up_limit', 'down_limit', 'call_auction_close', 'call_auction_open', 'call_auction_high', 'call_auction_low', 'call_auction_vol', 'call_auction_amount', 'call_auction_vwap', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'adj_factor', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 'kdj_bfq', 'kdj_d_bfq', 'kdj_k_bfq', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250', 'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq', 'macd_dea_bfq', 'macd_dif_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq', 'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 'wr_bfq', 'wr1_bfq', 'XIN9', 'HSI', 'HKAH', 'DJI', 'SPX', 'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'news_micro_score', 'news_macro_score', 'hs300_weight', 'zz500_weight', 'zz1000_weight', 'gz2000_weight', 'zz2000_weight', 'zz800_weight', 'exp_size', 'exp_non_linear_size', 'exp_momentum', 'exp_liquidity', 'exp_book_to_price', 'exp_leverage', 'exp_growth', 'exp_earnings_yield', 'exp_beta', 'exp_residual_volatility', 'exp_comovement', 'exp_gangtie', 'exp_feiyinjinrong', 'exp_qiche', 'exp_jiaotongyunshu', 'exp_jianzhucailiao', 'exp_shipinyinliao', 'exp_gongyongshiye', 'exp_yousejinshu', 'exp_zonghe', 'exp_tongxin', 'exp_fangdichan', 'exp_fangzhifushi', 'exp_jixieshebei', 'exp_qinggongzhizao', 'exp_shangmaolingshou', 'exp_nonglinmuyu', 'exp_chuanmei', 'exp_huanbao', 'exp_jichuhuagong', 'exp_dianlishebei', 'exp_dianzi', 'exp_guofangjungong', 'exp_yinhang', 'exp_shiyoushihua', 'exp_jisuanji', 'exp_jianzhuzhuangshi', 'exp_shehuifuwu', 'exp_meironghuli', 'exp_jiayongdianqi', 'exp_yiyaoshengwu', 'exp_meitan', 'ret_size', 'ret_non_linear_size', 'ret_momentum', 'ret_liquidity', 'ret_book_to_price', 'ret_leverage', 'ret_growth', 'ret_earnings_yield', 'ret_beta', 'ret_residual_volatility', 'ret_comovement', 'ret_gangtie', 'ret_feiyinjinrong', 'ret_qiche', 'ret_jiaotongyunshu', 'ret_jianzhucailiao', 'ret_shipinyinliao', 'ret_gongyongshiye', 'ret_yousejinshu', 'ret_zonghe', 'ret_tongxin', 'ret_fangdichan', 'ret_fangzhifushi', 'ret_jixieshebei', 'ret_qinggongzhizao', 'ret_shangmaolingshou', 'ret_nonglinmuyu', 'ret_chuanmei', 'ret_huanbao', 'ret_jichuhuagong', 'ret_dianlishebei', 'ret_dianzi', 'ret_guofangjungong', 'ret_yinhang', 'ret_shiyoushihua', 'ret_jisuanji', 'ret_jianzhuzhuangshi', 'ret_shehuifuwu', 'ret_meironghuli', 'ret_jiayongdianqi', 'ret_yiyaoshengwu', 'ret_meitan', 'is_suspend', 'st', 'limit_mark', 'gen_padding_flag']
        
        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        args.input_channels.remove('ts_code ')
        args.mv_index = CM.find_str_in_list(args.input_channels, 'circ_mv')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    elif args.DS == 'day104':
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'is_suspend', 'st',
                               'limit_mark', 'gen_padding_flag']  # 102 channels

        # =================真正送入模型的字段列表===============#
        args.input_channels = [  # 'ts_code ',
            'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
            'IXIC', '35_close', 'avg_price', '60m_avg_price', 'daily_avg_price', 'is_suspend', 'st',
            'limit_mark', 'gen_padding_flag']

        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, '60m_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'daily_avg_price'),
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    elif args.DS == 'day395': # 替换
        args.src_field_list = ["f_d_code_int", "f_d_code_id", "f_d_exchange_id", "f_d_market_id", "f_d_industry_id", "f_d_year", 
                                "f_d_month", "f_d_day", "f_d_week", "f_s_is_dividend", "f_p_open", "f_p_high", "f_p_low", "f_p_close", 
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
                                "f_i_news_macro", "f_s_is_suspend", "f_s_is_st", "f_w_hs300", "f_w_csi500", "f_w_csi1000", "f_w_gz2000", 
                                "f_w_csi2000", "f_w_csi800", "f_w_csi_all", "f_s_limit_mark", "f_s_padding_flag", "f_i_exp_size", 
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
                                "f_i_ret_jiayongdianqi", "f_i_ret_yiyaoshengwu", "f_i_ret_meitan"]  # 395 channels

        # =================真正送入模型的字段列表===============#
        args.input_channels = copy.deepcopy(args.src_field_list)
        #args.input_channels.remove("f_d_code_int")
        args.mv_index = CM.find_str_in_list(args.input_channels, 'f_i_mv_circ')
        args.vol_index = CM.find_str_in_list(args.input_channels, 'f_v_volume')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'f_v_amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'f_s_is_dividend')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'f_p_open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'f_p_close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'f_p_high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'f_p_low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'f_p_pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'f_p_open5min_avg')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, 'f_p_open5min_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, 'f_p_open1h_avg'),
                               CM.find_str_in_list(args.input_channels, 'f_p_avg_1d'),
                               CM.find_str_in_list(args.input_channels, 'f_p_open_adj'),
                               CM.find_str_in_list(args.input_channels, 'f_p_high_adj'),
                               CM.find_str_in_list(args.input_channels, 'f_p_low_adj'),
                               CM.find_str_in_list(args.input_channels, 'f_p_close_adj')
                               ]  # '价格类字段的列索引'
        '''
        args.priceRelatedField = ['f_p_open', 'f_p_high', 'f_p_low', 'f_p_close', 'f_p_pre_close', 'f_p_avg',
                                      'f_p_open_adj', 'f_p_high_adj', 'f_p_low_adj', 'f_p_close_adj', 'f_p_pre_close_adj',
                                      'f_p_limit_up', 'f_p_limit_down', 'f_p_high_52w', 'f_p_low_52w', 'f_p_high_52w_adj', 'f_p_low_52w_adj',
                                      'f_p_call_open', 'f_p_call_close',
                                      'f_p_open5min_close', 'f_p_open5min_avg', 'f_p_open1h_avg', 'f_p_avg_1d']
        args.mask_channel = ['f_d_code_int','f_d_code_id','f_i_news_micro','f_i_news_macro','f_w_hs300','f_w_csi500','f_w_csi1000','f_w_gz2000','f_w_csi2000','f_w_csi800','f_w_csi_all']
        '''
        
    else:  # 'day102'
        args.src_field_list = ['ts_code ', 'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
                               'IXIC', '35_close', 'avg_price', 'is_suspend', 'st',
                               'limit_mark', 'gen_padding_flag']  # 102 channels

        # =================真正送入模型的字段列表===============#
        args.input_channels = [  # 'ts_code ',
            'gen_natural_id', 'gen_exchange', 'gen_market', 'gen_industry',
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
            'IXIC', '35_close', 'avg_price', 'is_suspend', 'st',
            'limit_mark', 'gen_padding_flag']

        args.vol_index = CM.find_str_in_list(args.input_channels, 'vol')  # '成交量的列索引'
        args.amount_index = CM.find_str_in_list(args.input_channels, 'amount')  # '成交额的列索引'
        args.cq_index = CM.find_str_in_list(args.input_channels, 'gen_cq')  # '成交量的列索引'
        args.open_price_index = CM.find_str_in_list(args.input_channels, 'open')  # '开盘价的列索引'
        args.close_price_index = CM.find_str_in_list(args.input_channels, 'close')  # '收盘价的列索引'
        args.high_price_index = CM.find_str_in_list(args.input_channels, 'high')  # '最高价的列索引'
        args.low_price_index = CM.find_str_in_list(args.input_channels, 'low')  # '最低价的列索引'
        args.pre_close_price_index = CM.find_str_in_list(args.input_channels, 'pre_close')  # '前收盘价的列索引'
        args.first5m_avg_price_index = CM.find_str_in_list(args.input_channels, 'avg_price')  # '开盘后首个5分钟均价的列索引'
        args.p35_close_index = CM.find_str_in_list(args.input_channels, '35_close')  # '35分钟收盘价的列索引'
        args.price_channels = [args.open_price_index,
                               args.close_price_index,
                               args.high_price_index,
                               args.low_price_index,
                               args.pre_close_price_index,
                               args.first5m_avg_price_index,
                               args.p35_close_index,
                               CM.find_str_in_list(args.input_channels, 'open_hfq'),
                               CM.find_str_in_list(args.input_channels, 'high_hfq'),
                               CM.find_str_in_list(args.input_channels, 'low_hfq'),
                               CM.find_str_in_list(args.input_channels, 'close_hfq')
                               ]  # '价格类字段的列索引'
    if args.DS == 'day102':
        assert args.price_target in ['935', '5m', 'open'], 'day102数据集中，price_target只能为935或5m或open'
        pass

    if args.price_target == '935':  # 9:35分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open5min_close')
    elif args.price_target == '5m':  # 前5分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open5min_avg')
    elif args.price_target == '60m':  # 前60分钟均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_open1h_avg')
    elif args.price_target == 'day':  # 全天均价
        args.price_target_index = CM.find_str_in_list(args.src_field_list, 'f_p_avg')
    #elif args.price_target == 'day_hfq':  # 全天均价
        #args.price_target_index = CM.find_str_in_list(args.input_channels, 'f_p_avg_1d')
    elif args.price_target == 'close_hfq':  # 全天均价
        args.price_target_index = CM.find_str_in_list(args.input_channels, 'f_p_close_adj')
    else:  # 默认使用开盘价
        args.price_target_index = args.open_price_index

    if args.price_layer == 1:
        args.price_layer = args.price_target_index

    args.channels = len(args.input_channels)

    if args.task == 'classification':
        print(f"Classification task for:", args.num_class_value)

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


def cal_seq_len(args):
    """
    计算要进行一次推理所需要的最小序列长度,只对PM模型有效
    :param args: 参数
    :return: 序列长度
    """

    # ----PM模型----
    # 序列长度: 回看窗口(80) + episode_len(10) + 预测窗口即未来信息(5)
    ml = args.time_step + args.episode_len + args.pred_len  # 输入序列长度
    return ml


def try_load_mark_args(args, load_model_file):
    model_base_args = ["model", "time_step", "d_model", "risk_factor", "dnv", "price_layer", "data_norm", "maxw", "vf",
                       "price_target", "DS",
                       "global_stock_space", "sm_token", "sm_info_usage",
                       "patch_size", "patch_stride", "use_prev_w", "firstWeight", "ci",
                       "layers", "perceiver_depth", "blocks", "router_factor",
                       "use_minute", "ms_time_step", "ms_patch_size", "ms_d_model", "ms_layers", "ms_stock_attn",
                       "ms_drop_calendar"]
    if load_model_file != '':
        # 加载模型基础参数
        base_args = CM.load_base_args(load_model_file)
        if base_args is not None:
            for base_arg_name in base_args.keys():
                if base_arg_name in model_base_args:
                    setattr(args, base_arg_name, base_args[base_arg_name])
    return args


def make_expect_class_value(num_class_value: list):
    """
    制作期望分类值
    :param num_class_value: 分类数量列表
    :return: 期望分类值列表
    """
    expect_class_value = []
    C = len(num_class_value)
    expect_class_value.append(round(num_class_value[0] * 1.2, 4))
    for i in range(C - 1):
        expect_class_value.append(round((num_class_value[i] + num_class_value[i + 1]) / 2.0, 4))
    expect_class_value.append(round(num_class_value[-1] * 1.2, 4))
    return expect_class_value


def make_price_relative_v2(prices, price_channels: list, t1_open):
    """
    生成价格变化向量, 将所有价格类字段都除以T1的开盘价
    :param prices: [B, M, L, C]  M个资产的价格序列
    :param price_channels: 价格序列的通道数
    :param t1_open: 最后的开盘价[B,M]
    :return: [B,M, L, C]  M个资产的价格变化序列
    """

    B, M, L, C = prices.shape

    # 扩展维度
    t1_open = t1_open.view(B, M, 1).clone()

    # 计算价格占比
    for pos in price_channels:
        prices[:, :, :, pos] = prices[:, :, :, pos] / t1_open  # [B, M, L, C]
    return prices


def make_price_relative_v3(a, day_t0):
    """
    对回看窗口数据中除了前m列和后n列之外的价格字段除以t1日的对应字段
    :param a: [B, M, L, C]  M个资产的价格序列
    :param day_t0: t0日全部通道的数据[B,M,C]
    :return: [B,M, L, C]  M个资产的数据变化序列
    """
    B, M, L, C = a.shape

    # 将 day_t1 扩展为 [B, M, L, C]
    day_t0 = day_t0.view(B, M, 1, C).repeat(1, 1, L, 1)
    a[:, :, :, 8] = 0.

    # 确定需要处理的列范围
    # 需要处理的列索引从 m 到 C - n - 1（不包括 C - n）
    start = 14
    end = C - 2  # 切片时 end 是开区间，所以实际处理到 C - n -1

    # 使用 slice(start, end) 选择中间列
    columns_to_process = slice(start, end)

    # 对中间列进行除法操作
    a[:, :, :, columns_to_process] = a[:, :, :, columns_to_process] / (day_t0[:, :, :, columns_to_process] + 1e-8)

    return a


def make_price_relative(prices, price_channels: list, base_index):
    """
    生成价格变化向量, 将所有价格类字段都除以最后一天的收盘价
    :param prices: [B, M, L, C]  M个资产的价格序列
    :param price_channels: 价格序列的通道数
    :param base_index: 价格序列的基准指标的索引下标
    :return: [B,M, L, C]  M个资产的价格变化序列
    """

    B, M, L, C = prices.shape
    # C =4, 表示open,close,high,low四种价格数据

    # 获取最后一天的收盘价
    last_day_prices = (prices[:, :, -1, base_index])  # [B, M]
    # 扩展维度
    last_day_prices = last_day_prices.view(B, M, 1).clone()

    # 计算价格占比
    for pos in price_channels:
        prices[:, :, :, pos] = prices[:, :, :, pos] / last_day_prices  # [B, M, L, C]
    return prices


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
