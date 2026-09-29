import copy
import os.path
import platform
import random
import time
from argparse import Namespace
import sys
import numpy as np

sys.path.append('../')
import torch
import torch.nn.functional as F
from torch import optim
from common_module import yy_common as CM
import yy_param_v2 as PM
from torchinfo import summary
import pytorch_lightning as pl
from pm_model import PortfolioModel
from common_module.reader_tools_V11 import DataReader
import torch


def set_seed(args):
    """Set seed for reproducibility."""
    # 获取当前时间,精确到秒
    current_time = int(time.time() * 1000) % (2 ** 31)
    # 获取当前
    random.seed(current_time)
    np.random.seed(current_time + 1)
    torch.manual_seed(current_time + 2)
    if args.useGPU:
        torch.cuda.manual_seed(current_time + 3)
        if args.multi_device:
            torch.cuda.manual_seed_all(current_time + 4)  # if you are using multi-GPU.


def calc_ic(score, target):
    score = (score - score.mean(1, keepdim=True))
    target = (target - target.mean(1, keepdim=True))

    score = score / (score.std(1, keepdim=True) + 1e-8)
    target = target / (target.std(1, keepdim=True) + 1e-8)

    return (score * target).mean()

def rank_ic(score, target):

    score_rank = score.argsort(dim=1).argsort(dim=1).float()
    target_rank = target.argsort(dim=1).argsort(dim=1).float()

    return calc_ic(score_rank, target_rank)


class LightningModel_PM(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.model = PortfolioModel(args)
        self.try_reload_model()

        if self.global_rank == 0:
            if not CM.mark_tmp('check', args.mark):
                CM.mark_tmp('create', args.mark)
                self.model_info = CM.summary_train_info(args, verbose=True)
                summary(self.model, depth=3)

        # 自定义的检查早停方法
        self.stop_checker = CM.EarlyStoppingV2(target_name=['loss', 'ic'],
                                               target_type=['min', 'max'],
                                               patience=args.patience, verbose=True, save_path=args.save_path,
                                               mark=args.mark, save_every_epoch=True, save_every_better=False)
        self.epoch_no_better_count = 0

        '''train'''
        args = Namespace()
        args.ms_data_name = self.args.ms_dataset_name  # 分钟数据集
        args.data_name = self.args.dataset_name
        args.batch_size = self.args.batch_size
        args.minute_before_num = self.args.time_step  # 分钟数据前面的天数
        args.minute_after_num = self.args.episode_len - 1  # 分钟数据后面的天数
        args.day_before_num = self.args.time_step
        args.day_after_num = self.args.pred_len + self.args.episode_len
        args.sicount = self.args.M
        args.shuffle = True
        args.num_workers = 1
        args.stride = self.args.stride
        args.m5_first_only = False  # 当取两个数据集时，只返回5min的第一条记录
        args.chengfen = ''
        args.select_stock_code = []  # 指定股票的交集代码
        args.if_select_input_seq = False  # 返回的input_seq是否使用select_stock_code进行交集
        args.gt_mode = False  # 全局塔模式，ggdq个股等权，gbjq股本加权, 空或者Flase不返回全局塔数据
        args.if_return_all_stock = False  # 是否返回时间交集上全部股票的加权全局塔数据,gt_mode 和 select_stock_code生效时，该参数才会生效
        args.jq_column_index = []  # 加权列索引
        args.D = None  # 股本加权常数，gt_mode = 'gbjq'时该参数生效
        args.jiaoji_mode = self.args.jiaoji_mode
        args.stop_limit = self.args.stop_limit
        self.train_combine_data_reader = DataReader.create_data_reader(args)

        '''valid'''
        args_V = copy.deepcopy(args)
        args_V.stride = self.args.valid_stride
        # args_V.sicount = self.args.val_M
        # args_V.batch_size = self.args.val_B
        self.valid_combine_data_reader = DataReader.create_data_reader(args_V)

        self.cash_price_change = None

        self.cash_dist = [0 for _ in range(11)]

    def try_reload_model(self):
        """
        尝试从保存的模型文件中加载模型参数
        """

        load_path = None
        if self.args.loadcheck != '' and os.path.exists(self.args.loadcheck):
            # 如果指定了loadcheck参数，且文件存在，则尝试从文件中加载模型参数
            load_path = self.args.loadcheck
        else:
            # 自动寻找同名模型文件，并尝试加载
            save_path = os.path.join(self.args.save_path, f"{self.args.mark}.pth")
            if os.path.exists(save_path):
                load_path = save_path
        if load_path is not None:
            state_dict_pre = torch.load(load_path, map_location='cpu', weights_only=True)
            # print(f"{CM.timestr()}****文件中的参数****")
            # print(state_dict_pre.keys())
            state_dict_pre = CM.remove_module_prefix(state_dict_pre)
            loadret = self.model.load_state_dict(state_dict_pre, strict=False)
            print(f"{CM.timestr()}****模型参数加载成功:{load_path}****{loadret}")
            pass


    def forward_an_episode(self, src_data, src_data_min):
        """
        训练一个episode
        :param src_data: 一个batch的数据
        :param current_mode: 当前运行模式
        :return:
        """

        avg_price_index = self.args.price_target_index  # 原数据的均价索引不变

        device = 'cpu'
        LBW = self.args.time_step  # 历史长度
        if self.args.useGPU:
            device = torch.cuda.current_device()
        src_data = src_data.to(device).to(torch.float32)  # [B,M,L,C]
        src_data_m = src_data_min.to(device).to(torch.float32)  # [B,M,80,48,14]

        B, M, L, C = src_data.shape

        v = torch.ones(B).to(device)  # 初始投资组合的价值 [B]
        history_value = [v]
        market_value = []
        models_value = []

        # 循环进行episode_len次前向过程
        weights = F.softmax(torch.randn(B, M), dim=1).to(device)  # 生成随机权重，形状为[B, M]
        w = torch.cat([torch.zeros(B, 1).to(device), weights], dim=-1)
        history_portfolio = [w]
        for i in range(0, self.args.episode_len):
            # 取出前面time_step天的真实天数 # 回看窗口LBW天, 其最末尾一天即为t0日
            # t0日的后一天为t1(即为交易日)
            inputSeq = src_data[:, :, i:i + LBW, :]
            inputSeq_m = src_data_m[:, :, i:i + LBW, :, :]  # [B, M, L, m, C]
            
            # t1日的后一日为t2, 从t2开始的连续pred_len天(亦即patchsize天)为未来信息
            futureSeq = src_data[:, :, i + LBW + 1:i + LBW + 1 + self.args.pred_len, :]  # [B, M, pred_len, C]
            day_t0 = inputSeq[:, :, -1, :].clone()  # t0日的全部数据 [B, M, C]
            day_t1 = src_data[:, :, i + LBW, :].clone()  # t1日的全部数据 [B, M, C]
            day_t2 = futureSeq[:, :, 0, :].clone()  # t2日的全部数据 [B, M, C]
            day_tm = futureSeq[:, :, self.args.ffv_day - 1, :].clone()  # 未来信息第m天, 这个由参数args.ffv_day来指定, 这一天命名为tm

            target_price_t1 = day_t1[:, :, avg_price_index]  # 未来信息第1天(t1)的目标价 [B, M]
            target_price_t2 = day_t2[:, :, avg_price_index]  # 未来信息第2天(t2)的目标价 [B, M]
            target_price_tm = day_tm[:, :, avg_price_index]  # 未来信息第m天(tm)的目标价 [B, M]

            # 取出当前时刻的投资组合权重
            prev_w = history_portfolio[-1]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
               rep, next_w = self.model(inputSeq_m, dev=0.05, benchmark_weights=weights, prev_w=prev_w)

            # 计算下一时刻价格变化率 [B,M], 这里是收益率，在0附近
            next_relative_price = torch.where(target_price_t1 < 1e-3, torch.ones_like(target_price_t2), target_price_t2 / target_price_t1) - torch.ones_like(target_price_t2)
            # label_relative_price = torch.where(target_price_t1 < 1e-3, torch.ones_like(target_price_t2), target_price_tm / target_price_t1) - torch.ones_like(target_price_t2)

            # 大盘价值的计算方法为'FOOL',表示大盘平均权重价值，加1表示价格变化率，回到1附近
            fv_day = (next_relative_price * weights).sum(dim=-1) + 1  # [B]  权重加权的相对价格
            market_value.append(fv_day)  # 大盘价值(当天的)

            # 计算下一时刻的投资组合价值
            next_relative_price = torch.cat([torch.zeros_like(next_relative_price[:, :1]), next_relative_price], dim=1)  # [B,asset_num]
            # label_relative_price = torch.cat([torch.zeros_like(next_relative_price[:, :1]), label_relative_price], dim=1)  # [B,asset_num]

            # 同样对于模型收益，+1回到1附近
            value_1_to_2 = torch.mul(next_w, next_relative_price)
            value_1_to_2 = value_1_to_2.sum(dim=1) + 1  # [B]

            # value_1_to_m = torch.mul(next_w, label_relative_price)
            # value_1_to_m = value_1_to_m.sum(dim=1) + 1  # [B]
            models_value.append(value_1_to_2)

            # 训练和验证模型时, 价值的计算不考虑交易成本(这个数值只用来显示训练过程时使用,不用于损失函数计算)
            history_value.append(value_1_to_2)
            history_portfolio.append(next_w.detach())

        history_value = torch.stack(history_value, dim=1)  # [B,episode_len+1]
        episode_value = history_value.prod(dim=1)  # 最后时刻的投资组合价值
        episode_total_value = episode_value.mean().item()  # 整个Batch的各个episode的平均价值

        # 模型收益
        models_value = torch.stack(models_value, dim=1)  # [B, episode_len]
        # 大盘价值
        market_value = torch.stack(market_value, dim=1)  # [B, episode_len]
        # 计算超额收益率 (主动收益)
        excess_value = models_value - market_value  # [B, episode_len]
        ic = calc_ic(rep, next_relative_price[:, 1:])  # 计算信息系数

        # 计算整个batch中,每个样本的损失的平均值
        if self.args.lossfunc == 'ir':  # loss0831 以IR为唯一损失项
            mean_excess_value = torch.mean(excess_value, dim=1)  # [B]
            tracking_error = torch.std(excess_value, dim=1, unbiased=True)  # [B]
            loss_episode = mean_excess_value / tracking_error
        # elif self.args.lossfunc == 'sr':
        #     mean_value = torch.mean(models_value, dim=1)  # [B]
        #     std = torch.std(models_value, dim=1, unbiased=True)  # [B]
        #     loss_episode = mean_value / std  # [B]
        #     LP1 = loss_episode.mean().item()
        #     LP2 = ic.item()
        #     LP3 = 0

        loss_episode = -loss_episode.mean()  # 同一个batch内所有样本的平均损失,取负值,越小越好

        anaInfo = {
            'ic': ic.item(),
        }

        return loss_episode, episode_total_value, anaInfo

    def training_step(self, batch, batch_idx):
        # 一次训练操作 one batch
        tb = self.trainer.num_training_batches
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        if batch_idx == 0:
            self.start_time = time.time()
            self.train_batch_loss = []
            self.train_batch_value = []
        showbatch = CM.make_fit_showbatch(None, tb)
        batch_loss, batch_value, _ = self.forward_an_episode(batch[0], batch[1])
        avg_loss = self.all_gather(batch_loss).mean().item()
        avg_value = self.all_gather(batch_value).mean().item()
        self.train_batch_loss.append(avg_loss)
        self.train_batch_value.append(avg_value)
        rb = batch_idx + 1
        CM.mark_tmp('clear', self.args.mark)
        if self.trainer.is_global_zero and (rb % showbatch == 0 or rb == tb):
            rtime = time.time() - self.start_time  # 到目前为止的耗时S
            rtime = rtime / rb  # 平均每Batch耗时S
            rtime = round(rtime * (tb - rb))  # 剩余用时
            avg_loss = sum(self.train_batch_loss) / len(self.train_batch_loss)
            avg_value = sum(self.train_batch_value) / len(self.train_batch_value)
            self.output_info(
                f"{CM.timestr()}TrainEpoch:{epoch}/{self.args.train_epochs}:Batch:{rb}/{tb}:loss:{avg_loss:.6f}:value:{avg_value:.6f}:time:{rtime}s")
        return {'loss': batch_loss}

    def on_train_epoch_end(self):
        # 训练集上完成一个epoch
        epoch_loss = sum(self.train_batch_loss) / len(self.train_batch_loss)
        epoch_value = sum(self.train_batch_value) / len(self.train_batch_value)
        self.train_batch_loss = []
        self.train_batch_value = []
        self.output_info(
            f"{CM.timestr()}MPE_Train:{self.trainer.current_epoch + 1 + self.args.epno}/{self.args.train_epochs}:loss:{epoch_loss:.6f}:value:{epoch_value:.6f}")

    def validation_step(self, batch, batch_idx):
        tb = self.trainer.num_val_batches[0]
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        # 一次训练操作 one batch
        if batch_idx == 0:
            self.start_time = time.time()
            self.val_batch_loss = []
            self.val_batch_value = []
            self.val_batch_ic = []
        showbatch = CM.make_fit_showbatch(None, tb)
        batch_loss, batch_value, anaInfo = self.forward_an_episode(batch[0], batch[1])
        ic = torch.tensor(anaInfo['ic'], device=batch_loss.device)
        ic = self.all_gather(ic).mean().item()
        avg_loss = self.all_gather(batch_loss).mean().item()
        avg_value = self.all_gather(batch_value).mean().item()
        self.val_batch_loss.append(avg_loss)
        self.val_batch_value.append(avg_value)
        self.val_batch_ic.append(ic)
        rb = batch_idx + 1
        if self.trainer.is_global_zero and (rb % showbatch == 0 or rb == tb):
            rtime = time.time() - self.start_time  # 到目前为止的耗时S
            rtime = rtime / rb  # 平均每Batch耗时S
            rtime = round(rtime * (tb - rb))  # 剩余用时
            avg_loss = sum(self.val_batch_loss) / len(self.val_batch_loss)
            avg_value = sum(self.val_batch_value) / len(self.val_batch_value)
            avg_ic = sum(self.val_batch_ic) / len(self.val_batch_ic)
            self.output_info(
                f"{CM.timestr()}ValEpoch:{epoch}/{self.args.train_epochs}:Batch:{rb}/{tb}:loss:{avg_loss:.6f}:value:{avg_value:.6f}:ic:{avg_ic:.4f}:time:{rtime}s")

    def on_validation_epoch_end(self):
        # 验证集完整的验证集验证结束后，计算验证集上的平均损失
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        epoch_loss = sum(self.val_batch_loss) / len(self.val_batch_loss)
        epoch_value = sum(self.val_batch_value) / len(self.val_batch_value)
        epoch_ic = sum(self.val_batch_ic) / len(self.val_batch_ic)
        self.val_batch_loss = []
        self.val_batch_value = []
        self.val_batch_ic = []
        addinfo = ''
        if not self.trainer.is_global_zero:
            return
            # 进行早停检查（保存模型）
        get_better = self.stop_checker([epoch_loss, epoch_ic], epoch, self.model, rank=0)
        if get_better:
            self.epoch_no_better_count = 0
            addinfo += "***"
        else:
            self.epoch_no_better_count += 1
            if self.epoch_no_better_count >= self.args.lr_patience:
                # 学习率衰减
                self.adjust_learning_rate_dynamic()
                self.epoch_no_better_count = 0
        if self.stop_checker.early_stop:
            print(f"{CM.timestr()}****Early stopping****")
            self.trainer.should_stop = True

        self.output_info(
            f"{CM.timestr()}MPE_Val:{epoch}/{self.args.train_epochs}:loss:{epoch_loss:.6f}:value:{epoch_value:.6f}:ic:{epoch_ic:.4f}{addinfo}")

    def save_test_result(self, trade_process, last_value, cmark='C0', fvlist=None):
        # 保存测试结果到文件
        # 获取当前训练的进程编号
        rank = self.trainer.global_rank
        # print(f"Rank{rank} save test result--{len(trade_process)}--{len(last_value)}")
        filename = f"./TMP_test_process_{cmark}_{rank}.txt"
        with open(filename, 'a') as file:  # 使用'a'模式以追加方式打开文件
            B = len(trade_process)
            for i in range(B):
                if fvlist is not None:
                    data_list = [trade_process[i], last_value[i], fvlist[i]]
                    file.write(str(data_list) + '\n')  # 将列表转换为字符串，并追加到文件中
                else:
                    data_list = [trade_process[i], last_value[i]]
                    file.write(str(data_list) + '\n')  # 将列表转换为字符串，并追加到文件中

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return self.optimizer

    def adjust_learning_rate_dynamic(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * self.args.lr_decay  # 将学习率调整为原来的75%
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.output_info(f'{CM.timestr()}Updating learning rate to {new_lr:.7f}')

    def output_info(self, info):
        # 输出信息到控制台,但只有主进程才输出
        if self.trainer.is_global_zero:
            print(info)

    def train_dataloader(self):
        d_loader = self.train_combine_data_reader.create_data_loader(self.args.train_date_area[0], self.args.train_date_area[1], refresh=True, stride_shift=True)
        if self.trainer.is_global_zero:
            totalBatchs = len(d_loader)
            print(f"训练集参数配置：{args}")
            print(f"{CM.timestr()}***训练集总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")
        return d_loader

    def val_dataloader(self):
        d_loader = self.valid_combine_data_reader.create_data_loader(self.args.valid_date_area[0], self.args.valid_date_area[1], refresh=True, stride_shift=False)
        if self.trainer.is_global_zero:
            totalBatchs = len(d_loader)
            print(f"训练集参数配置：{args}")
            print(f"{CM.timestr()}***训练集总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")
        return d_loader


if __name__ == "__main__":
    args = PM.get_args(run_mode='train', model='PM_SMV2')
    print(f"{CM.timestr()}***标的:{args.price_target}:{args.src_field_list[args.price_target_index]}")
    if args.RUN_MODE == 'train':
        print(f"{CM.timestr()}***训练集时段:{args.train_date_area}***")
        print(f"{CM.timestr()}***单独验证模式--验证区间:{args.valid_date_area}***")
        if args.epno > 0:
            print(f"{CM.timestr()}***继续训练，从{args.epno + 1}轮开始***")
    if args.useGPU:
        torch.set_float32_matmul_precision('medium')  # 或 'high'，根据你的需求选择
    set_seed(args)

    # 创建模型实例
    pm_model = LightningModel_PM(args)

    # 定义训练器(单机：CPU或者一张或者多张GPU）
    if args.useGPU:
        devices = args.device_list  # 这里的设备列表表示的是使用哪些GPU进行训练，它一定是个list[int]类型
        accelerator = "cuda"
        strategy = 'auto'
    else:
        # 如果没有GPU，使用CPU进行单进程训练
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    # 创建Trainer实例
    trainer = pl.Trainer(num_nodes=args.num_node,
                         logger=False,
                         num_sanity_val_steps=0,
                         max_epochs=[args.train_epochs,1][platform.system()=='Windows'],
                         accelerator=accelerator,
                         devices=devices,
                         enable_progress_bar=False,
                         log_every_n_steps=0,
                         strategy=['ddp_find_unused_parameters_true','auto'][platform.system()=='Windows'],
                         precision='bf16-mixed',
                         accumulate_grad_batches=args.agb,
                         reload_dataloaders_every_n_epochs=1,
                         limit_train_batches=[None,1][platform.system()=='Windows'],
                         limit_val_batches=[None,1][platform.system()=='Windows'],
                         )
    if args.RUN_MODE == 'train':
        trainer.fit(model=pm_model)
        print(f"{CM.timestr()}TrainingDone.")
