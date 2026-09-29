import copy
import json
import os.path
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
import yy_param_ms as PM
from torchinfo import summary
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from pm_model_ms import PortfolioModel_SMV2
# V11: 同时读取日线与5分钟数据(ms_data_name为空时退化为纯日线reader)
from common_module.reader_tools_V11 import DataReader
#from common_module.data_utils import DataNormalizer_SWSJ
from torch.utils.data import ConcatDataset
import torch.nn as nn
import os
import warnings

warnings.filterwarnings('ignore')



@torch.no_grad()
def cs_clip_return_2sigma(
    future_return: torch.Tensor,
    valid_mask: torch.Tensor = None,
    n_sigma: float = 2.0,
    eps: float = 1e-8,
):
    """
    T+1 股票收益横截面 2σ Winsorization。

    参数
    ----
    future_return: [B, M]
        当天 M 只股票的 T+1 真实收益，小数形式。
    valid_mask: [B, M]
        True 表示该股票有有效的未来收益标签.
    n_sigma:
        截断标准差倍数，实验 B 固定为 2.0。
    返回
    ----
    clipped_return: [B, M]
        截断后的股票收益，无效位置设为 0。
    metrics:
        横截面 Std 和截断比例等统计信息。
    """

    assert future_return.ndim == 2

    B, M = future_return.shape

    if valid_mask is None:
        valid_mask = torch.ones(
            (B, M),
            dtype=torch.bool,
            device=future_return.device,
        )
    else:
        assert valid_mask.shape == future_return.shape
        valid_mask = valid_mask.bool()

    # 非有限收益不能参与横截面统计
    valid_mask = (
        valid_mask
        & torch.isfinite(future_return)
    )

    # ========================================
    # 1. 无效收益归零
    # ========================================

    safe_return = torch.where(
        valid_mask,
        future_return,
        torch.zeros_like(future_return),
    )

    valid_count = valid_mask.sum(
        dim=1,
        keepdim=True,
    )

    count = valid_count.clamp_min(1)

    # ========================================
    # 2. 横截面 Mean
    # ========================================

    cs_mean = (
        safe_return.sum(dim=1, keepdim=True)
        / count
    )

    # ========================================
    # 3. 横截面 Std
    # ========================================

    centered = torch.where( valid_mask, safe_return - cs_mean,torch.zeros_like(safe_return), )
    cs_var = (centered.square().sum(  dim=1, keepdim=True, ) / count)
    cs_std = torch.sqrt( cs_var.clamp_min(0.0) )

    # ========================================
    # 4. 计算截断区间
    # ========================================
    lower_bound = ( cs_mean - n_sigma * cs_std)
    upper_bound = (  cs_mean + n_sigma * cs_std)
    # ========================================
    # 5. 对收益进行截断
    # ========================================

    clipped_return = torch.maximum(
        torch.minimum(
            safe_return,
            upper_bound,
        ),
        lower_bound,
    )

    # 无效收益标签设为 0
    clipped_return = torch.where(
        valid_mask,
        clipped_return,
        torch.zeros_like(clipped_return),
    )

    # 有效股票不足两只时，不做截断
    usable_cs = valid_count >= 2

    clipped_return = torch.where(
        usable_cs,
        clipped_return,
        safe_return,
    )

    # ========================================
    # 6. 记录截断统计
    # ========================================

    clipped_mask = (
        valid_mask
        & usable_cs
        & (
            (safe_return < lower_bound)
            | (safe_return > upper_bound)
        )
    )

    clip_ratio = (
        clipped_mask.sum().float()
        / valid_mask.sum().clamp_min(1)
    )

    metrics = {
        "cs_mean": cs_mean.mean().item(),
        "cs_std": cs_std.mean().item(),
        "clip_ratio": clip_ratio.item(),
        "clip_abs_change": (
            (
                clipped_return - safe_return
            ).abs().sum()
            / valid_mask.sum().clamp_min(1)
        ).item(),
    }

    return clipped_return, metrics

class DualRegimeDynamicGatingLoss(nn.Module):
  """连续门控因子-IR自适应损失函数 (Factor-Deviation Gated IR Loss)

  【取消波动率约束，仅保留因子偏离与IR动态权衡】：
      1. factor_deviation 偏大 (未对齐):
          factor_gate -> 1.0
          -> w_alpha -> alpha_track (低), w_lambda -> lambda_track (高)
          -> 优化器优先把组合的 Barra 风格因子暴露拉回 Benchmark
      2. factor_deviation 偏小 (已对齐):
          factor_gate -> 0.0
          -> w_alpha -> 1.0 (全速追求 IR/超额), w_lambda -> lambda_min (释放自由度)
  """

  def __init__(
      self,
      # 因子偏离阈值 (Smoothstep 区间)
      dev_good: float = 0.005,  # 良好偏离下界 (低于此值进入全力 Alpha 模式)
      dev_bad: float = 0.020,  # 严重偏离上界 (高于此值进入强力追踪模式)
      # 惩罚系数边界
      lambda_track: float = 80.0,  # 因子严重偏离时的强力惩罚系数 (上限)
      lambda_min: float = 2.0,  # 因子对齐后的底线微弱惩罚系数 (下限)
      # IR 进攻系数边界
      alpha_track: float = 0.15,  # 因子偏离过大时保留的基础收益动力 (下限)
      # 数值稳定项

      eps: float = 1e-6,
  ):
    super().__init__()
    self.dev_good = dev_good
    self.dev_bad = dev_bad
    self.lambda_track = lambda_track
    self.lambda_min = lambda_min
    self.alpha_track = alpha_track

    self.eps = eps

  def forward(
      self,
      IR_excess: torch.Tensor,
      factor_deviation: torch.Tensor,
      aux_loss: torch.Tensor = None,
  ):
    """输入:

        IR_excess: [B, T] 连续 T 步的主动超额收益序列 (如 T=20)
        factor_deviation: [B] 每个样本在 Episode 内的平均绝对风格暴露偏离度
        aux_loss: [B] 或标量 (可选) 换手率惩罚等辅助损失项
    输出:
        final_loss: 可微标量损失 (用于 backward)
        metrics: 监控指标字典 (包含 gate、权重及各项分量)
    """
    B, T = IR_excess.shape

    # =========================================================
    # 1. 样本级超额收益统计指标与安全 IR
    # =========================================================
    mean_excess_return = IR_excess.mean(dim=1)  # [B]
    std_excess = IR_excess.std(dim=1)  # [B]

    # 保护性分段 IR: 避免负超额下波动率越小 Loss 越负的梯度反转陷阱
    safe_sample_ir = mean_excess_return / (std_excess + self.eps) 


    # =========================================================
    # 2. 因子门控与权重动态映射 (完全隔离计算图)
    # =========================================================
    with torch.no_grad():
      # 归一化因子偏离状态到 [0.0, 1.0]
      dev_state = factor_deviation.detach()
      dev_pos = torch.clamp(
          (dev_state - self.dev_good)
          / (self.dev_bad - self.dev_good + self.eps),
          min=0.0,
          max=1.0,
      )
      # Smoothstep 平滑曲线: S(x) = 3x^2 - 2x^3 (两端导数为0，杜绝硬切换跳变)
      factor_gate = (dev_pos**2) * (3.0 - 2.0 * dev_pos)  # [B]

      # 动态权重合成:
      # factor_gate = 1 -> Tracking 模式 (w_alpha 最小, w_lambda 最大)
      # factor_gate = 0 -> Alpha 模式    (w_alpha = 1.0, w_lambda 降至 lambda_min)
      w_alpha = factor_gate * self.alpha_track + (1.0 - factor_gate) * 1.0
      w_lambda = (
          factor_gate * self.lambda_track + (1.0 - factor_gate) * self.lambda_min
      )

    # =========================================================
    # 3. 样本级损失构造
    # =========================================================
    loss_return = -w_alpha * safe_sample_ir  # [B] 收益最大化项 (可微)
    loss_factor = w_lambda * factor_deviation  # [B] 因子惩罚项 (可微)

    total_loss_per_sample = loss_return + loss_factor  # [B]

    if aux_loss is not None:
      total_loss_per_sample = total_loss_per_sample + aux_loss

    # 标量聚合
    final_loss = total_loss_per_sample.mean()

    # =========================================================
    # 4. 训练监控指标提取
    # =========================================================
    metrics = {
        "loss_total": final_loss.item(),
        "mean_return": mean_excess_return.mean().item(),
        "std_excess": std_excess.mean().item(),  # 仅作观察，不参与门控
        "sample_ir": safe_sample_ir.mean().item(),
        "factor_deviation": factor_deviation.mean().item(),
        "factor_gate": factor_gate.mean().item(),
        "w_alpha": w_alpha.mean().item(),
        "w_lambda": w_lambda.mean().item(),
        "loss_return": loss_return.mean().item(),
        "loss_factor": loss_factor.mean().item(),
    }

    return final_loss, metrics

class SoftConstraintLoss_mask(nn.Module):
    def __init__(self, tolerance=0.05):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, next_w, factor_exp, target_exp, mask):
        """
        next_w: (B, 1+M) 第一项是现金
        factor_exp: (B, M, F)
        target_exp: (B, F) 这是模块1提前生成的
        mask: (B, F) 掩码张量，1 表示该因子需要被约束，0 表示不约束
        返回: batch_loss (B,) 每个批次样本的平均约束损失
               actual_exp (B, F) 实际暴露度
        """
        # 1. 计算实际暴露
        w_stocks = next_w[:, 1:]  # 剥离现金 (B, M)
        # bmm: (B, 1, M) @ (B, M, F) -> (B, 1, F) -> (B, F)
        actual_exp = torch.bmm(w_stocks.unsqueeze(1), factor_exp).squeeze(1)

        # 2. 计算绝对偏差
        abs_diff = torch.abs(actual_exp - target_exp)  # (B, F)

        # 3. 应用掩码：只考虑需要约束的因子
        masked_abs_diff = abs_diff * mask  # (B, F) * (B, F) -> (B, F)
        # 未约束的因子处，masked_abs_diff 为 0

        # 4. 宽容度拦截 (ReLU)，只对需要约束的因子应用
        violation = torch.relu(masked_abs_diff - self.tolerance)  # (B, F)

        # 5. 计算 MSE，只针对被约束的因子
        # squared_violations = violation ** 2 # (B, F)

        # --- 方案一：对所有 F 个因子计算 MSE，未约束的贡献为 0 ---
        # batch_loss = torch.mean(squared_violations, dim=1) # (B,)

        # --- 方案二：只对被约束的因子计算平均 MSE (推荐) ---
        squared_violations = violation ** 2  # (B, F)
        num_constrained_factors_per_sample = mask.sum(dim=1)  # (B,) 每个样本被约束的因子数量

        # 处理没有约束任何因子的情况（虽然根据你的 sampler，这种情况不会出现，但为了健壮性）
        total_squared_violations_per_sample = squared_violations.sum(dim=1)  # (B,)

        # 避免除以 0。如果某个样本没有任何约束，则其 loss 贡献为 0。
        # PyTorch 通常能处理 0 / 0 的情况，但明确处理更安全。
        batch_loss = torch.where(
            num_constrained_factors_per_sample > 0,
            total_squared_violations_per_sample / num_constrained_factors_per_sample,
            torch.tensor(0.0, dtype=squared_violations.dtype, device=squared_violations.device)
        )  # (B,)

        return batch_loss, actual_exp



class IronBottomMaskSampler:
    def __init__(self, f_dim=10):
        self.f_dim = f_dim
        # 因子自然索引：
        # 0:Size, 1:NL_Size, 2:Mom, 3:Liq, 4:BP, 5:Lev, 6:Growth, 7:EY, 8:Beta, 9:ResVol

        # 无论什么情况，这三个风控因子绝对不放开！
        self.core_risk_indices = [0, 8, 9]
        self.core_risk_indices_nosize = [8, 9]


    def sample_mask(self, batch_size, device):
        p = random.random()
        mask = torch.zeros(batch_size, self.f_dim, device=device)

        if p < 0.37:
            # 场景 1: 全约束 (30% 概率)
            mask.fill_(1.0)
            print("MaskSampler: 全约束场景触发 (30%)")

        elif p < 0.40:
            mask.fill_(0)
            print("MaskSampler: 0约束场景触发 (10%)")

        elif p < 0.60:
            # 场景 3: 核心风控 (15% 概率)
            # 仅约束: 0(市值), 8(Beta), 9(残余波动率)
            mask[:, [0, 8, 9]] = 1.0
            print("MaskSampler: 核心风控场景触发 (15%)")

        elif p < 0.80:
            # 场景 4: 风格约束 (20% 概率)
            # 随机挑选几个风格因子进行约束，这里做了一个小随机
            # 比如强行约束 价值(4) 和 成长(6)
            mask[:, [4, 6]] = 1.0
            print("MaskSampler: 风格约束场景触发 (20%)")

            # 50% 概率额外约束动量(2)
            if random.random() > 0.5:
                mask[:, 2] = 1.0

        else:
            # 场景 5: 完全随机探索 (20% 概率)
            # 随机生成 0 和 1
            mask = torch.randint(0, 2, (batch_size, self.f_dim), device=device).float()
            print("MaskSampler: 完全随机探索 (20%)")

            # 【重要补丁】：防止荒谬的物理冲突
            # 如果约束了非线性市值(1)，则必须约束市值(0)
            # (在连续批处理中，用布尔逻辑纠正)
            mask[:, 0] = torch.logical_or(mask[:, 0].bool(), mask[:, 1].bool()).float()

        return mask


'''模拟成分股生成器'''
class IndexTargetGenerator:
    def __init__(self, num_stocks_total=600, num_stocks_select=200):
        self.M = num_stocks_total
        self.K = num_stocks_select

        # 因子索引映射
        self.FACTOR_IDX = {
            "size": 0, "non_linear_size": 1, "momentum": 2, "liquidity": 3,
            "b2p": 4, "leverage": 5, "growth": 6, "earnings_yield": 7,
            "beta": 8, "residual_volatility": 9
        }
        self.num_factors = len(self.FACTOR_IDX)

    def _solve_temperature(self, size_exp: torch.Tensor, target_max_w: float) -> torch.Tensor:
        """
        利用二分查找寻找能使 Softmax 最大权重等于 target_max_w 的 cap_temp
        """
        # 极端情况防御：如果所有股票 size 一模一样，只能返回均匀分布
        if size_exp.max() - size_exp.min() < 1e-5:
            return F.softmax(torch.ones_like(size_exp), dim=0)

        T_low = 0.01  # 极度集中 (最大权重趋近 100%)
        T_high = 100.0  # 极度分散 (最大权重趋近 1/K = 0.5%)

        # 二分查找 15 次，精度已经能达到 0.001 级别，计算开销极小
        for _ in range(15):
            T_mid = (T_low + T_high) / 2.0
            weights = F.softmax(size_exp / T_mid, dim=0)
            current_max_w = weights.max().item()

            if current_max_w > target_max_w:
                # 当前集中度太高，说明温度太低了，需要升温
                T_low = T_mid
            else:
                # 当前集中度太低，说明温度太高了，需要降温
                T_high = T_mid

        # 使用最终找到的温度计算权重
        final_T = (T_low + T_high) / 2.0
        return F.softmax(size_exp / final_T, dim=0)

    @torch.no_grad()
    def generate_index_weights(self, barra_exp: torch.Tensor) -> torch.Tensor:
        """
        输入: barra_exp, shape [B, M, F=10] (标准化的 Barra 因子暴露)
        输出: index_weights, shape [B, M, 1]
        """
        B, M, F_dim = barra_exp.shape
        weights_out = torch.zeros((B, M, 1), device=barra_exp.device, dtype=barra_exp.dtype)

        for b in range(B):
            sample_exp = barra_exp[b]

            # --- 1. 决定抽样策略获取 200 只成分股 (包含 10% 纯随机) ---
            dice = torch.rand(1).item()

            if dice < 0.10:
                # 策略 1: 纯随机抽样 (10%)
                selected_idx = torch.randperm(M, device=barra_exp.device)[:self.K]

            elif dice < 0.15:
                # 策略 2 升级版: 单因子区间硬切 (Single-Factor Block Slicing) (35%)
                # 随机选出一个因子
                factor_i = 0

                # 对该因子在 600 只股票上进行从大到小排序 (得到排好序的索引)
                # torch.sort 返回 (values, indices)，我们只需要 indices
                _, sorted_idx = torch.sort(sample_exp[:, factor_i], descending=True)
                max_start_idx = self.M - self.K
                selected_idx = sorted_idx[max_start_idx: max_start_idx + self.K]

            elif dice < 0.45:
                # 策略 2 升级版: 单因子区间硬切 (Single-Factor Block Slicing) (35%)
                # 随机选出一个因子
                factor_i = torch.randint(0, F_dim, (1,)).item()

                # 对该因子在 600 只股票上进行从大到小排序 (得到排好序的索引)
                # torch.sort 返回 (values, indices)，我们只需要 indices
                _, sorted_idx = torch.sort(sample_exp[:, factor_i], descending=True)

                # 核心魔法：随机生成一个截断起点 (Start Index)
                # 既然要切出 200 (self.K) 只，起点最大只能是 600 - 200 = 400
                max_start_idx = self.M - self.K
                start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()

                # 在排好序的名单里，从起点开始连切 200 个！
                selected_idx = sorted_idx[start_idx: start_idx + self.K]

            elif dice < 0.80:
                # 策略 3 升级版: 概率随机抽样 (包含极值与中段偏好)
                factor_i = torch.randint(0, F_dim, (1,)).item()

                # 随机决定当前是喜欢：0 (最大), 1 (最小), 还是 2 (最中庸)
                preference = torch.randint(0, 3, (1,)).item()

                if preference == 0:
                    # 喜欢最大：原封不动
                    target_scores = sample_exp[:, factor_i]
                elif preference == 1:
                    # 喜欢最小：反转符号
                    target_scores = -sample_exp[:, factor_i]
                else:
                    # 喜欢中段：计算距离均值(0)的绝对值，然后取负！
                    # 这样越靠近 0（中位数）的股票，得分越高，越容易被抽中
                    target_scores = -torch.abs(sample_exp[:, factor_i])

                # 转成概率并抽签
                prob = F.softmax(target_scores / 2.0, dim=0)
                selected_idx = torch.multinomial(prob, self.K, replacement=False)
            else:
                # 策略 4 升级版: 双因子异构共振 (Top-K 硬切)
                f1, f2 = torch.randperm(F_dim)[:2].tolist()

                # 为因子 1 决定偏好：0(最大), 1(最小), 2(中庸)
                pref1 = torch.randint(0, 3, (1,)).item()
                if pref1 == 0:
                    score1 = sample_exp[:, f1]
                elif pref1 == 1:
                    score1 = -sample_exp[:, f1]
                else:
                    # 中庸偏好：距离中位数(0)越近得分越高
                    score1 = -torch.abs(sample_exp[:, f1])

                # 为因子 2 决定偏好：0(最大), 1(最小), 2(中庸)
                pref2 = torch.randint(0, 3, (1,)).item()
                if pref2 == 0:
                    score2 = sample_exp[:, f2]
                elif pref2 == 1:
                    score2 = -sample_exp[:, f2]
                else:
                    score2 = -torch.abs(sample_exp[:, f2])

                # 合成共振得分：寻找同时满足两个偏好的交集股票
                composite_score = score1 + score2

                # 在共振总分上，绝对坚持硬切 Top-K！
                _, selected_idx = torch.topk(composite_score, self.K)

            # --- 2. 提取这 200 只股票的 Size 因子暴露 ---
            # 因子 0 是 size (标准化后的市值的代理变量)，市值的因子暴露被提取作为权重分配的依据
            size_exposure = sample_exp[selected_idx, 0]

            # --- 全新的多元化权重分配系统 ---
            weight_dice = torch.rand(1).item()

            if weight_dice < 0.05:
                # [新增] 规则 1：等权分配 (Equal Weighting) (20%)
                # 能够完美保留中小盘股的原始暴露度，拯救左侧断崖！
                final_weights = torch.ones(self.K, device=barra_exp.device) / self.K

            elif weight_dice < 0.4:
                # [新增] 规则 2：极度平滑加权 (High Temperature Weighting) (20%)
                # 类似某些受限指数，最大权重被死死压在 1.5% 以下，极大削弱大盘股虹吸效应
                target_max_w = torch.rand(1).item() * (0.015 - 0.005) + 0.005
                final_weights = self._solve_temperature(size_exposure, target_max_w)

            else:
                # 规则 3：经典的自适应市值加权 (Cap Weighting) (60%)
                # 允许出现巨头，最大权重在 2% ~ 5% 之间游走，保留右侧的大盘股特征
                target_max_w = torch.rand(1).item() * (0.10 - 0.02) + 0.02
                final_weights = self._solve_temperature(size_exposure, target_max_w)

            # 将算出的 200 个股票权重填充到全局的 600 张量中
            weights_out[b, selected_idx, 0] = final_weights
        # 这里保留 B*M*1 的张量结构，方便之后作点积，在计算收益率时，用 squeeze(-1) 收束掉最后的一维
        return weights_out


'''十因子目标生成器'''
class HistoricalCorrelatedTargetGenerator(nn.Module):
    def __init__(self, target_scale=0.8):
        super().__init__()
        self.target_scale = target_scale

    def forward(self, factor_exp_hist):
        B, M, L, F = factor_exp_hist.shape
        device = factor_exp_hist.device
        orig_dtype = factor_exp_hist.dtype

        # 确定当前的设备类型，用于关闭 autocast
        # 如果是 cuda 就传 'cuda'，否则传 'cpu'
        device_type = 'cuda' if factor_exp_hist.is_cuda else 'cpu'

        # ==========================================
        # [核心魔法] 强制关闭当前的混合精度上下文
        # 这样 torch.bmm 就绝对不会偷偷降级回 BFloat16 了
        # ==========================================
        with torch.autocast(device_type=device_type, enabled=False):
            # 在安全的结界内，强转为 float32
            calc_dtype = torch.float32
            factors_f32 = factor_exp_hist.to(calc_dtype)

            # 1. 提取最新一天锚点
            current_factors = factors_f32[:, :, -1, :]
            mu_today = torch.nanmean(current_factors, dim=1)

            curr_mask = torch.isnan(current_factors)
            safe_max_factors = torch.where(curr_mask, torch.tensor(float('-inf'), dtype=calc_dtype, device=device),
                                           current_factors)
            safe_min_factors = torch.where(curr_mask, torch.tensor(float('inf'), dtype=calc_dtype, device=device),
                                           current_factors)

            min_val = safe_min_factors.min(dim=1)[0]
            max_val = safe_max_factors.max(dim=1)[0]

            # 2. 计算 80 天全量历史协方差
            hist_flat = factors_f32.reshape(B, M * L, F)
            hist_mu = torch.nanmean(hist_flat, dim=1, keepdim=True)

            hist_filled = torch.where(torch.isnan(hist_flat), hist_mu, hist_flat)
            centered = hist_filled - hist_mu

            N_samples = M * L
            # 因为关了 autocast，这里的 bmm 输出必定是 float32！
            cov_matrix = torch.bmm(centered.transpose(1, 2), centered) / (N_samples - 1)

            # 3. 特征值分解 (EVD)
            # 此时 cov_matrix 绝对是 Float32，不再报错！
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            eigenvalues = torch.relu(eigenvalues)
            sqrt_eigvals = torch.sqrt(eigenvalues)
            T = eigenvectors * sqrt_eigvals.unsqueeze(1)

            # 4. 生成联合相关噪声
            white_noise = torch.randn(B, F, 1, dtype=calc_dtype, device=device)
            correlated_noise = torch.bmm(T, white_noise).squeeze(-1)

            # 5. 合成最终目标并兜底
            target_exp = mu_today + self.target_scale * correlated_noise
            target_exp = torch.clamp(target_exp, min=min_val, max=max_val)

        # ==========================================
        # 脱离结界后，切回原始精度 (BFloat16)
        # ==========================================
        return target_exp.to(orig_dtype).detach()


# ==========================================
# 模块 1：目标生成器 (无需训练的辅助模块)
# ==========================================
class TargetGenerator:
    def __init__(self, target_scale=0.8):
        self.target_scale = target_scale

    def generate(self, factor_exp):
        """
        factor_exp: (B, M, F)
        返回: target_exp (B, F)
        """
        # 计算截面均值和标准差
        mu = factor_exp.mean(dim=1)
        sigma = factor_exp.std(dim=1)

        # 计算极值用于兜底
        min_val = factor_exp.min(dim=1)[0]
        max_val = factor_exp.max(dim=1)[0]

        # 基于正态分布生成合理随机目标
        noise = torch.randn_like(mu)
        target_exp = mu + noise * self.target_scale * sigma

        # 截断限制
        target_exp = torch.clamp(target_exp, min=min_val, max=max_val)

        # 目标值作为环境给定，不需要计算梯度
        return target_exp.detach()


# ==========================================
# 模块 3：宽容度软约束 Loss 模块
# ==========================================
class SoftConstraintLoss(nn.Module):
    def __init__(self, tolerance=0.05):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, next_w, factor_exp, target_exp):
        """
        next_w: (B, 1+M) 第一项是现金
        factor_exp: (B, M, F)
        target_exp: (B, F) 这是模块1提前生成的
        返回: batch_loss (B,)
        """
        # 1. 计算实际暴露
        w_stocks = next_w[:, 1:]  # 剥离现金 (B, M)

        # bmm: (B, 1, M) @ (B, M, F) -> (B, 1, F) -> (B, F)
        actual_exp = torch.bmm(w_stocks.unsqueeze(1), factor_exp).squeeze(1)

        # 2. 计算绝对偏差
        abs_diff = torch.abs(actual_exp - target_exp)

        # 3. 宽容度拦截 (ReLU)
        violation = torch.relu(abs_diff - self.tolerance)

        # 4. 化十为一：计算各个因子的 MSE 并求平均
        batch_loss = torch.mean(violation ** 2, dim=1)

        return batch_loss, actual_exp


def format_id(val):
    """杜绝科学计数法，保留全精度 ID"""
    if np.isnan(val): return "NaN"
    if np.issubdtype(type(val), np.floating):
        return str(int(val)) if val == int(val) else f"{val:.10f}"
    return str(val)


def generate_episode_report(daily_ee_list, daily_nextw_list, daily_next_relative_price_list,
                            daily_date_list, stock_id, output_dir='./model_debug'):
    """
    生成单文件 Episode 诊断报告：
    - 文件名以首个交易日命名
    - 上半部分：20天聚合收益归因与异常检测
    - 下半部分：每日细部数据 (ee, Top200权重, Bottom200收益)
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(daily_date_list) == 0:
        raise ValueError("daily_date_list 不能为空")

    # 1. 提取首个交易日用于文件命名 & 基础张量预处理
    start_date = int(daily_date_list[0].detach().cpu().item())
    ids = stock_id.detach().cpu().squeeze().numpy()  # shape: (2000,)
    n_days = len(daily_ee_list)

    daily_stats = []
    daily_detail_blocks = []
    all_stock_contribs = np.zeros(len(ids))

    # 2. 逐日计算与细部格式化
    for d in range(n_days):
        ee_val = daily_ee_list[d].detach().cpu().item()
        t1Day_val = int(daily_date_list[d].detach().cpu().item())
        w = daily_nextw_list[d].detach().cpu().squeeze().numpy()  # (2001,)
        ret = daily_next_relative_price_list[d].detach().cpu().squeeze().numpy()  # (2001,)

        cash_w, stock_w = w[0], w[1:]
        cash_ret, stock_ret = ret[0], ret[1:]  # ret 为乘数型 (P_t+1/P_t)

        # 核心收益计算
        port_ret = np.dot(stock_w, stock_ret) + cash_w * cash_ret - 1.0  # 简单收益率
        ew_ret = np.mean(stock_ret) - 1.0
        calc_ee = port_ret - ew_ret

        # 辅助指标
        w_sum = np.sum(w)
        has_nan = np.any(np.isnan(w)) or np.any(np.isnan(ret))
        std_w, std_r = np.std(stock_w), np.std(stock_ret)
        corr = np.corrcoef(stock_w, stock_ret)[0, 1] if std_w > 1e-8 and std_r > 1e-8 else 0.0

        # 单日股票贡献度 (权重 * 简单收益)
        daily_contrib = stock_w * (stock_ret - 1.0)
        all_stock_contribs += daily_contrib

        daily_stats.append({
            'day': d + 1, 'date': t1Day_val, 'ee': ee_val, 'calc_ee': calc_ee,
            'port_ret': port_ret, 'ew_ret': ew_ret, 'w_sum': w_sum,
            'corr': corr, 'has_nan': has_nan
        })

        # --- 细部数据格式化 ---
        top200_idx = np.argsort(stock_w)[::-1][:200]
        bot200_idx = np.argsort(stock_ret)[:200]

        blk = []
        blk.append(f"\n{'=' * 90}")
        blk.append(f"【Day {d + 1:02d} | Date: {t1Day_val}】")
        blk.append(
            f"  📊 ee: {ee_val:+.6f} | calc_ee: {calc_ee:+.6f} | Port: {port_ret:+.4%} | EW: {ew_ret:+.4%} | W_Sum: {w_sum:.4f} | Corr: {corr:.4f} {'⚠️NaN' if has_nan else ''}")

        blk.append(f"  🏆 Top 200 持仓权重 (降序):")
        blk.append(f"  {'Stock_ID':<20} | {'Weight':<12} | {'Next_Return':<12}")
        for idx in top200_idx:
            blk.append(f"  {format_id(ids[idx]):<20} | {stock_w[idx]:<12.6f} | {stock_ret[idx] - 1.0:+.6f}")

        blk.append(f"  📉 Bottom 200 隔日收益 (升序):")
        blk.append(f"  {'Stock_ID':<20} | {'Next_Return':<12}")
        for idx in bot200_idx:
            blk.append(f"  {format_id(ids[idx]):<20} | {stock_ret[idx] - 1.0:+.6f}")

        daily_detail_blocks.append("\n".join(blk))

    # 3. Episode 级聚合分析
    port_rets = np.array([s['port_ret'] for s in daily_stats])
    ew_rets = np.array([s['ew_ret'] for s in daily_stats])
    ee_vals = np.array([s['ee'] for s in daily_stats])

    cum_port_ret = np.prod(1 + port_rets) - 1
    cum_ew_ret = np.prod(1 + ew_rets) - 1
    cum_ee = cum_port_ret - cum_ew_ret

    top_days_idx = np.argsort(ee_vals)[::-1]
    days_contrib_pct = (ee_vals[top_days_idx] / (np.sum(ee_vals) + 1e-9)) * 100

    top_stk_idx = np.argsort(all_stock_contribs)[::-1][:20]
    pos_contribs = all_stock_contribs[all_stock_contribs > 0]
    top_stk_contrib_pct = (all_stock_contribs[top_stk_idx] / (np.sum(pos_contribs) + 1e-9)) * 100

    avg_corr = np.mean([s['corr'] for s in daily_stats])
    ee_consistent = np.all(np.isclose(ee_vals, [s['calc_ee'] for s in daily_stats], atol=1e-7))

    # 4. 组装完整报告
    lines = []
    sep = "=" * 90
    lines.append(f"{sep}")
    lines.append(f"📊 EPISODE DIAGNOSTIC REPORT | Start Date: {start_date} | Days: {n_days}")
    lines.append(f"{sep}\n")

    lines.append("【📈 核心收益总览】")
    lines.append(f"🎯 目标阈值 (总净值 >1.2) : {'✅ 已突破' if cum_port_ret > 0.20 else '❌ 未达标'}")
    lines.append(f"📊 组合累计收益率       : {cum_port_ret:+.4%}")
    lines.append(f"📉 等权基准累计收益率   : {cum_ew_ret:+.4%}")
    lines.append(f"💰 累计超额收益 (ee)    : {cum_ee:+.4%}")
    lines.append(f"📅 ee 公式一致性校验    : {'✅ 通过' if ee_consistent else '🚨 失败 (内部计算与打印不一致)'}\n")

    lines.append("【🔍 高收益来源拆解】")
    lines.append(
        f"📌 收益高度集中在前 5 天 : {'是' if np.sum(days_contrib_pct[:5]) > 70 else '否'} (占比: {np.sum(days_contrib_pct[:5]):.1f}%)")
    lines.append(f"📌 Top 5 贡献日 (按 ee):")
    for i, idx in enumerate(top_days_idx[:5]):
        s = daily_stats[idx]
        lines.append(
            f"   {i + 1}. Day {s['day']:02d} (Date:{s['date']}) | ee={s['ee']:+.4%} | Port={s['port_ret']:+.4%} | Corr={s['corr']:.3f}")
    lines.append("")

    lines.append(f"📌 Top 20 贡献股票 (20天累计加权收益):")
    lines.append(f"{'Rank':<5} {'Stock_ID':<20} {'Total_Contrib':<15} {'% of Profit'}")
    for i, idx in enumerate(top_stk_idx):
        contrib = all_stock_contribs[idx]
        if contrib <= 0: break
        lines.append(f"{i + 1:<5} {format_id(ids[idx]):<20} {contrib:+.6f} {top_stk_contrib_pct[i]:<12.1f}%")
    lines.append("")

    lines.append("【⚠️ 异常与风险诊断】")
    leak_tag = "🚨 极高(>0.3)" if avg_corr > 0.3 else "⚠️ 偏高(0.15-0.3)" if avg_corr > 0.15 else "✅ 正常(<0.15)"
    lines.append(f"🕳️ 数据泄露风险 (Avg Corr) : {avg_corr:.4f} {leak_tag}")
    lines.append(
        f"📉 权重归一化偏差          : Min={min(s['w_sum'] for s in daily_stats):.4f} | Max={max(s['w_sum'] for s in daily_stats):.4f}")
    lines.append(f"💀 含 NaN/Inf 天数         : {sum(1 for s in daily_stats if s['has_nan'])}/{n_days}")
    lines.append(
        f"📊 收益分布集中度         : {'高(少数股票驱动)' if np.sum(top_stk_contrib_pct[:5]) > 50 else '中' if np.sum(top_stk_contrib_pct[:10]) > 70 else '低(分散)'}")
    lines.append(
        f"\n💡 核心提示: 若 ee 异常偏高且 Corr>0.2，极大概率为特征包含未来价格/标签泄露，或权重未做软约束导致过拟合噪声。\n")

    lines.append(sep)
    lines.append("📅 逐日详细数据 (Daily Breakdown)")
    lines.append(sep)
    lines.extend(daily_detail_blocks)
    lines.append(f"\n{sep}\n🔚 END OF REPORT")

    # 5. 写入文件
    fname = os.path.join(output_dir, f"episode_debug_start_{start_date}.txt")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"✅ 诊断报告已生成: {fname}")
    return "\n".join(lines)


def generate_debug_report(ee, next_w, next_relative_price, t1Day, stock_id, output_dir='debug_logs'):
    """
    生成按交易日命名的诊断 txt 文件，用于排查 ee 异常偏高问题
    """
    # 1. 安全迁移至 CPU 并转为 numpy
    ee_val = ee.detach().cpu().item()
    t1Day_val = int(t1Day.detach().cpu().item())

    w = next_w.detach().cpu().squeeze(0).numpy()  # shape: (2001,)
    ret = next_relative_price.detach().cpu().squeeze(0).numpy()  # shape: (2001,)
    ids = stock_id.detach().cpu().squeeze(0).numpy()  # shape: (2000,)

    # 分离现金与股票部分
    cash_w, stock_w = w[0], w[1:]  # stock_w: (2000,)
    cash_ret, stock_ret = ret[0], ret[1:]  # stock_ret: (2000,)

    # 2. 核心收益计算 (兼容 乘数型 vs 简单收益率)
    # 若 next_relative_price 是 P_t+1/P_t (乘数)，则收益 = dot(w, ret) - 1
    # 若 next_relative_price 已经是 (P_t+1-P_t)/P_t (简单收益)，则收益 = dot(w, ret)
    port_ret_mult = np.dot(stock_w, stock_ret) + cash_w * cash_ret
    port_ret_simple = port_ret_mult - 1.0

    ew_ret_mult = np.mean(stock_ret)
    ew_ret_simple = ew_ret_mult - 1.0

    calc_ee = port_ret_simple - ew_ret_simple

    # 3. 辅助指标
    w_sum = np.sum(w)
    has_nan = np.any(np.isnan(w)) or np.any(np.isnan(ret))
    has_inf = np.any(np.isinf(w)) or np.any(np.isinf(ret))
    # 权重与未来收益的相关系数 (核心泄露检测指标)
    corr = np.corrcoef(stock_w, stock_ret)[0, 1] if np.std(stock_w) > 1e-8 and np.std(stock_ret) > 1e-8 else 0.0

    # ID 格式化：杜绝科学计数法，保留全精度
    def fmt_id(val):
        if np.issubdtype(type(val), np.floating):
            return str(int(val)) if val == int(val) else f"{val:.10f}"
        return str(val)

    # 排序索引
    top200_idx = np.argsort(stock_w)[::-1][:200]
    bot200_idx = np.argsort(stock_ret)[:200]

    # 4. 组装文本
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"📊 诊断报告 | 交易日: {t1Day_val}")
    lines.append(f"{'=' * 60}\n")

    lines.append("【核心指标核对】")
    lines.append(f"模型报告 ee                : {ee_val:.8f}")
    lines.append(f"重新计算组合收益 (简单)    : {port_ret_simple:.8f}")
    lines.append(f"重新计算等权基准收益 (简单): {ew_ret_simple:.8f}")
    lines.append(f"重新计算 ee                : {calc_ee:.8f}")
    lines.append(f"ee 计算是否一致            : {np.isclose(ee_val, calc_ee, atol=1e-7)}\n")

    lines.append(f"现金权重: {cash_w:.6f} | 现金收益: {cash_ret - 1.0:.6f}")
    lines.append(f"总权重和 (含现金): {w_sum:.6f} (⚠️ 理论应严格为 1.0)")
    lines.append(f"数据完整性: 含 NaN={has_nan} | 含 Inf={has_inf}\n")

    lines.append("【🔝 持仓权重 Top 200 (降序)】")
    lines.append(f"{'Stock_ID':<20} | {'Weight':<15} | {'Next_Return':<15}")
    for idx in top200_idx:
        lines.append(f"{fmt_id(ids[idx]):<20} | {stock_w[idx]:<15.6f} | {stock_ret[idx] - 1.0:<15.6f}")

    lines.append(f"\n【🔻 隔日收益 Bottom 200 (升序)】")
    lines.append(f"{'Stock_ID':<20} | {'Next_Return':<15}")
    for idx in bot200_idx:
        lines.append(f"{fmt_id(ids[idx]):<20} | {stock_ret[idx] - 1.0:<15.6f}")

    lines.append(f"\n【🔍 附加诊断指标】")
    lines.append(f"权重 vs 隔日收益 相关系数: {corr:.4f} (🚨 >0.3 需高度警惕未来函数/数据泄露)")
    lines.append(
        f"权重分布: Mean={np.mean(stock_w):.6f}, Std={np.std(stock_w):.6f}, Min={np.min(stock_w):.6f}, Max={np.max(stock_w):.6f}")
    lines.append(
        f"收益分布: Mean={np.mean(stock_ret) - 1.0:.6f}, Std={np.std(stock_ret):.6f}, Min={np.min(stock_ret) - 1.0:.6f}, Max={np.max(stock_ret) - 1.0:.6f}")
    lines.append(f"\n💡 提示: 若 ee 异常偏高，请重点检查 '相关系数' 是否过高，以及 '现金权重/收益' 定义是否与实际对齐。")

    content = "\n".join(lines)

    # 5. 写入文件
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"debug_day_{t1Day_val}.txt")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 诊断文件已生成: {fname}")
    return content


# torch.set_printoptions(precision=10, sci_mode=False)

def winsorize_3sigma(x, dim=-1):
    """3σ 缩尾去极值 (Winsorization)"""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False)
    lower = mean - 3 * std
    upper = mean + 3 * std
    return torch.clamp(x, min=lower, max=upper)


def generate_market_cap_exposure(input_tensor, cap_channel_idx=33):
    """
    输入: input_tensor shape=(B, M, L, C)
    输出: market_cap_factor shape=(B, M, L)
          对 L 维度的每一天独立计算截面因子暴露，完全向量化无循环
    """
    # 1. 提取流通市值通道 (B, M, L)
    cap = input_tensor[:, :, :, cap_channel_idx]

    # 防御性处理：防止数据中存在 0 或负数导致 log 报错
    cap = torch.clamp(cap, min=1e-8)

    # ================= 对应图片步骤 =================
    # 注意：此时张量形状为 (B, M, L)，股票维度 M 位于 dim=-2
    # 步骤1: log总市值 + std去极值
    log_cap = torch.log(cap)
    wins_log_cap = winsorize_3sigma(log_cap, dim=-2)  # 沿 M 维度去极值

    # 步骤3: 市值加权平均
    # 权重使用原始流通市值 (Barra标准做法)
    weights = cap
    # 沿 M 维度 (dim=-2) 加权求和
    weighted_mean = (wins_log_cap * weights).sum(dim=-2, keepdim=True) / (
                weights.sum(dim=-2, keepdim=True) + 1e-8)  # (B, 1, L)

    # 步骤4: 标准化 (暴露度 - 市值加权均值) / 截面标准差
    std = wins_log_cap.std(dim=-2, keepdim=True, unbiased=False)  # (B, 1, L)
    standardized = (wins_log_cap - weighted_mean) / (std + 1e-8)  # (B, M, L)

    # 步骤5: 再做一次3std去极值
    final_exposure = winsorize_3sigma(standardized, dim=-2)  # (B, M, L)

    return final_exposure


# 格式化股票代码函数
def format_stock_code(stock_num):
    """将股票ID转为六位字符串 + .SZ 或 .SH 后缀"""
    code_str = str(int(stock_num)).zfill(6)  # 补零到六位
    if code_str.startswith('6') or code_str.startswith('9'):
        suffix = '.SH'  # 上海市场
    else:
        suffix = '.SZ'  # 深圳市场（包括0、3开头）
    return code_str + suffix


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


def make_info_str(costmark, M, trade_process_list, fv_list, last_value_list, fooler_value, weight_cash, weight_change,
                  weight_entropy, BigStockCount):
    total_count = len(last_value_list)
    if total_count <= 0:
        print(f"{CM.timestr()}***!!!!!没有找到测试结果文件或者文件为空!!!!!***")
        return f"{costmark}-Error-No-Result"
    mvl = len(trade_process_list)
    fvl = len(fv_list)
    assert mvl == fvl, f"交易过程数{mvl}和大盘价格变化数{fvl}不一致"

    model_value = sum(last_value_list) / total_count
    mfr = (model_value - fooler_value) / fooler_value * 100.
    losecount = sum([i < 1 for i in last_value_list])
    metrics = CM.get_pm_metrics(trade_process_list)
    metrics_mtof = CM.get_pm_metrics_2(trade_process_list, fv_list)
    index_str = f"{costmark}"
    index_str += f",M{M}-{total_count}"
    index_str += f",{model_value:.4f}"
    index_str += f",{fooler_value:.4f}"
    index_str += f",{mfr:.2f}%"
    index_str += f",{weight_cash:.4f}"
    index_str += f",{weight_change / 2:.4f}"
    index_str += f",{weight_entropy:.4f}"
    index_str += f",{BigStockCount[0]:.1f}/{BigStockCount[1]:.1f}/{BigStockCount[2]:.1f}/{BigStockCount[3]:.1f}/{BigStockCount[4]:.1f}/{BigStockCount[5] * 100:.1f}/{BigStockCount[6] * 100:.1f}"
    index_str += f",{max(last_value_list):.4f}"
    index_str += f",{min(last_value_list):.4f}"
    index_str += f",{losecount}"
    index_str += f",{losecount / total_count * 100:.2f}%"
    index_str += f",{metrics[0]:.4f}"
    index_str += f",{metrics[1]:.4f}"
    index_str += f",{metrics[2]:.4f}"
    index_str += f",{metrics[3]:.4f}"
    index_str += f",{metrics[4]:.4f}"
    index_str += f",{metrics[5]:.4f}"
    index_str += f",{metrics[6]:.4f}"
    index_str += f",{metrics_mtof[0]:.4f}"
    index_str += f",{metrics_mtof[1]:.4f}"
    index_str += f",{metrics_mtof[2]:.4f}"
    return index_str


def clear_temp_files():
    # 删除所有临时文件 当前目录下以"TMP_test_process_"开头的所有文件
    for file in os.listdir("."):
        if file.startswith("TMP_test_process_"):
            os.remove(file)


def read_test_result(mark):
    trade_process_list = []
    fv_list = []  # 同一时期,大盘(平均M只股票)的价格变化列表,每日相对前一日比值
    last_value_list = []
    fnmark = f"TMP_test_process_{mark}"
    for file in os.listdir('.'):
        if file.startswith(fnmark):
            listx = read_lists_from_file(file)
            for t in listx:
                trade_process_list.append(t[0])
                last_value_list.append(t[1])
                fv_list.append(t[2])
    return trade_process_list, fv_list, last_value_list


def read_lists_from_file(filename):
    higher_dim_list = []  # 初始化一个更高一维度的列表
    with open(filename, 'r') as file:  # 以读取模式打开文件
        for line in file:  # 遍历文件的每一行
            line = line.strip()  # 去除行尾的换行符
            if line:  # 如果行不为空
                # 使用eval将字符串转换回列表，并添加到更高一维度的列表中
                higher_dim_list.append(eval(line))
    return higher_dim_list


def weighted_loss_log_scale(loss1, loss2):
    # 计算对数尺度的损失大小
    log_norm1 = torch.log(torch.mean(loss1.detach() ** 2) + 1e-8)
    log_norm2 = torch.log(torch.mean(loss2.detach() ** 2) + 1e-8)

    # 损失越大，log_norm越大，1/log_norm越小
    weight1 = 1.0 / (log_norm1 + 1e-8)
    weight2 = 1.0 / (log_norm2 + 1e-8)

    # 归一化权重
    total_weight = weight1 + weight2
    alpha = weight1 / total_weight
    beta = weight2 / total_weight

    # 加权求总损失
    return alpha * loss1 + beta * loss2


def read_zz1000():
    # 从本地文件中读取中证1000指数,生成一个字典
    zzfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "全A.json")
    if os.path.exists(zzfile):
        with open(zzfile, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


class LamCalibrator:
    def __init__(self, target_ratio=0.3, ema_alpha=0.05, lam_min=1e-4, lam_max=1e4, eps=1e-8):
        self.target_ratio = target_ratio
        self.ema_alpha = ema_alpha
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.eps = eps
        self.ema_lam = None  # keep smoothed lam

    def calibrate(self, loss_down_batch, var_raw_batch):
        """
        loss_down_batch: tensor (B,) per-episode asymmetric loss (raw, not mean across batch)
        var_raw_batch: tensor (B,) per-episode downside-variance BEFORE multiplying lam
        returns: scalar lam (python float)
        """
        L = loss_down_batch.detach().mean().item()
        V = var_raw_batch.detach().mean().item()
        # if V tiny, avoid huge lam by using eps
        lam_new = (self.target_ratio * L) / (V + self.eps)

        # EMA smoothing
        if self.ema_lam is None:
            self.ema_lam = lam_new
        else:
            self.ema_lam = (1 - self.ema_alpha) * self.ema_lam + self.ema_alpha * lam_new

        # clamp and return
        lam_clamped = float(max(self.lam_min, min(self.lam_max, self.ema_lam)))
        return lam_clamped



class LightningModel_PM(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.use_w_loss = args.use_w_loss == 1
        self.testing = args.RUN_MODE.startswith('check')
        args.testing = self.testing
        if self.testing:
            clear_temp_files()
        self.model = PortfolioModel_SMV2(args)
        self.lam_calibrator = LamCalibrator(target_ratio=0.2, ema_alpha=0.05, lam_min=1e-4, lam_max=1e4, eps=1e-8)

        self.try_reload_model()

        # 你的主干网络
        # self.backbone = ...
        self.dynamic_loss_fn = DualRegimeDynamicGatingLoss(
                dev_good=0.02,
                dev_bad=0.2,
                lambda_track=80.0,
                lambda_min=20,
                alpha_track=0,
            )

        if self.global_rank == 0:
            if not CM.mark_tmp('check', args.mark):
                CM.mark_tmp('create', args.mark)
                self.model_info = CM.summary_train_info(args, verbose=True)
                summary(self.model, depth=3)

        self.start_time = 0
        self.train_batch_loss = []  # 用于记录训练集上每个batch的loss
        self.train_batch_value = []  # 用于记录训练集上每个batch的value
        self.val_batch_loss = []  # 用于记录验证集上每个batch的loss
        self.val_batch_value = []  # 用于记录验证集上每个batch的value
        self.val_batch_ee = []
        self.test_batch_loss = []  # 用于记录测试集上每个batch的loss
        self.test_batch_value = []  # 用于记录测试集上每个batch的value

        self.test_fooler_value = []  # 用于记录测试集上每个fooler的value
        self.test_weight_entropy = []  # 用于记录测试集上的weight entropy
        self.test_weight_cash = []  # 用于记录测试集上的weight cash
        self.test_BigStockCount0 = []
        self.test_BigStockCount2 = []  # 用于记录测试集上的大盘股票数量
        self.test_BigStockCount5 = []  # 用于记录测试集上的大盘股票数量
        self.test_BigStockCount10 = []  # 用于记录测试集上的大盘股票数量
        self.test_BigStockCount20 = []  # 用于记录测试集上的大盘股票数量
        self.test_weight_change_total = []  # 用于记录测试集上换手率
        self.test_TopStock1 = []  # 用于记录测试集上Top1股票
        self.test_TopStock2 = []  # 用于记录测试集上Top2股票

        self.zzindex = read_zz1000()

        # 自定义的检查早停方法
        self.stop_checker = CM.EarlyStoppingV2(target_name=['log_returns', 'value_return', 'IR_real', 'IR_step','ee'],
                                               target_type=['min', 'max', 'max','max','max'],
                                               patience=args.patience, verbose=True, save_path=args.save_path,
                                               mark=args.mark, save_every_epoch=True, save_every_better=False)
        if self.args.double_val:
            self.stop_checker_2 = CM.EarlyStoppingV2(target_name=['log_returns', 'value_return', 'IR_real', 'IR_step','ee'],
                                                     target_type=['min', 'max', 'max', 'max', 'max'],
                                                     patience=args.patience, verbose=True, save_path=args.save_path,
                                                     mark=args.mark, save_every_epoch=True, save_every_better=False)
        self.epoch_no_better_count = 0
        self._check_align = False  # 为True时, 下一个batch做日线/5分钟日期对齐校验
        self.penality_sum = 0
        self.t1ds_sum = 0

        data_args = Namespace()
        data_args.ms_data_name = args.ms_dataset_name if args.use_minute else ''  # 分钟数据集
        # data_args.data_name = 'ts_orig_104f' if args.DS == 'day104' else 'ts_orig_102f'  # 日线数据集
        data_args.data_name = 'ts_orig_111f_zh' if args.DS == 'day111' else 'wd_260819_395f'
        data_args.jiaoji_mode = 'stop_limit_mode'
        data_args.stop_limit = 10  # 当jiaoji_mode ='stop_limit_mode'生效，每个时间范围内停牌的天数大于stop_limit会被过滤掉
        data_args.zero_check_cols = [10, 11, 12, 13]
        data_args.batch_size = args.batch_size  # b
        data_args.minute_before_num = 0  # 分钟数据前面的天数
        data_args.minute_after_num = 0  # 分钟数据后面的天数
        data_args.day_before_num = PM.cal_seq_len(args)  # 天数据前面的天数
        data_args.day_after_num = 0  # 天数据后面的天数
        if args.use_minute:
            # 第i个交易步的日线回看窗口为日线序列的[i, i+time_step), 分钟回看窗口取其最后ms_time_step天,
            # 即日线序列的[i+time_step-ms_time_step, i+time_step)
            # 分钟序列从日线序列第(time_step-ms_time_step)天开始, 共读取 ms_time_step+episode_len-1 天,
            # 于是第i步的分钟窗口 = 分钟序列的[i, i+ms_time_step), 不会读到任何未来的分钟数据
            data_args.minute_offset_days = args.time_step - args.ms_time_step
            data_args.minute_before_num = args.ms_time_step + args.episode_len - 1
        data_args.sicount = args.M  # s
        data_args.shuffle = True
        data_args.if_bj = False  # 是否返回补集
        #data_args.chengfen = '/data/raw_generated_data/tushare_data/index_weight/index_data_by_date_000852.SH.json'
        data_args.chengfen = ''
        data_args.num_workers = 1
        data_args.stride = args.stride  # 步长
        data_args.m5_first_only = False  # 当取两个数据集时，只返回5min的第一条记录

        self.data_reader = DataReader.create_data_reader(data_args)

        self.vbs = args.batch_size
        if not self.testing:
            data_args_v = copy.deepcopy(data_args)
            data_args_v.batch_size = 1
            if data_args.chengfen == '' :
                data_args_v.sicount = 2000
            else:
                data_args_v.sicount = 500
            data_args_v.stride = args.stride_val
            self.data_reader_val = DataReader.create_data_reader(data_args_v)
            self.vbs = 1

        #self.data_normalizer = DataNormalizer_SWSJ(self.data_reader.args.scaler_file)
        self.check_month = []
        self.recorded_month = []
        self.record_count = 120
        self.cash_price_change = None

        self.cash_dist = [0 for _ in range(11)]

        self.target_gen = TargetGenerator(target_scale=0.8)
        self.tenfactor_target_generator = HistoricalCorrelatedTargetGenerator(target_scale=0.8) # 十因子版本
        self.Index_stocks_generator_train = IndexTargetGenerator()
        self.Index_stocks_generator_val = IndexTargetGenerator(num_stocks_total=2000, num_stocks_select=600)

        self.constraint_loss_fn = SoftConstraintLoss_mask(tolerance=0.05)
        self.mask_sampler = IronBottomMaskSampler(f_dim=10)

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
            state_dict_pre = CM.remove_module_prefix(state_dict_pre)
            self.model.load_state_dict(state_dict_pre, strict=False)
            print(f"{CM.timestr()}****模型参数加载成功:{load_path}****")

    def save_model(self):
        save_path = os.path.join(self.args.save_path, f"{self.args.mark}.pth")
        torch.save(self.model.state_dict(), save_path)

    def cal_pm_value(self, next_w, base_price, target_price):
        """
        计算一个投资组合的未来价值
        :param next_w: 未来时刻的投资组合权重 [B, M+1]
        :param base_price: 基准价格 [B, M]
        :param target_price: 目标价格 [B, M]
        :return: 未来价值 [B]
        """

        relative_price = target_price / base_price
        #torch.where(base_price < 1e-3, torch.ones_like(target_price), target_price / base_price)  #target_price / base_price  # [B,M]  相对价格
        relative_price = torch.cat([self.cash_price_change, relative_price], dim=1)  # [B,1+M]
        value = torch.mul(next_w, relative_price)
        value = value.sum(dim=1)  # [B]
        return value

    def calc_risk_penalty_term(self, R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Calculates the (SR / sigma) * (R_t - mu) term.

        Args:
            R (torch.Tensor): Portfolio returns, shape (B, T).
            eps (float): Small constant for numerical stability.

        Returns:
            torch.Tensor: The calculated term, shape (B, T).
        """
        # 1. Calculate mean (\mu) along time dimension -> Shape: (B, 1)
        mu = R.mean(dim=1, keepdim=True)

        # 2. Calculate standard dev (\sigma) along time dimension -> Shape: (B, 1)
        # Note: unbiased=False uses N in the denominator matching our previous 1/T derivation
        sigma = R.std(dim=1, unbiased=False, keepdim=True)

        # 3. Calculate Sharpe Ratio (SR = \mu / \sigma) -> Shape: (B, 1)
        SR = mu / (sigma + eps)

        # 4. Calculate the final term -> Shape: (B, T)
        # R - mu broadcasts (B, T) - (B, 1) to (B, T)
        term = (SR / (sigma + eps)) * (R - mu)

        return term, SR, sigma, mu, R

    # 4 / 还缺少一个读取数据的函数, 也是在lightning框架下

    def get_zz_value(self, t1Day, t2Day):
        """
        根据交易日,获取中证1000指数的相对价值
        :param t1Day # t1日期 [B]
        :param t2Day # t2日期 [B]
        """

        # self.zzindex 是一个字典表key为str(日期),value为当日的中证1000指数
        B = t1Day.shape[0]
        fv = torch.zeros(B).to(t1Day.device)  # 构造全1张量 [B]
        for i in range(B):
            s = str(int(t1Day[i].item()))
            t = str(int(t2Day[i].item()))
            if not self.zzindex.__contains__(s):
                # 如果当前日期在字典中并不存在,则表明这个日期是一个非交易日,
                # 相应的大盘价值直接设置为1, 下同
                fv[i] = 1.
            elif not self.zzindex.__contains__(t):
                fv[i] = 1.
            else:
                szz = self.zzindex[s]
                tzz = self.zzindex[t]
                fv[i] = tzz / szz
        return fv

    def downside_volatility_loss(self, excess, calibrator: LamCalibrator, k=50.0, eps=1e-8):
        """
        Args:
            excess: Tensor of shape (B, T) 
                    (B = batch size, T = timesteps per episode).
            k: sharpness for softplus(-k * e_t).
            lam: weight for downside volatility penalty.
            eps: numerical stability term.
        Returns:
            loss: scalar tensor
        """
        # --- 1. Asymmetric excess penalty ---
        excess_mean = excess.mean(dim=1)
        loss_down = F.softplus(-k * excess_mean)
        # print('loss_down before', loss_down.shape, loss_down)

        # --- 2. Downside-only volatility penalty ---
        downside = torch.minimum(excess, torch.zeros_like(excess))  # (B, T)
        mean_down = downside.mean(dim=1, keepdim=True)  # (B, 1)
        var_down = ((downside - mean_down) ** 2).mean(dim=1) + eps  # (B,)
        lam = calibrator.calibrate(loss_down, var_down)
        var_down = lam * var_down
        # print('lam', lam ,'var', var_down.shape, var_down)
        # add epsilon for numerical stability
        # vol_penalty = var_down.mean()

        # --- 3. Total loss ---
        loss = loss_down + var_down
        # print('loss_down', loss.shape, loss)
        return loss

    @torch.no_grad()
    def check_minute_align(self, src_data, ms_data):
        """
        校验5分钟数据与日线数据的日期是否对齐: 分钟序列第j天 应等于 日线序列第(time_step-ms_time_step+j)天
        :param src_data: [B, M, L, C] 日线数据
        :param ms_data: [B, M, Lm, 48, C_ms] 5分钟数据
        """
        ms_fields = self.model.representation_ms.normalizer.src_field_list
        ms_date_fields = ('gen_year', 'gen_month', 'gen_day')
        day_date_fields = ('f_d_year', 'f_d_month', 'f_d_day')
        if not all(f in ms_fields for f in ms_date_fields) or \
                not all(f in self.args.src_field_list for f in day_date_fields):
            print(f"{CM.timestr()}分钟/日线数据缺少日期字段, 跳过对齐校验")
            return
        offset = self.args.time_step - self.args.ms_time_step
        Lm = ms_data.shape[2]
        day = src_data[:, :, offset:offset + Lm]  # [B, M, Lm, C]
        bar0 = ms_data[:, :, :, 0]  # 每天第一根5分钟bar [B, M, Lm, C_ms]

        # 年月日分别比较, 避免float32无法精确表示yyyymmdd
        d_ymd = [day[..., self.args.src_field_list.index(f)].round() for f in day_date_fields]
        m_ymd = [bar0[..., ms_fields.index(f)].round() for f in ms_date_fields]
        both_valid = (d_ymd[0] > 0) & (m_ymd[0] > 0)  # 停牌/填充的全0记录不参与比较
        same = (d_ymd[0] == m_ymd[0]) & (d_ymd[1] == m_ymd[1]) & (d_ymd[2] == m_ymd[2])
        bad = both_valid & ~same
        if bad.any():
            b, m, j = [int(v) for v in bad.nonzero()[0]]
            raise ValueError(
                f"日线与5分钟数据未对齐: b={b} m={m} 分钟第{j}天="
                f"{int(m_ymd[0][b, m, j])}-{int(m_ymd[1][b, m, j])}-{int(m_ymd[2][b, m, j])}, "
                f"日线第{offset + j}天={int(d_ymd[0][b, m, j])}-{int(d_ymd[1][b, m, j])}-{int(d_ymd[2][b, m, j])}")
        print(f"{CM.timestr()}日线/5分钟数据对齐校验通过, 有效比较{int(both_valid.sum())}个(股票,日)")

    def forward_an_episode(self, src_data, training=True):
        """
        训练一个episode
        :param src_data: 一个batch的数据
        :param training: 是否训练模式
        :return:
        """

        device = 'cpu'
        if self.args.useGPU:
            device = torch.cuda.current_device()

        # 双数据集reader返回[日线, 5分钟], 纯日线reader只返回日线
        ms_data = None
        if isinstance(src_data, (list, tuple)):
            src_data, ms_data = src_data
        if self.args.use_minute:
            assert ms_data is not None, "use_minute=1 但数据中没有5分钟数据"
            ms_data = ms_data.to(device).to(torch.float32)  # [B, M, ms_time_step+episode_len-1, 48, C_ms]
        src_data = src_data.to(device).to(torch.float32)  # [B,M,L,C]
        if self.args.use_minute and self._check_align:
            self._check_align = False
            self.check_minute_align(src_data, ms_data)

        '''指数成分股权重打印'''
        #weights = src_data[:, :, -1, 104]
        print('原始形状', src_data.shape)

        stock_id = src_data[:,:,0, 0] # 输出ID序列以备debug

    
        '''提取行业通道 和 42个因子收益'''
        '''
        Industry_exp = src_data[..., 119:150]  # 取出31个行业因子暴露
        Industry_exp_sum_check = Industry_exp[0, :, 0, :].sum(dim = -1)
        Industry_exp_ID = src_data[:, :, 80, 119:150]  # 取出31个行业因子暴露
        #print('行业暴露check', Industry_exp_sum_check, Industry_exp)

        style_return = src_data[..., 150:161]  # 取出十个因子收益
        #print('风格因子和国家因子收益', style_return[0,0,0,:])

        Industry_return = src_data[..., 161:192] # 取出31个行业因子收益
        Industry_return_sum_check = Industry_return[0, 0, 0, :].sum(dim=-1)
        #print('行业因子收益check', Industry_return_sum_check, Industry_return[0, 0, 0, :])
        '''

        '''
        Version 2026 Aug 2 6:35pm
        '''
        Barra_return = src_data[:, 0, :, 353: ].unsqueeze(1)  # 取出42个因子收益
        print(' Barra_return',Barra_return[0,0,0,:], "shape", Barra_return.shape)
        
        '''提取十个风格因子的通道'''
        style_factor = src_data[..., 311:321] # 取出十个风格因子
        #factor_step = style_factor[:, :, 0:0 + 80, :]  # 取出最初始80天的所有股票的风格因子暴露
        #core_indices = [0, 2, 4, 6, 9]
        #ten_target_factor = self.tenfactor_target_generator(factor_step) # 基于80天的统计样本，生成合适的随机十因子暴露目标，作为连续二十天交易的目标暴露
        #five_target_factor = ten_target_factor[:, core_indices]

        '''抽样200只虚拟指数'''
        size_exp_Market = style_factor[:, :, self.args.time_step - 1, :]  # 回看窗口最后一天(time_step=80时即原来的第79天)
        if training:
            target_weights = self.Index_stocks_generator_train.generate_index_weights(size_exp_Market)
        else:
            target_weights = self.Index_stocks_generator_val.generate_index_weights(size_exp_Market)
        index_weights = target_weights.squeeze(-1)
        B, M = index_weights.shape
        # 1. 创建一个代表现金权重的全 0 张量
        # shape: [B, 1] 必须保证 dtype 和 device 与 index_weights 绝对一致
        cash_weight = torch.zeros((B, 1), dtype=index_weights.dtype, device=index_weights.device)

        # 2. 将现金权重拼接到个股权重的最前面 (dim=1)
        # 拼接后的 shape: [B, 1 + M]
        index_weights_with_cash = torch.cat([cash_weight, index_weights], dim=1)

        size_factor = src_data.clone() # 作为原始通道，为了提取市值通道的因子暴露

        B, M, L, C = src_data.shape
        '''
        if self.args.train_with_noise == 1 and training:
            # 训练时加入噪声
            # 需要添加噪音的通道索引范围
            start_idx = self.args.open_price_index  # 开盘价索引,是第1个需要添加噪声的通道
            noisey_channels = C - start_idx - 2  # 需要添加噪声的通道数
            src_data = CM.noisy_data(src_data, start_idx, noisey_channels, method='random')
        '''

        active_mask = self.mask_sampler.sample_mask(B, device)

        asset_num = M + 1  # 资产数量
        v = torch.ones(B).to(device)  # 初始投资组合的价值 [B]
        history_value = [v]  # 记录每个历史时刻的投资组合总价值
        history_value_c1 = [v]
        history_value_c2 = [v]
        LBW = self.args.time_step  # 历史长度

        # 创建代表现金价格变化的张量，值永远为1，因为现金价格不变
        self.cash_price_change = torch.ones(B, 1).to(device)  # [B, 1]
        # 随机分配权重
        w = torch.rand(B, asset_num).to(device)  # 初始投资组合的权重 [B, asset_num]
        w = F.softmax(w, dim=1)  # 归一化权重
        history_portfolio = [w]  # 记录每个历史时刻的投资组合权重
        wloss_episode = 0.0  # 记录调整仓位合理性loss

        show_process = training and self.trainer.is_global_zero and random.randint(1, 30) == 1
        loss_list = []  # 记录每个时刻的loss
        # loss_list2 = []
        prev_penalty = None  # 记录上一时刻的惩罚项
        penality_count = 0  # 记录惩罚事件的次数
        t1ds_count = 0  # 记录T1DS事件的次数

        fool_buy_price = src_data[:, :, LBW, self.args.close_price_index].squeeze()  # 假定傻瓜平均买入价
        fool_sell_price = src_data[:, :, LBW + self.args.episode_len,
                          self.args.close_price_index].squeeze()  # 假定傻瓜平均卖出价
        full_realy_day = src_data

        fooler_value = fool_sell_price / fool_buy_price
        fooler_value = fooler_value.sum(dim=-1) / M  # 计算傻瓜策略的收益率 [B] B个样本,每个样本的傻瓜收益

        weight_entropy = 0.0  # 记录权重的熵
        weight_cash = 0.0  # 记录权重中现金的占比
        BigStockCount0 = 0.  # 记录重仓股票数量
        BigStockCount2 = 0.  # 记录重仓股票数量
        BigStockCount5 = 0.  # 记录重仓股票数量
        BigStockCount10 = 0.  # 记录重仓股票数量
        BigStockCount20 = 0.  # 记录重仓股票数量
        TopStock1 = 0.
        TopStock2 = 0.
        weight_change_total = 0.  # 记录权重变化的总量
        fv_list = []  # 记录每个时刻的fv
        excess_earnings_list = []  # 记录每个时刻的超额收益率
        RMSE_list = []
        MAE_list = []

        # For Dirichlet and DSR
        burn_in = 30  # 30 days warm-up for Simple Moving Average
        eta = 0.01  # EMA decay rate for DSR

        # --- Tracking Variables ---
        burn_in_returns = []
        log_probs_list = []
        rewards_list = []

        A_prev = None  # Moving average of returns
        B_prev = None  # Moving average of squared returns

        SR_excess_period_list = []
        IR_excess_period_list = []
        SR_step_list = []
        IR_step_list = []
        SR_period_epoch_list = []
        IR_period_epoch_list = []
        Factor_deviation_list = []
        holding_price_list = []

        #ee, next_w, next_relative_price, t1Day,

        daily_ee_list = []
        daily_nextw_list = []
        daily_next_relative_price_list = []
        daily_date_list = []

        loss_weights_list = []

        train_step = 1
        val_step = 1


        # 循环进行episode_len次前向过程
        total_trade_count = 0  # 记录总交易次数
        for i in range(0, self.args.episode_len, self.args.tr_days):
            total_trade_count += 1
            # print('episode_i_check', i)
            
            factor_step = style_factor[:, :,  i * train_step : i * train_step  + LBW, :] # 取出当前回看窗口序列的十个风格因子，完整的80天回看进入模型，作ID embedding
            factor_last_step = factor_step[:,:,-1,:] #取出回看的第一天，作为因子暴露的计算标的

            # 1. 定义你要保留的 5 个核心因子的索引
            # 对应: Size, Momentum, Book_to_Price, Growth, Residual_Volatility
            #core_indices = [0, 2, 4, 6, 9]

            # 1. 广播相乘：[B, M, 1] * [B, M, F] -> 得到 [B, M, F] (每只股票贡献的加权因子暴露)
            # 因为非成分股权重为0，所以乘完之后它们贡献的值全部是 0
            weighted_factors = target_weights * factor_last_step

            # 2. 沿股票维度 (dim=1) 聚合求和，得到最终指数暴露
            # shape: [B, M, F] -> [B, F]
            index_factor_exposure = weighted_factors.sum(dim=1)
            #five_target_factor = index_factor_exposure[:, core_indices]

            # 2. 一行代码完成剔除与降维
            # 取所有 Batch, 所有 M只股票，以及指定的 5 个特征维度
            #factor_exp_5 = factor_last_step[:, :, core_indices]


            inputSeq = full_realy_day[:, :, i * train_step :i * train_step  + LBW, :]  # 取出前面time_step天的真实天数
            # 与日线回看窗口最后ms_time_step天对齐的5分钟数据 [B, M, ms_time_step, 48, C_ms]
            msSeq = ms_data[:, :, i * train_step: i * train_step + self.args.ms_time_step] if self.args.use_minute else None
            futureSeq = full_realy_day[:, :, i * train_step  + LBW + 1:i * train_step  + LBW + 1 + val_step, :]
            Barra_return_episode = Barra_return[:, :, i * train_step  : i * train_step  + LBW, :]

      
            
            day_t0 = inputSeq[:, :, -1, :].clone()  # 回看窗口最后一天(t0)的全部数据 [B, M, C]
            day_t1 = full_realy_day[:, :,i * train_step  + LBW, :].clone()  # 未来信息第1天(t1)的全部数据
            day_t2 = futureSeq[:, :, self.args.tr_days - 1, :].clone()  # 未来信息第2天(t2)的全部数据
            day_tm = futureSeq[:, :, val_step - 1, :].clone()  # 未来信息第m天(tm)的全部数据

            close_t0 = day_t0[:, :, self.args.close_price_index]  # 回看窗口最后一天(t0)的收盘价 [B, M]
            open_t1 = day_t1[:, :, self.args.open_price_index]  # 未来信息第1天(t1)的开盘价
            p935_t1 = day_t1[:, :, self.args.p35_close_index]  # 未来信息第1天(t1)的9:35分价格

            target_price_t1 = day_t1[:, :, self.args.price_target_index]  # 未来信息第1天(t1)的目标价 [B, M]
            target_price_t2 = day_t2[:, :, self.args.price_target_index]  # 未来信息第2天(t2)的目标价 [B, M]
            target_price_tm = day_tm[:, :, self.args.price_target_index]  # 未来信息第m天(tm)的目标价 [B, M]
            target_price_allfuture = futureSeq[:, :, :, self.args.price_target_index]  # 未来信息的目标价 [B, M, pred_len]

            stock_type_t1 = day_t1[:, :, -2]  # [B,M] 未来信息第1天(t1)的的股票类型
            # 需要将这个数据反归一化,并转换为int类型
            stock_type_t1 = stock_type_t1 * self.args.limit_mark_type + self.args.limit_mark_base  # [B,M]
            stock_type_t1 = stock_type_t1.int()  # [B,M], 取值范围为1-5  {'主板': 1, '创业板': 2, '北交所': 3, '科创板': 4, 'ST': 5}

            inputSeq = inputSeq.clone()

            # PM_SM模型需要原始数据
            globalSeq = inputSeq.clone()
            globalSeq[:, :, :, self.args.cq_index] = 0.  # 除权除息数据不参与模型训练
            futureSeq = futureSeq.clone()
            futureSeq[:, :, :, self.args.pre_known_future:] = 0.0

            # 取出当前时刻的投资组合权重
            prev_w = history_portfolio[-1]

            # 数据处理： 将前time_step天的时序数据中所有价格类字段（共5个）全部除以T1开盘价，变成相对价格
            norm_base = p935_t1 if self.args.price_target == '935' else open_t1
            #inputSeq = PM.make_price_relative_v2(inputSeq, self.args.price_channels, norm_base)

            # 取出当前时刻的投资组合价值
            if self.args.dnv == 1:
                # 除了前面的日期类字段,最末尾2个字段(类型,pad标识)外, 其他列都除以t1日当天对应的列,全部变成相对波动曲线
                inputSeq = PM.make_price_relative_v3(inputSeq, day_t0)

            # 基于111通道，去掉 5 个通道 的操作
            
            # print('before train', inputSeq[0,0,0,10:40])
            #next_w = self.model(prev_w, inputSeq, Barra_return_episode, factor_step, index_factor_exposure, active_mask, Industry_exp_ID)  # [B,asset_num]
            next_w = self.model(
                    prev_w=prev_w,                              # [B, 1+M]
                    raw_seq=inputSeq,                           # [B, M, L, C] 
                    target_exp=index_factor_exposure,           # [B, 10] 目标 Barra 暴露
                    active_mask=active_mask,
                    target_weights = target_weights,
                    ms_seq=msSeq,                               # [B, M, ms_time_step, 48, C_ms]
                )

            diff_weights = next_w - prev_w  # (B, M)
            loss_weights = (diff_weights ** 2).sum(dim=-1)  # (B,) 对M求和
            loss_weights_list.append(loss_weights)

            #portfolio_exp = next_w[:, 1:]
            #equal_weight_exposure = size_factor_Last_step.mean(dim=-1)
            #portfolio_exp = torch.sum(portfolio_exp * size_factor_Last_step, dim=-1)  # shape -> (B,)
            #print('策略因子暴露', portfolio_exp.shape, portfolio_exp, equal_weight_exposure)

            batch_constraint_loss, actual_exp = self.constraint_loss_fn(next_w, factor_last_step, index_factor_exposure, active_mask)
            Factor_deviation_list.append(batch_constraint_loss)
            #print('当日因子约束损失', batch_constraint_loss)


            # print('date_check', date_str, 'next_w', next_w, inputSeq[0,0,0,0:40])
            # print('next_w', next_w, inputSeq, futureSeq, globalSeq)
            if random.randint(1, 96) == 1:
                print('exp_match', 'actual_exp', actual_exp, 'target_exp', index_factor_exposure, 'mask', active_mask)

            # 我们假定交易都在指定的t1日某个时刻瞬间完成, 但需要先判断在这一时刻,相对于t0日收盘价,是否已经处于涨停/跌停状态
            # 在本实验中, 由于是以每日5分钟均价成交,所以涨跌停暂时以开盘价为依据: 即如果开盘价已经涨/跌停, 则认为
            # 在接下来的5分钟内, 也无法对涨停股票加仓或者对跌停股票减仓
            is_up_stop, is_down_stop = CM.cal_if_updown_stop(close_t0, norm_base, stock_type_t1)

            if self.testing:
                # 将权重中过小的值全部清0
                next_w = CM.smooth_weight(next_w, smooth_weight=self.args.minw)
                # 做推理测试时,对输出权重进行再处理,去除所有非法的加/减仓操作
                next_w = CM.deal_illeagle_weight(prev_w, next_w, is_up_stop, is_down_stop)

            # 计算哪些资产的权重上涨了
            up_weight = (next_w - prev_w) >= 0.005  # [B,asset_num]
            up_weight = up_weight[:, 1:]  # 去掉现金，只保留股票，确保形状为[B,M]

            # 计算下一时刻价格变化率 [B,M]
            next_relative_price = target_price_t2 / target_price_t1
            #torch.where(target_price_t1 < 1e-3, torch.ones_like(target_price_t2), target_price_t2 / target_price_t1)   #target_price_t2 / target_price_t1  # [B,M]  相对价格
            next_relative_price_5 = target_price_tm / target_price_t1
            #torch.where(target_price_t1 < 1e-3, torch.ones_like(target_price_t2), target_price_tm / target_price_t1)    #target_price_tm / target_price_t1

            if self.testing:
                # 测试模型时,还要考虑padding的影响, 如果买入日或者交割日是padding的数据,表示真实交易无法发生
                # 在这种情况下,将相对价格置为1
                padd_mask_t2 = (day_t2[:, :, -1] > 0.0001)  # [B,M]
                padd_mask_t1 = (day_t1[:, :, -1] > 0.0001)  # [B,M]
                pm = padd_mask_t1 | padd_mask_t2  # [B,M]
                next_relative_price[pm] = 1.0  # 置为1.0,表示真实交易无法发生

            #print('self.args.fvtype', self.args.fvtype)
            #breakpoint()

            # 4 / 修改训练框架代码中关于fv_day的计算部分:
            if self.args.fvtype == 'FOOL':
                # 大盘价值的计算方法为'FOOL',表示大盘平均权重价值
                fv_day = next_relative_price.sum(dim=-1) / M  # [B]  相对价格的总和
            elif self.args.fvtype == 'ZZ1000':
                # 根据中证1000指数的变化来计算大盘价值
                # 在前面的代码中已经有计算好的交易日t1的日期张量如下:
                # 根据中证1000指数的变化来计算大盘价值
                # 在前面的代码中已经有计算好的交易日t1的日期张量如下:
                
                tradeDay = self.data_normalizer.get_batch_date(day_t1)  # 生成交易日期张量 [B, M]
                # 现在需要得到t2日的日期张量
                t1Day = tradeDay[:, 0]  # [B]
                t2Day = self.data_normalizer.get_batch_date(day_t2)  # [B, M]
                t2Day = t2Day[:, 0]  # [B]
                fv_day = self.get_zz_value(t1Day, t2Day)
                t6Day = self.data_normalizer.get_batch_date(day_tm)  # [B, M]
                t6Day = t6Day[:, 0]  # [B]
                fv_day5 = self.get_zz_value(t1Day, t6Day)
                
            else:
                # 否则为'WEIGHT', 表示对价值进行加权求和
                # next_relative_price本是相对价值(1附近的), 减去1后就变成了涨跌幅(0附近)
                # 然后再乘上各自股票的市值占比, 就得到了每只股票加权后的涨跌幅
                fv_day = (next_relative_price - 1.) * fv_weight  # [B, M]
                # 对样本内所有股票加权涨跌幅求和, 则得到了加权之后的大盘涨跌幅, 它仍然是在0附近的
                # 最后再加上1, 则得到了加权之后的大盘价值
                fv_day = torch.sum(fv_day, dim=1) + 1.  # [B]

            # fv_day = next_relative_price.sum(dim=-1) / M  # [B]  相对价格的总和
            temp = next_relative_price.sum(dim=-1) / M  # [B]  相对价格的总和
            # print('fv_day', fv_day, 'temp', temp)
            fv_list.append(round(fv_day.mean().item(), 4))

            # 对相对价格做修正，以惩罚错误的行为(--给涨停股票增加持仓)
            ccrela_price = next_relative_price.clone()  # [B,M]
            ccrela_price_5steps = next_relative_price_5.clone()  # [B,M]

            # 20241214--将惩罚措施统一,无论是训练/验证/测试,都使用完全相同的惩罚措施
            # 测试模型时,对涨停且加仓的股票,惩罚措施为: 相对价格设置为1.0 (!!!持续惩罚!!!)
            this_step_panalty = is_up_stop & up_weight  # 本次需要惩罚的股票 [B,M]
            if prev_penalty is not None:
                keep_weight = next_w[:, 1:] >= 0.005
                this_step_panalty[prev_penalty & keep_weight] = True
            ccrela_price[this_step_panalty] = 1.0 if self.testing else 0.95
            ccrela_price_5steps[this_step_panalty] = 1.0 if self.testing else 0.95


            # ==========================================
            # Experiment B: CS 2σ Winsorization
            # ==========================================

            if training:
               
                # 2. 原始未来收益
                future_return = (
                    target_price_tm
                    / target_price_t1.clamp_min(1e-6)
                    - 1.0
                )

                # 3. 横截面3σ截断
                clipped_return, clip_metrics = (
                    cs_clip_return_2sigma(
                        future_return=future_return,
                        n_sigma=3.0,
                    )
                )

                # 4. 截断收益转换为相对价格
                clipped_relative_price = (
                    1.0 + clipped_return
                )

                # 5. 保留原来的非法交易惩罚
                clipped_relative_price = torch.where(
                    this_step_panalty,
                    torch.full_like(
                        clipped_relative_price,
                        0.95,
                    ),
                    clipped_relative_price,
                )

                # 6. 添加现金通道
                clipped_relative_price = torch.cat(
                    [
                        self.cash_price_change,
                        clipped_relative_price,
                    ],
                    dim=1,
                )

                # 7. 截断后的组合价值
                value_1_to_m_clipped = (
                    next_w * clipped_relative_price
                ).sum(dim=1)

                clipped_relative_price_benchmark = torch.cat(
                    [
                        self.cash_price_change,
                        1.0 + clipped_return,
                    ],
                    dim=1,
                )
    
                fv_daym_indexReturn_clipped = (
                    index_weights_with_cash
                    * clipped_relative_price_benchmark
                ).sum(dim=1)
                '''
                print("\n[1] 原始收益率")
                print(f"Mean : {future_return.mean().item():+.6f}")
                print(f"Std  : {future_return.std(unbiased=False).item():.6f}")
                print(f"Min  : {future_return.min().item():+.6f}")
                print(f"Max  : {future_return.max().item():+.6f}")

                print("\n[2] 3σ 截断之后")
                print(f"Mean : {clipped_return.mean().item():+.6f}")
                print(f"Std  : {clipped_return.std(unbiased=False).item():.6f}")
                print(f"Min  : {clipped_return.min().item():+.6f}")
                print(f"Max  : {clipped_return.max().item():+.6f}")
                '''


            if self.testing:
                # 统计惩罚事件的次数
                penality_count += this_step_panalty.sum().item()
                prev_penalty = this_step_panalty.clone()
                down_weight = (next_w - prev_w) <= -0.001  # [B,asset_num]
                down_weight = down_weight[:, 1:]  # 去掉现金，只保留股票，确保形状为[B,M]
                down_weight = down_weight & is_down_stop  # [B,M]  T1日开盘即跌停且模型预测要减仓的股票
                t1ds_count += down_weight.sum().item()

            # 计算下一时刻的投资组合价值
            next_relative_price = torch.cat([self.cash_price_change, next_relative_price], dim=1)  # [B,asset_num]
            ccrela_price = torch.cat([self.cash_price_change, ccrela_price], dim=1)  # [B,asset_num]

            value_1_to_2 = torch.mul(next_w, ccrela_price)
            value_1_to_2 = value_1_to_2.sum(dim=1)  # [B]

            next_relative_price_5 = torch.cat([self.cash_price_change, next_relative_price_5], dim=1)  # [B,asset_num]
            ccrela_price_5steps = torch.cat([self.cash_price_change, ccrela_price_5steps], dim=1)  # [B,asset_num]

            value_1_to_m = torch.mul(next_w, ccrela_price_5steps)
            value_1_to_m = value_1_to_m.sum(dim=1)  # [B]

            fv_day5_indexReturn = self.cal_pm_value(index_weights_with_cash, target_price_t1, target_price_tm)  # [B]

            

            # 计算这次交易的仓位变化量,并计算交易成本折扣
            weight_change = torch.abs(next_w - prev_w)  # [B,asset_num]
            weight_change = weight_change[:, 1:]  # 去掉现金，只保留股票，确保形状为[B,M]
            weight_change = weight_change.sum(dim=1)  # [B]
            trade_cost = weight_change * self.args.cost_factor  # [B]
            trade_cost_real = weight_change * 0.001  # [B]
            weight_change_total += weight_change.mean().item()

            step_value = value_1_to_2 * self.args.vf + value_1_to_m * (1 - self.args.vf)

            fv_day5 =  target_price_tm / target_price_t1
            #torch.where(target_price_t1 < 1e-3, torch.ones_like(target_price_tm), target_price_tm / target_price_t1) # target_price_tm / target_price_t1
            fv_day5 = fv_day5.sum(dim=-1) / M

            SR_excess = value_1_to_m * (1. - trade_cost_real) - 1.0
            #IR_excess_real = value_1_to_m * (1. - trade_cost_real) - fv_day5_indexReturn
            #IR_excess = step_value * (1. - trade_cost) - fv_day
            #IR_excess = value_1_to_m * (1. - trade_cost) - fv_day5
            #IR_excess = value_1_to_m * (1. - trade_cost) - fv_day5_indexReturn
            #IR_excess = value_1_to_m * (1. - trade_cost) - 1.0

            if training:
                # Experiment B：训练使用截断后的收益
                IR_excess = (
                    value_1_to_m_clipped
                    * (1.0 - trade_cost)
                    - fv_daym_indexReturn_clipped
                )
                IR_excess_real = value_1_to_m_clipped * (1. - trade_cost_real) - fv_daym_indexReturn_clipped
            else:
                # 测试使用真实收益
                IR_excess = (
                    value_1_to_m
                    * (1.0 - trade_cost)
                    - fv_day5_indexReturn
                )
                IR_excess_real = value_1_to_m * (1. - trade_cost_real) - fv_day5_indexReturn

            excess_earnings = value_1_to_m * (1 - trade_cost_real) - fv_day5_indexReturn # 无交易费超额
            excess_earnings_list.append(excess_earnings.mean().item())

            SR_step_list.append(SR_excess)
            IR_step_list.append(IR_excess)



            SR_period_epoch_list.append(weight_change)
            IR_period_epoch_list.append(IR_excess_real)

            if self.testing:
                # 推理测试时,根据参数count_cost决定是否输出交易费折扣后的目标价值
                history_value.append(value_1_to_2)  # 不计成本
                history_value_c1.append(value_1_to_2 * (1. - weight_change * self.args.tc1))  # 考虑交易成本0.001
                history_value_c2.append(value_1_to_2 * (1. - weight_change * self.args.tc2))  # 考虑交易成本0.003
            else:
                # 训练和验证模型时, 价值的计算不考虑交易成本
                history_value.append(value_1_to_2)

            ee = value_1_to_2 - fv_day
    


            history_portfolio.append(next_w)

            # 随机输出一个权重组合，用于观察训练过程
            if show_process:
                debug_portfolio = next_w[0].tolist()
                debug_w = [round(w, 4) for w in debug_portfolio]
                debug_w2 = CM.listnum_tostr(debug_portfolio, 4, 7)
                rlp = next_relative_price[0].tolist()
                debug_p = CM.listnum_tostr(rlp, 4, 7)
                # print(f"P{i}:{debug_p}")
                # print(f"W{i}:{debug_w2}----m={max(debug_w)},{min(debug_w)},v={round(value_1_to_2[0].item(), 4)}")

            # 计算权重的熵以及头部权重中股票的数量
            if self.testing:
                we, ce = CM.cal_weight_entropy(next_w)
                weight_entropy += we
                weight_cash += ce
                b0, b2, b5, b10, b20, t1, t2 = CM.cal_big_stock_count(next_w, smooth_weight=1e-3)
                BigStockCount0 += b0
                BigStockCount2 += b2
                BigStockCount5 += b5
                BigStockCount10 += b10
                BigStockCount20 += b20
                TopStock1 += t1
                TopStock2 += t2

                # 现金的分布情况
                cash_taken = next_w[0, 0].item()
                # 将这个现金转换为0~9范围的整数
                cash_taken = int(cash_taken * 10)
                self.cash_dist[cash_taken] += 1

            if self.use_w_loss == 1:
                pass
            elif self.use_w_loss == 2:
                # 计算两个权重之间的KL散度
                wloss = CM.distribution_distance(next_w, prev_w, method='KL')
                wloss_episode += wloss.mean()

        history_value = torch.stack(history_value, dim=1)  # [B,episode_len+1]
        eipsode_value = history_value.prod(dim=1)  # 最后时刻的投资组合价值
        if show_process:
            print(f"EPValue:{eipsode_value[0].item():.4f}")
        episode_total_value = eipsode_value.mean().item()  # 整个Batch的各个episode的平均价值

        '''
        Loss designing:
        _epoch period_list 是每一次持仓被记为强制持仓，迭代20步的统计评估
        _step 才是原始的迭代20步的持续交易的实际评估
        '''

        SR_epoch = torch.stack(SR_period_epoch_list, dim=1)
        sr = SR_epoch.mean(dim=1)

        IR_epoch = torch.stack(IR_period_epoch_list, dim=1)
        ir = IR_epoch.mean(dim=1) / IR_epoch.std(dim=1)

        SR_step = torch.stack(SR_step_list, dim=1)
        SR_step = SR_step.mean(dim=1) / SR_step.std(dim=1)

        
        

        SR_soft_step = torch.stack(SR_step_list, dim=1)
        SR_excess = SR_soft_step.mean(dim=1)
        k = 300
        loss_softplus = F.softplus(-k * SR_excess)

        #factor_exp_factor = IR_step.mean().detach()
      

        loss_weights_step = torch.stack(loss_weights_list, dim=1)
        #loss_weights_step = 8 * IR_step.detach() * loss_weights_step.mean(dim=1)

        turnover_penalty = 10
        loss_weights_step_loss = turnover_penalty * loss_weights_step.mean(dim=1)

        factor_deviation_episode = torch.stack(Factor_deviation_list, dim=1)
        factor_deviation_episode = factor_deviation_episode.mean(dim=1)
        IR_step = torch.stack(IR_step_list, dim=1)


        loss_episode, dynamic_metrics = self.dynamic_loss_fn(
                IR_excess=IR_step,
                factor_deviation= factor_deviation_episode,
            )
        
        
        IR_step = IR_step.mean(dim=1) / IR_step.std(dim=1)
        IR_step_for_train = IR_step.mean()
        factor_deviation_episode_for_train = factor_deviation_episode.mean().detach().item()


        if random.randint(1, 60) == 1:
            term, SR, sigma, mu, R = self.calc_risk_penalty_term(SR_epoch)
            print('SR term check', 'SR term:', term[0, :], 'SR:', SR[0, :], 'Sigma:', sigma[0, :], 'Average', mu[0, :],
                  'Real excess', R[0, :])

        if self.use_w_loss > 0 and training:
            # 计算整个batch所有episode的平均权重损失
            wloss_episode = wloss_episode / total_trade_count  # 权重损失的平均值
            loss_episode = loss_episode + self.args.w_loss_factor * wloss_episode  # 加上权重损失
            # loss_episode = weighted_loss_log_scale(loss_episode,wloss_episode)

        if show_process:
            print(f"BL:{loss_episode:.6f}:BV-avg:{episode_total_value:.4f}")

        trade_process = []
        last_value = []

        trade_process_c1 = []
        last_value_c1 = []

        trade_process_c2 = []
        last_value_c2 = []

        if self.testing:
            history_value_c1 = torch.stack(history_value_c1, dim=1)  # [B,episode_len+1]
            eipsode_value_c1 = history_value_c1.prod(dim=1)  # 最后时刻的投资组合价值
            history_value_c2 = torch.stack(history_value_c2, dim=1)  # [B,episode_len+1]
            eipsode_value_c2 = history_value_c2.prod(dim=1)  # 最后时刻的投资组合价值

            # 测试模型时,记录每个交易过程
            history_value = history_value[:, 1:]
            for i in range(B):
                sample_process = []
                for j in range(total_trade_count):
                    v = round(history_value[i, j].item(), 4)
                    sample_process.append(v)
                trade_process.append(sample_process)
                v = round(eipsode_value[i].item(), 4)
                last_value.append(v)

            # 测试模型时,记录每个交易过程
            history_value_c1 = history_value_c1[:, 1:]
            for i in range(B):
                sample_process = []
                for j in range(total_trade_count):
                    v = round(history_value_c1[i, j].item(), 4)
                    sample_process.append(v)
                trade_process_c1.append(sample_process)
                v = round(eipsode_value_c1[i].item(), 4)
                last_value_c1.append(v)

            # 测试模型时,记录每个交易过程
            history_value_c2 = history_value_c2[:, 1:]
            for i in range(B):
                sample_process = []
                for j in range(total_trade_count):
                    v = round(history_value_c2[i, j].item(), 4)
                    sample_process.append(v)
                trade_process_c2.append(sample_process)
                v = round(eipsode_value_c2[i].item(), 4)
                last_value_c2.append(v)

        weight_entropy = weight_entropy / total_trade_count  # 计算平均的权重熵
        weight_cash = weight_cash / total_trade_count  # 计算平均的权重的现金占比
        BigStockCount0 = BigStockCount0 / total_trade_count  # 计算平均的重仓股票数量
        BigStockCount2 = BigStockCount2 / total_trade_count  # 计算平均的重仓股票数量
        BigStockCount5 = BigStockCount5 / total_trade_count  # 计算平均的重仓股票数量
        BigStockCount10 = BigStockCount10 / total_trade_count  # 计算平均的重仓股票数量
        BigStockCount20 = BigStockCount20 / total_trade_count  # 计算平均的重仓股票数量
        TopStock1 = TopStock1 / total_trade_count  # 计算平均的头部股票数量
        TopStock2 = TopStock2 / total_trade_count  # 计算平均的头部股票数量
        weight_change_total = weight_change_total / total_trade_count  # 计算平均的权重变化

        sr = sr.mean().item()
        ir = ir.mean().item()
        sr_step = SR_step.mean().item()
        ir_step = IR_step.mean().item()
        size_exp = factor_deviation_episode.mean().item()

        track_loss = loss_weights_step.mean().item()



        anaInfo = {
            'weight_entropy': weight_entropy,
            'weight_cash': weight_cash,
            'BigStockCount': [BigStockCount0, BigStockCount2, BigStockCount5, BigStockCount10, BigStockCount20,
                              TopStock1, TopStock2],
            'weight_change_total': weight_change_total,
            't1ds_count': t1ds_count,
            'trade_process': trade_process,
            'last_value': last_value,
            'penality_count': penality_count,
            'fooler_value': fooler_value,
            'fv_list': [fv_list],
            'ee': sum(excess_earnings_list) / len(excess_earnings_list),

            'sr': sr,
            'ir': ir,
            'sr_step': sr_step,
            'ir_step': ir_step,
            'size_exp' : size_exp,

            'track_loss': track_loss,

            # 'Date':date_str,
            # 'RMSE': sum(RMSE_list) / len(RMSE_list),
            # 'MAE': sum(MAE_list) / len(MAE_list),
            'trade_process_c1': trade_process_c1,
            'last_value_c1': last_value_c1,
            'trade_process_c2': trade_process_c2,
            'last_value_c2': last_value_c2,
        }

        return loss_episode, episode_total_value, anaInfo, IR_step_for_train, factor_deviation_episode_for_train, active_mask

    def training_step(self, batch, batch_idx):
        # 一次训练操作 one batch
        tb = self.trainer.num_training_batches
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        if batch_idx == 0:
            self._check_align = True
            self.start_time = time.time()
            self.train_batch_loss = []
            self.train_batch_value = []
            self.train_batch_ee = []
            self.train_batch_sr = []
            self.train_batch_ir = []
            self.train_batch_sr_step = []
            self.train_batch_ir_step = []
            self.train_batch_size_step = []
            self.train_batch_track_loss = []

        showbatch = CM.make_fit_showbatch(None, tb)
        batch_loss, batch_value, anaInfo, IR_loss_tensor, deviation_tensor, active_mask = self.forward_an_episode(batch, training=True)

        # ==========================================
        # 极速单次反向传播
        # ==========================================

        ee = torch.tensor(anaInfo['ee'], device=batch_loss.device)
        ee = self.all_gather(ee).mean().item()
        self.train_batch_ee.append(ee)

        sr = torch.tensor(anaInfo['sr'], device=batch_loss.device)
        sr = self.all_gather(sr).mean().item()
        self.train_batch_sr.append(sr)

        ir = torch.tensor(anaInfo['ir'], device=batch_loss.device)
        ir = self.all_gather(ir).mean().item()
        self.train_batch_ir.append(ir)

        sr_step = torch.tensor(anaInfo['sr_step'], device=batch_loss.device)
        sr_step = self.all_gather(sr_step).mean().item()
        self.train_batch_sr_step.append(sr_step)

        track_loss = torch.tensor(anaInfo['track_loss'], device=batch_loss.device)
        track_loss = self.all_gather(track_loss).mean().item()
        self.train_batch_track_loss.append(track_loss)

        ir_step = torch.tensor(anaInfo['ir_step'], device=batch_loss.device)
        ir_step = self.all_gather(ir_step).mean().item()
        self.train_batch_ir_step.append(ir_step)

        size_step = torch.tensor(anaInfo['size_exp'], device=batch_loss.device)
        size_step = self.all_gather(size_step).mean().item()
        self.train_batch_size_step.append(size_step)

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
            avg_ee = sum(self.train_batch_ee) / len(self.train_batch_ee)
            avg_sr = sum(self.train_batch_sr) / len(self.train_batch_sr)
            avg_ir = sum(self.train_batch_ir) / len(self.train_batch_ir)
            avg_sr_step = sum(self.train_batch_sr_step) / len(self.train_batch_sr_step)
            avg_ir_step = sum(self.train_batch_ir_step) / len(self.train_batch_ir_step)
            avg_size_step = sum(self.train_batch_size_step) / len(self.train_batch_size_step)
            avg_track_loss = sum(self.train_batch_track_loss) / len(self.train_batch_track_loss)
            # avg_mae = sum(self.train_batch_mae) / len(self.train_batch_mae)
            self.output_info(
                f"{CM.timestr()}TrainEpoch:{epoch}/{self.args.train_epochs}:Batch:{rb}/{tb}:loss:{avg_loss:.6f}:value:{avg_value:.6f}:ee:{avg_ee:.6f}:sr:{avg_sr:.6f}:ir:{avg_ir:.6f}:sr_step:{avg_sr_step:.6f}:ir_step:{avg_ir_step:.6f}:Exp_dev:{avg_size_step:.6f}:track_loss:{avg_track_loss:.6f}:time:{rtime}s")

        return {'loss': batch_loss}

    '''
    def on_after_backward(self):
        if self.trainer.is_global_zero:  # 多卡时只打印一次
            unused = []
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    unused.append((name, tuple(p.shape)))

            if unused:
                print("⚠️ Unused parameters detected:")
                for name, shape in unused:
                    print(f"   • {name} | shape={shape}")'''

    def on_train_epoch_end(self):
        # 训练集上完成一个epoch
        epoch_loss = sum(self.train_batch_loss) / len(self.train_batch_loss)
        epoch_value = sum(self.train_batch_value) / len(self.train_batch_value)
        epoch_sr = sum(self.train_batch_sr) / len(self.train_batch_sr)
        epoch_ir = sum(self.train_batch_ir) / len(self.train_batch_ir)
        epoch_sr_step = sum(self.train_batch_sr_step) / len(self.train_batch_ir_step)
        epoch_ir_step = sum(self.train_batch_ir_step) / len(self.train_batch_ir_step)
        epoch_size_step = sum(self.train_batch_size_step) / len(self.train_batch_size_step)
        epoch_track_loss= sum(self.train_batch_track_loss) / len(self.train_batch_track_loss)

        self.train_batch_loss = []
        self.train_batch_value = []
        self.train_batch_sr = []
        self.train_batch_ir = []
        self.train_batch_sr_step = []
        self.train_batch_ir_step = []
        self.train_batch_size_step = []
        self.train_batch_track_loss = []

        self.output_info(
            f"{CM.timestr()}MPE_Train:{self.trainer.current_epoch + 1 + self.args.epno}/{self.args.train_epochs}:loss:{epoch_loss:.6f}:value:{epoch_value:.6f}:Exp_dev:{epoch_size_step:.6f}:Turnover:{epoch_sr:.6f}:IR_real:{epoch_ir:.6f}:SR_step:{epoch_sr_step:.6f}:IR_step:{epoch_ir_step:.6f}:Track_loss:{epoch_track_loss:.6f}")

    def validation_step(self, batch, batch_idx):
        tb = self.trainer.num_val_batches[0]
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        # 一次训练操作 one batch
        if batch_idx == 0:
            self._check_align = True
            self.start_time = time.time()
            self.val_batch_loss = []
            self.val_batch_value = []
            self.val_batch_ee = []
            self.val_batch_sr = []
            self.val_batch_ir = []
            self.val_batch_sr_step = []
            self.val_batch_ir_step = []
            self.val_batch_size_step = []
            self.val_batch_track_loss = []
            # self.val_batch_mae = []

        showbatch = CM.make_fit_showbatch(None, tb)
        batch_loss, batch_value, anaInfo, IR_loss_tensor, deviation_tensor, active_mask = self.forward_an_episode(batch, training=False)
        ee = torch.tensor(anaInfo['ee'], device=batch_loss.device)
        ee = self.all_gather(ee).mean().item()

        sr = torch.tensor(anaInfo['sr'], device=batch_loss.device)
        sr = self.all_gather(sr).mean().item()

        ir = torch.tensor(anaInfo['ir'], device=batch_loss.device)
        ir = self.all_gather(ir).mean().item()
        sr_step = torch.tensor(anaInfo['sr_step'], device=batch_loss.device)
        sr_step = self.all_gather(sr_step).mean().item()
        ir_step = torch.tensor(anaInfo['ir_step'], device=batch_loss.device)
        ir_step = self.all_gather(ir_step).mean().item()
        size_step = torch.tensor(anaInfo['size_exp'], device=batch_loss.device)
        size_step = self.all_gather(size_step).mean().item()
        track_loss = torch.tensor(anaInfo['track_loss'], device=batch_loss.device)
        track_loss = self.all_gather(track_loss).mean().item()

        avg_loss = self.all_gather(batch_loss).mean().item()
        avg_value = self.all_gather(batch_value).mean().item()
        self.val_batch_loss.append(avg_loss)
        self.val_batch_value.append(avg_value)
        self.val_batch_ee.append(ee)
        self.val_batch_sr.append(sr)
        self.val_batch_ir.append(ir)
        self.val_batch_sr_step.append(sr_step)
        self.val_batch_ir_step.append(ir_step)
        self.val_batch_size_step.append(size_step)
        self.val_batch_track_loss.append(track_loss)

        rb = batch_idx + 1
        if self.trainer.is_global_zero and (rb % showbatch == 0 or rb == tb):
            rtime = time.time() - self.start_time  # 到目前为止的耗时S
            rtime = rtime / rb  # 平均每Batch耗时S
            rtime = round(rtime * (tb - rb))  # 剩余用时
            avg_loss = sum(self.val_batch_loss) / len(self.val_batch_loss)
            avg_value = sum(self.val_batch_value) / len(self.val_batch_value)
            avg_ee = sum(self.val_batch_ee) / len(self.val_batch_ee)
            avg_sr = sum(self.val_batch_sr) / len(self.val_batch_sr)
            avg_size = sum(self.val_batch_size_step) / len(self.val_batch_size_step)
            # avg_rmse = sum(self.val_batch_rmse) / len(self.val_batch_rmse)
            # avg_rmse = sum(self.val_batch_rmse) / len(self.val_batch_rmse)
            # print('mae_print_check', self.val_batch_mae, len(self.val_batch_mae))
            # avg_mae = sum(self.val_batch_mae) / len(self.val_batch_mae)
            self.output_info(
                f"{CM.timestr()}ValEpoch:{epoch}/{self.args.train_epochs}:Batch:{rb}/{tb}:loss:{avg_loss:.6f}:value:{avg_value:.6f}:ee:{avg_ee:.6f}:sr:{avg_sr:.6f}:size:{avg_size:.6f}:time:{rtime}s")

    def on_validation_epoch_end(self):
        # 验证集完整的验证集验证结束后，计算验证集上的平均损失
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        epoch_loss = sum(self.val_batch_loss) / len(self.val_batch_loss)
        epoch_value = sum(self.val_batch_value) / len(self.val_batch_value)
        epoch_ee = sum(self.val_batch_ee) / len(self.val_batch_ee)
        epoch_sr = sum(self.val_batch_sr) / len(self.val_batch_sr)
        epoch_ir = sum(self.val_batch_ir) / len(self.val_batch_ir)
        epoch_sr_step = sum(self.val_batch_sr_step) / len(self.val_batch_sr_step)
        epoch_ir_step = sum(self.val_batch_ir_step) / len(self.val_batch_ir_step)
        epoch_size_step = sum(self.val_batch_size_step) / len(self.val_batch_size_step)
        epoch_track_loss = sum(self.val_batch_track_loss) / len(self.val_batch_track_loss)

        # epoch_mae = sum(self.val_batch_mae) / len(self.val_batch_mae)
        # epoch_mae = sum(self.val_batch_mae) / len(self.val_batch_mae)

        self.val_batch_loss = []
        self.val_batch_value = []
        self.val_batch_ee = []
        self.val_batch_sr = []
        self.val_batch_ir = []
        self.val_batch_sr_step = []
        self.val_batch_ir_step = []
        self.val_batch_size_step = []
        self.val_batch_track_loss = []
        
        addinfo = ''
        sc1 = True
        if self.args.double_val:
            if self.trainer.current_epoch % 2 == 0:
                addinfo = ":@V1"
            else:
                addinfo = ":@V2"
                sc1 = False
        if not self.trainer.is_global_zero:
            return
            # 进行早停检查（保存模型）

        if sc1:
            get_better = self.stop_checker([epoch_loss, epoch_value,  epoch_ir, epoch_ir_step, epoch_ee], epoch, self.model, rank=0)
        else:
            get_better = self.stop_checker_2([epoch_loss, epoch_value, epoch_ir, epoch_ir_step, epoch_ee], epoch, self.model, rank=0)

        if get_better:
            self.epoch_no_better_count = 0
            addinfo += "***"
        else:
            self.epoch_no_better_count += 1
            if self.epoch_no_better_count >= self.args.lr_patience:
                # 学习率衰减
                self.adjust_learning_rate_dynamic()
                self.epoch_no_better_count = 0
        if sc1:
            if self.stop_checker.early_stop:
                print(f"{CM.timestr()}****Early stopping****")
                self.trainer.should_stop = True
        else:
            if self.stop_checker_2.early_stop:
                print(f"{CM.timestr()}****Early stopping****")
                self.trainer.should_stop = True
        self.output_info(
            f"{CM.timestr()}MPE_Val:{epoch}/{self.args.train_epochs}:loss:{epoch_loss:.6f}:value:{epoch_value:.6f}:Turnover:{epoch_sr:.6f}:IR_real:{epoch_ir:.6f}:SR_step:{epoch_sr_step:.6f}:IR_step:{epoch_ir_step:.6f}:Size_step:{epoch_size_step:.6f}:Track_loss:{epoch_track_loss:.6f}:ee:{epoch_ee:.6f}{addinfo}")

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

    def test_step(self, batch, batch_idx):
        # 一次test操作 one batch
        if batch_idx == 0:
            self._check_align = True
            clear_temp_files()
            self.start_time = time.time()
        tb = self.trainer.num_test_batches[0]
        epoch = self.trainer.current_epoch + 1 + self.args.epno
        # showbatch = 1  # CM.make_fit_showbatch(None, tb)
        batch_loss, batch_value, anaInfo, IR_loss_val, deviation_val, active_mask = self.forward_an_episode(batch, training=False)

        trade_process = anaInfo['trade_process']
        last_value = anaInfo['last_value']
        fv_list = None
        if anaInfo.__contains__("fv_list"):
            fv_list = anaInfo['fv_list']
        penality_count = anaInfo['penality_count']
        fooler_value = anaInfo['fooler_value']
        weight_entropy = anaInfo['weight_entropy']
        weight_cash = anaInfo['weight_cash']
        BigStockCount = anaInfo['BigStockCount']
        wc = anaInfo['weight_change_total']
        t1ds = anaInfo['t1ds_count']

        trade_process_c1 = anaInfo['trade_process_c1']
        last_value_c1 = anaInfo['last_value_c1']
        trade_process_c2 = anaInfo['trade_process_c2']
        last_value_c2 = anaInfo['last_value_c2']

        # penality_count是一个标量, 这里需要将其转换为tensor,才能进行all_gather
        penality_count = torch.tensor(penality_count, device=batch_loss.device)
        weight_entropy = torch.tensor(weight_entropy, device=batch_loss.device)
        weight_cash = torch.tensor(weight_cash, device=batch_loss.device)
        BigStockCount0 = torch.tensor(BigStockCount[0], device=batch_loss.device)
        BigStockCount2 = torch.tensor(BigStockCount[1], device=batch_loss.device)
        BigStockCount5 = torch.tensor(BigStockCount[2], device=batch_loss.device)
        BigStockCount10 = torch.tensor(BigStockCount[3], device=batch_loss.device)
        BigStockCount20 = torch.tensor(BigStockCount[4], device=batch_loss.device)
        TopStock1 = torch.tensor(BigStockCount[5], device=batch_loss.device)
        TopStock2 = torch.tensor(BigStockCount[6], device=batch_loss.device)
        wc = torch.tensor(wc, device=batch_loss.device)
        t1ds = torch.tensor(t1ds, device=batch_loss.device)

        # 将交易过程和最后时刻的投资组合价值 记录到文件中
        self.save_test_result(trade_process, last_value, 'C0', fvlist=fv_list)
        self.save_test_result(trade_process_c1, last_value_c1, 'C1', fvlist=fv_list)
        self.save_test_result(trade_process_c2, last_value_c2, 'C2', fvlist=fv_list)

        avg_loss = self.all_gather(batch_loss).mean().item()
        avg_value = self.all_gather(batch_value).mean().item()
        fooler_value = self.all_gather(fooler_value).mean().item()
        weight_entropy = self.all_gather(weight_entropy).mean().item()
        weight_cash = self.all_gather(weight_cash).mean().item()
        BigStockCount0 = self.all_gather(BigStockCount0).mean().item()
        BigStockCount2 = self.all_gather(BigStockCount2).mean().item()
        BigStockCount5 = self.all_gather(BigStockCount5).mean().item()
        BigStockCount10 = self.all_gather(BigStockCount10).mean().item()
        BigStockCount20 = self.all_gather(BigStockCount20).mean().item()
        TopStock1 = self.all_gather(TopStock1).mean().item()
        TopStock2 = self.all_gather(TopStock2).mean().item()
        wc = self.all_gather(wc).mean().item()
        self.test_fooler_value.append(fooler_value)
        self.test_weight_entropy.append(weight_entropy)
        self.test_weight_cash.append(weight_cash)
        self.test_BigStockCount0.append(BigStockCount0)
        self.test_BigStockCount2.append(BigStockCount2)
        self.test_BigStockCount5.append(BigStockCount5)
        self.test_BigStockCount10.append(BigStockCount10)
        self.test_BigStockCount20.append(BigStockCount20)
        self.test_TopStock1.append(TopStock1)
        self.test_TopStock2.append(TopStock2)
        self.test_weight_change_total.append(wc)
        self.penality_sum += self.all_gather(penality_count).sum().item()
        self.t1ds_sum += self.all_gather(t1ds).sum().item()
        self.test_batch_loss.append(avg_loss)
        self.test_batch_value.append(avg_value)
        rb = batch_idx + 1
        last_batch = rb >= tb
        CM.mark_tmp('clear', self.args.mark)
        if self.trainer.is_global_zero:
            rtime = time.time() - self.start_time  # 到目前为止的耗时S
            rtime = rtime / rb  # 平均每Batch耗时S
            rtime = round(rtime * (tb - rb))  # 剩余用时
            avg_loss = sum(self.test_batch_loss) / len(self.test_batch_loss)
            avg_value = sum(self.test_batch_value) / len(self.test_batch_value)
            self.output_info(
                f"{CM.timestr()}TestEpoch:{epoch}/{self.args.train_epochs}:Batch:{rb}/{tb}:loss:{avg_loss:.6f}:value:{avg_value:.6f}:time:{rtime}s")

            # 每次经过showbatch个batch，就输出一次测试集的结果
            # 计算测试集上的指标
            fooler_value = sum(self.test_fooler_value) / len(self.test_fooler_value)
            weight_entropy = sum(self.test_weight_entropy) / len(self.test_weight_entropy)
            weight_cash = sum(self.test_weight_cash) / len(self.test_weight_cash)
            BigStockCount0 = sum(self.test_BigStockCount0) / len(self.test_BigStockCount0)
            BigStockCount2 = sum(self.test_BigStockCount2) / len(self.test_BigStockCount2)
            BigStockCount5 = sum(self.test_BigStockCount5) / len(self.test_BigStockCount5)
            BigStockCount10 = sum(self.test_BigStockCount10) / len(self.test_BigStockCount10)
            BigStockCount20 = sum(self.test_BigStockCount20) / len(self.test_BigStockCount20)
            TopStock1 = sum(self.test_TopStock1) / len(self.test_TopStock1)
            TopStock2 = sum(self.test_TopStock2) / len(self.test_TopStock2)
            BigStockCount = [BigStockCount0, BigStockCount2, BigStockCount5, BigStockCount10, BigStockCount20,
                             TopStock1, TopStock2]
            wc = sum(self.test_weight_change_total) / len(self.test_weight_change_total)
            self.cal_and_show_metrics(last_batch, ps=self.penality_sum, fooler_value=fooler_value,
                                      weight_entropy=weight_entropy, weight_cash=weight_cash,
                                      BigStockCount=BigStockCount, weight_change=wc, t1ds=self.t1ds_sum)
            # if 1 < self.args.checktimes <= total_count:
            #     self.trainer.should_stop = True
            #     print(f"{CM.timestr()}***要求测试次数{self.args.checktimes},实际测试次数{total_count},测试结束***")

    def cal_and_show_metrics(self, last_batch, ps=0, fooler_value=0., weight_entropy=0., weight_cash=0.,
                             BigStockCount=None, weight_change=0., t1ds=0):
        # 遍历当前目录下的所有文件，找到以TMP_test_process_开头的文件，读取其中的内容，汇总到一个列表中，然后输出到控制台
        if BigStockCount is None:
            BigStockCount = [0, 0, 0, 0, 0, 0, 0]

        trade_process_list, fv_list, last_value_list = read_test_result('C0')
        trade_process_list_c1, fv_list_c1, last_value_list_c1 = read_test_result('C1')
        trade_process_list_c2, fv_list_c2, last_value_list_c2 = read_test_result('C2')

        # 输出表头
        print(
            f"***{CM.timestr()}模型'{os.path.basename(self.args.loadcheck)}'-数据{self.args.DS}-交易期{self.args.episode_len}-调仓周期{self.args.tr_days}-目标{self.args.price_target}-测试期[{self.args.test_date_area[0]}-{self.args.test_date_area[1]}]-首个交易日{self.args.firstTD}***")
        title = f"COST,M-TESTCOUNT,MV,FV,MFR,CASH,TR,WE,B0/2/5/10/20/T1/T2,MAX,MIN,LOSS,LR,ARR,VOL,DD,MDD,SR,CR,SOR,MER,DIR,AIR"
        info1 = make_info_str("C-0", self.args.M, trade_process_list, fv_list, last_value_list, fooler_value,
                              weight_cash, weight_change, weight_entropy, BigStockCount)
        info2 = make_info_str(f"C-{self.args.tc1}", self.args.M, trade_process_list_c1, fv_list_c1, last_value_list_c1,
                              fooler_value, weight_cash, weight_change, weight_entropy, BigStockCount)
        info3 = make_info_str(f"C-{self.args.tc2}", self.args.M, trade_process_list_c2, fv_list_c2, last_value_list_c2,
                              fooler_value, weight_cash, weight_change, weight_entropy, BigStockCount)
        print(title)
        print(info1)
        print(info2)
        print(info3)
        cplist = CM.get_percent_list(self.cash_dist)
        cps = [str(i) for i in cplist]
        # print(f"现金占比情况c:{self.cash_dist}")
        print(f"现金占比情况,{','.join(cps)}")
        print(f"价值惩罚总次数:{ps},T1日开盘跌停且减仓次数:{t1ds}")
        if last_batch:
            clear_temp_files()
        return 1

    def on_test_epoch_end(self):
        self.test_batch_loss = []
        self.test_batch_value = []
        self.penality_sum = 0
        self.t1ds_sum = 0
        self.test_fooler_value = []
        self.test_weight_entropy = []
        self.test_weight_cash = []
        self.test_BigStockCount0 = []
        self.test_BigStockCount2 = []
        self.test_BigStockCount5 = []
        self.test_BigStockCount10 = []
        self.test_BigStockCount20 = []
        self.test_TopStock1 = []
        self.test_TopStock2 = []
        self.test_weight_change_total = []


    def configure_optimizers(self):
        # 仅保留主模型的优化器
        optimizer_model = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        return optimizer_model

    
    def adjust_learning_rate_dynamic(self):
        opts = self.optimizers()
        opt_model = opts[0] if isinstance(opts, (list, tuple)) else opts
        current_lr = opt_model.param_groups[0]['lr']
        new_lr = current_lr * self.args.lr_decay
        for param_group in opt_model.param_groups:
            param_group['lr'] = new_lr
        self.output_info(f'{CM.timestr()}Updating learning rate to {new_lr:.7f}')

    def output_info(self, info):
        # 输出信息到控制台,但只有主进程才输出
        if self.trainer.is_global_zero:
            print(info)

    '''
    def train_dataloader(self):

        # 获取两个时间区间的dataset
        dataset1 = self.data_reader.create_data_loader(
            self.args.train_date_area[0],
            self.args.train_date_area[1],
            refresh=True,
            stride_shift=True
        ).dataset  # 假设返回的是包含dataset属性的对象

        dataset2 = self.data_reader.create_data_loader(
            20230401,
            20241231,
            refresh=True,
            stride_shift=True
        ).dataset

        # 合并数据集
        combined_dataset = ConcatDataset([dataset1, dataset2])

        # 创建新的dataloader并启用shuffle
        d_loader = DataLoader(
            combined_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,  # 重要：启用打乱
            drop_last=False
        )

        if self.trainer.is_global_zero:
            totalBatchs = len(d_loader)
            print(
                f"{CM.timestr()}***训练集总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")

        return d_loader

    '''

    def train_dataloader(self):
        d_loader = self.data_reader.create_data_loader(self.args.train_date_area[0], self.args.train_date_area[1],
                                                       refresh=True, stride_shift=True)
        if self.trainer.is_global_zero:
            totalBatchs = len(d_loader)
            print(
                f"{CM.timestr()}***训练集总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")
        return d_loader

    def val_dataloader(self):
        addinfo = ''
        if self.args.double_val:
            # 双验证
            if self.trainer.current_epoch % 2 == 0:
                addinfo = '@V1'
                da, db = self.args.valid_date_area[0], self.args.valid_date_area[1]
            else:
                addinfo = '@V2'
                da, db = self.args.test_date_area[0], self.args.test_date_area[1]
            d_loader = self.data_reader.create_data_loader(da, db, refresh=True, stride_shift=True)
            if self.trainer.is_global_zero:
                totalBatchs = len(d_loader)
                print(
                    f"{CM.timestr()}***验证集{addinfo}总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")
            return d_loader
        else:
            # 单独验证
            da, db = self.args.valid_date_area[0], self.args.valid_date_area[1]
            d_loader = self.data_reader_val.create_data_loader(da, db, refresh=True, stride_shift=True)
            if self.trainer.is_global_zero:
                totalBatchs = len(d_loader)
                print(
                    f"{CM.timestr()}***验证集{addinfo}总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{self.args.batch_size * totalBatchs}")
            return d_loader

    def test_dataloader(self):
        d_loader = self.data_reader.create_data_loader(self.args.test_date_area[0], self.args.test_date_area[1],
                                                       refresh=True, stride_shift=True)
        if self.trainer.is_global_zero:
            totalBatchs = len(d_loader)
            tsc = self.args.batch_size * totalBatchs
            if self.args.RUN_MODE == 'check_m':
                self.record_count = int(tsc * 0.8)
            print(
                f"{CM.timestr()}***测试集{self.args.test_date_area},集内总交易天数:{CM.count_trade_days(self.args.test_date_area[0], self.args.test_date_area[1])},总Batch数:{totalBatchs},batchSize:{self.args.batch_size},总样本数:{tsc}")
        if self.args.useGPU and self.args.multi_device:
            sampler = DistributedSampler(d_loader.dataset)
            d_loader = DataLoader(d_loader.dataset, batch_size=self.args.batch_size, sampler=sampler,
                                  num_workers=self.args.num_workers)
        return d_loader


# pm_v2是训练头5分钟成交均价成交的模型, 采用的数据集是ts_day_55f_avg_price, 包含有一个头5分钟成交均价数据列
# post ./yy_train_pm_v2.py G1 G3 '--mark pm_0320m200 --M 200 --use_prev_w 1 --batch_size 5 --epno 53'
if __name__ == "__main__":
    args = PM.get_args(run_mode='train', model='PM_SMV2')
    if args.RUN_MODE == 'train':
        args.G = args.K = 0
        print(
            f"{CM.timestr()}***{args.DS}***训练：{args.train_date_area}***验证：{args.valid_date_area}***标的:{args.input_channels[args.price_target_index]}")
        if args.double_val:
            print(f"{CM.timestr()}***双验证模式--验证1：{args.valid_date_area}--验证2：{args.test_date_area}")
        if args.epno > 0:
            print(f"{CM.timestr()}***{args.DS}***继续训练，从{args.epno + 1}轮开始***")
    elif args.RUN_MODE == 'check':
        args.num_workers = 1  # 禁用多进程加速
        assert os.path.exists(args.loadcheck), "请指定模型文件路径"
        sample_length = PM.cal_seq_len(args)
        print(
            f"测试数据集时段:***{args.DS}***测试时段:{args.test_date_area}***标的:{args.input_channels[args.price_target_index]}")
        tdays = CM.count_trade_days(args.test_date_area[0], args.test_date_area[1])
        first_trade_date = CM.get_next_trade_day(args.test_date_area[0], args.time_step)
        args.firstTD = first_trade_date
        print(f"该时段内共有{tdays}个交易日,一次推理需要{sample_length}天数据, 最早的交易日是{first_trade_date}")
        if tdays <= sample_length:
            print(f"该时段内数据不足,无法进行测试")
            exit()
    elif args.RUN_MODE == 'check_m':
        # python ./yy_train_pm_v2.py --RUN_MODE=check_m --loadcheck=./output/pm_0103_epoch16.pth --batch_size=1 --episode_len=22 --M=3000 --cfd=20210101 --cld=20240901 --checktimes=32
        args.num_workers = 1  # 禁用多进程加速
        assert os.path.exists(args.loadcheck), "请指定模型文件路径"
        sample_length = PM.cal_seq_len(args)
        print(
            f"测试数据集时段:***{args.DS}***按月测试总时段:{args.test_date_area},一次推理需要{sample_length}天数据***")
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
                         max_epochs=args.train_epochs,
                         accelerator=accelerator,
                         devices=devices,
                         enable_progress_bar=False,
                         log_every_n_steps=0,
                         strategy=strategy,  # 'ddp_find_unused_parameters_true',
                         #strategy='ddp_find_unused_parameters_true',  # 'ddp_find_unused_parameters_true',
                         precision='bf16-mixed',
                         reload_dataloaders_every_n_epochs=1)
    if args.RUN_MODE == 'train':
        trainer.fit(model=pm_model)
        print(f"{CM.timestr()}TrainingDone.")
    elif args.RUN_MODE == 'check':
        trainer.test(model=pm_model)
        print(f"{CM.timestr()}TestingDone.")
    elif args.RUN_MODE == 'check_m':
        # 指定开始日期和结束日期(yymm01),按月进行测试,并输出最后结果
        pm_model.check_month = [args.cfd, args.cld]
        logfn = os.path.basename(args.loadcheck).split('.')[0]
        logfn = f"Month_{logfn}_M{args.M}_{pm_model.check_month[0]}_{pm_model.check_month[1]}.csv"
        if os.path.exists(logfn):
            os.remove(logfn)
        args.m_log_file = logfn

        date_area_list = CM.make_date_area_list(pm_model.check_month, args.time_step, args.episode_len + args.buffdays)
        for x in date_area_list:
            pm_model.args.first_trade_date = int(x[0])
            pm_model.args.test_date_area = [int(x[1]), int(x[2])]
            trainer.test(model=pm_model)
        print(f"{CM.timestr()}TestingMonthlyDone.")
