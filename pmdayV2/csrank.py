import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional


class CSRankPreprocessor(nn.Module):
    """
    股票时序特征截面秩正态化预处理器 (PyTorch nn.Module 标准子模块)。

    输入:
        input_seq: [B, M, L, C] (例如 B: Batch Size, M: 股票截面数量, L: 时序长度, C: 原始特征维度如 311)
        valid_mask: [B, M, L] 或 [B, M, 1, 1] (可选，1/True 为正常交易，0/False 为停牌/无效标的)

    处理逻辑:
        1. price         -> 相对于最后一天 base price 做比例化: x / base - 1
        2. log           -> log1p -> 截面秩归一化 (CS-Rank, 默认 gaussian 模式映射为 N(0, 1))
        3. normal numeric-> 截面秩归一化 (CS-Rank, 沿股票截面维 M 消除极端离群点与宏观牛熊均值漂移)
        4. categorical   -> min-max scaling 映射到 [0, 1]
        5. keep_original -> 保持原值
        6. delete        -> 剔除无效通道 (剔除 9 个通道，输出 C_new = 302)
    """

    def __init__(self, rank_mode: str = "gaussian", eps: float = 1e-6):
        super().__init__()
        assert rank_mode in ["gaussian", "uniform"], "rank_mode 必须为 'gaussian' 或 'uniform'"
        self.rank_mode = rank_mode
        self.eps = eps
        self.register_buffer("sqrt_two", torch.tensor(math.sqrt(2.0), dtype=torch.float32))

        self.feature_index: Dict[str, List[int]] = {
            # 剔除字段 (共 9 个通道)
            "delete": [
                0, 1,
                302, 303, 304, 305, 306, 307, 308,  # index weights
            ],
            # 价格字段：除以最后一天对应基准价格，再减 1
            "price": [
                10, 11, 12, 13, 14, 15,
                19, 20, 21, 22, 23,
                25,
                34, 35, 36, 37,
                66,
                294, 295, 296, 297,
            ],
            # 保持原值
            "keep_original": [
                16,
            ],
            # 分类变量 (Min-Max 缩放)
            "categorical": [
                2, 3, 4,
                5, 6, 7, 8,
                9,
                28,
                59,
                300, 301,  # suspend, st
                309,
                310,       # padding_flag
            ],
            # 对数变换特征 (log1p -> CS-Rank)
            "log": [
                17, 18,
            ],
        }

        self.categorical_minmax: Dict[int, Tuple[float, float]] = {
            2: (1.0, 10.0),       # exchange_id
            3: (1.0, 20.0),       # market_id
            4: (0.0, 200.0),      # industry_id
            5: (1990.0, 2040.0),  # year
            6: (1.0, 12.0),       # month
            7: (1.0, 31.0),       # day
            8: (0.0, 6.0),        # week
            9: (0.0, 1.0),        # is_dividend
            28: (-2.0, 7.0),      # trade_status
            59: (-1.0, 1.0),      # limit_status
            300: (0.0, 1.0),      # suspend
            301: (0.0, 1.0),      # st
            309: (1.0, 10.0),     # limit_mark
            310: (0.0, 1.0),      # padding_flag
        }

    def cs_rank_norm(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        沿股票截面维度 (dim=1, M) 执行快速 GPU 截面秩归一化:
        输入: x [B, M, L, C_sub]
        输出: out [B, M, L, C_sub]
        """
        if not torch.isfinite(x).all():
            bad_count = (~torch.isfinite(x)).sum().item()
            raise ValueError(f"cs_rank_norm 输入包含 {bad_count} 个 NaN/Inf。")

        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        B, M, L, C_sub = x_f32.shape

        # 1. 停牌与有效掩码处理
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                mask_4d = valid_mask.unsqueeze(-1)  # [B, M, L, 1]
            elif valid_mask.dim() == 2:
                mask_4d = valid_mask.view(B, M, 1, 1)  # [B, M, 1, 1]
            else:
                mask_4d = valid_mask

            x_f32 = x_f32.masked_fill(~mask_4d.bool(), -1e9)
            valid_counts = mask_4d.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            valid_counts = torch.full((B, 1, L, 1), fill_value=float(M), device=x.device, dtype=torch.float32)

        # 2. 沿股票截面维度 (dim=1) 双重 argsort 提取排名 (0 ~ M-1)
        rank = torch.argsort(torch.argsort(x_f32, dim=1), dim=1).float()

        # 3. 停牌标的排名偏移修正
        if valid_mask is not None:
            invalid_counts = M - valid_counts
            rank = (rank - invalid_counts).clamp(min=0.0)

        # 4. 映射到目标分布 (Gaussian N(0,1) 或 Uniform [-0.5, 0.5])
        if self.rank_mode == "uniform":
            out = (rank / (valid_counts - 1.0 + self.eps)) - 0.5
        elif self.rank_mode == "gaussian":
            u = (rank + 0.5) / (valid_counts + self.eps)
            u = u.clamp(min=self.eps, max=1.0 - self.eps)
            out = self.sqrt_two * torch.erfinv(2.0 * u - 1.0)
        else:
            raise ValueError(f"不支持的 rank_mode: {self.rank_mode}")

        # 5. 无效标的归零并恢复类型
        if valid_mask is not None:
            out = out.masked_fill(~mask_4d.bool(), 0.0)

        return out.to(orig_dtype)

    def _validate_config(self, num_features: int):
        groups = {name: set(indices) for name, indices in self.feature_index.items()}
        all_indices = set().union(*groups.values()) if groups else set()

        invalid = sorted(idx for idx in all_indices if idx < 0 or idx >= num_features)
        if invalid:
            raise IndexError(f"feature_index 包含越界索引 {invalid}，当前 C={num_features}。")

        names = list(groups.keys())
        overlaps = []
        for i, name_i in enumerate(names):
            for name_j in names[i + 1:]:
                overlap = groups[name_i] & groups[name_j]
                if overlap:
                    overlaps.append(f"{name_i} vs {name_j}: {sorted(overlap)}")

        if overlaps:
            raise ValueError("同一特征不能同时属于多个处理组：\n" + "\n".join(overlaps))

        for idx in self.feature_index.get("categorical", []):
            if idx not in self.categorical_minmax:
                raise KeyError(f"categorical_minmax 缺少 channel={idx} 的配置。")

    def forward(
        self,
        input_seq: torch.Tensor,
        base_price_index: int = 22,
        valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        前向处理入口:
        输入: 
            input_seq: [B, M, L, C] (例如 C=311)
            valid_mask: [B, M, L] 或 [B, M, 1, 1] (可选)
        输出: 
            clean_seq: [B, M, L, C_new] (剔除 delete 中的 9 个通道后 C_new = 302)
        """
        if not isinstance(input_seq, torch.Tensor):
            raise TypeError(f"input_seq 必须是 torch.Tensor，当前为 {type(input_seq)}")

        if input_seq.ndim != 4:
            raise ValueError(f"input_seq 应为 [B, M, L, C] 四维张量，当前 shape={tuple(input_seq.shape)}")

        if input_seq.shape[2] == 0:
            raise ValueError("时间维 L 不能为 0。")

        x = input_seq.clone().float()
        num_features = x.shape[-1]

        if not 0 <= base_price_index < num_features:
            raise IndexError(f"base_price_index={base_price_index} 越界 (0 ~ {num_features - 1})。")

        self._validate_config(num_features)

        delete_index = self.feature_index.get("delete", [])
        price_index = self.feature_index.get("price", [])
        keep_original_index = self.feature_index.get("keep_original", [])
        categorical_index = self.feature_index.get("categorical", [])
        log_index = self.feature_index.get("log", [])

        delete_set = set(delete_index)
        excluded_set = delete_set | set(price_index) | set(keep_original_index) | set(categorical_index) | set(log_index)
        normalization_index = [idx for idx in range(num_features) if idx not in excluded_set]

        # 1. 价格归一化 (相对于最后一天 base price 做比例化)
        if price_index:
            base_price = x[:, :, -1:, base_price_index:base_price_index + 1]
            if not torch.isfinite(base_price).all():
                raise ValueError(f"base_price channel={base_price_index} 包含 NaN/Inf。")

            safe_base_price = torch.where(
                base_price.abs() < eps,
                torch.where(base_price >= 0, torch.full_like(base_price, eps), torch.full_like(base_price, -eps)),
                base_price,
            )
            x_price = x[..., price_index]
            if not torch.isfinite(x_price).all():
                raise ValueError("price 特征包含 NaN/Inf。")
            x[..., price_index] = x_price / safe_base_price - 1.0

        # 2. 对数转换特征 + 截面秩归一化 (log1p -> CS-Rank)
        if log_index:
            x_log = x[..., log_index]
            if not torch.isfinite(x_log).all():
                raise ValueError("log 特征包含 NaN/Inf。")
            if (x_log <= -1.0).any():
                raise ValueError("log1p 特征要求 x > -1。")
            x[..., log_index] = self.cs_rank_norm(torch.log1p(x_log), valid_mask=valid_mask)

        # 3. 普通连续数值特征截面秩归一化 (CS-Rank)
        if normalization_index:
            x[..., normalization_index] = self.cs_rank_norm(x[..., normalization_index], valid_mask=valid_mask)

        # 4. 分类变量 Min-Max 映射
        if categorical_index:
            for idx in categorical_index:
                min_v, max_v = self.categorical_minmax[idx]
                x_cat = x[..., idx]
                if not torch.isfinite(x_cat).all():
                    raise ValueError(f"categorical channel={idx} 包含 NaN/Inf。")
                if max_v > min_v:
                    x[..., idx] = (x_cat - min_v) / (max_v - min_v)
                else:
                    x[..., idx] = 0.0

        # 5. 剔除无效通道
        if delete_index:
            keep_index = [idx for idx in range(num_features) if idx not in delete_set]
            x = x[..., keep_index]

        return x

    def pre_processor_csrank(
        self,
        input_seq: torch.Tensor,
        base_price_index: int = 22,
        valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """向后兼容旧方法名调用"""
        return self.forward(input_seq, base_price_index=base_price_index, valid_mask=valid_mask, eps=eps)


# 别名兼容
csrank = CSRankPreprocessor
CrossSectionalRankPreprocessor = CSRankPreprocessor


class CSRank_channel(nn.Module):
    """
    针对 [B, M, L, C] 时序张量，在股票截面维 (M, dim=1) 上
    对指定的连续特征通道执行截面秩归一化 (CS-Rank, 映射至高斯分布或均匀分布)。
    """
    def __init__(
        self,
        mode: str = "gaussian",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert mode in ["gaussian", "uniform"], "mode 必须为 'gaussian' 或 'uniform'"
        self.mode = mode
        self.eps = eps
        self.register_buffer("sqrt_two", torch.tensor(math.sqrt(2.0), dtype=torch.float32))

    def forward(
        self,
        global_seq: torch.Tensor,
        target_channel_indices: Union[List[int], torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            global_seq: [B, M, L, C] 四维张量
            target_channel_indices: 需要进行截面秩归一化的通道索引列表
            valid_mask: [B, M, L] 或 [B, M, 1, 1] 停牌/有效股票掩码
        返回:
            norm_seq: 处理后的 [B, M, L, C] 张量
        """
        if not isinstance(global_seq, torch.Tensor):
            raise TypeError(f"global_seq 必须是 torch.Tensor，实际类型为 {type(global_seq)}")
        if global_seq.ndim != 4:
            raise ValueError(f"期望 global_seq 为 4 维张量 [B, M, L, C]，实际形状为 {tuple(global_seq.shape)}")

        if isinstance(target_channel_indices, torch.Tensor):
            target_indices = target_channel_indices.tolist()
        else:
            target_indices = list(target_channel_indices)

        if not target_indices:
            return global_seq.clone()

        max_idx = global_seq.shape[-1]
        for idx in target_indices:
            if idx < 0 or idx >= max_idx:
                raise IndexError(f"通道索引 {idx} 越界 (有效范围: 0 ~ {max_idx - 1})")

        out_seq = global_seq.clone().float()
        sub_seq = out_seq[..., target_indices]

        # NaN/Inf 保护填充
        if not torch.isfinite(sub_seq).all():
            sub_seq = torch.nan_to_num(sub_seq, nan=0.0, posinf=1e4, neginf=-1e4)

        orig_dtype = global_seq.dtype
        x_f32 = sub_seq.to(torch.float32)
        B, M, L, C_sub = x_f32.shape

        if valid_mask is not None:
            if valid_mask.dim() == 3:
                mask_4d = valid_mask.unsqueeze(-1)
            elif valid_mask.dim() == 2:
                mask_4d = valid_mask.view(B, M, 1, 1)
            else:
                mask_4d = valid_mask

            x_f32 = x_f32.masked_fill(~mask_4d.bool(), -1e9)
            valid_counts = mask_4d.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            valid_counts = torch.full((B, 1, L, 1), fill_value=float(M), device=global_seq.device, dtype=torch.float32)

        rank = torch.argsort(torch.argsort(x_f32, dim=1), dim=1).float()

        if valid_mask is not None:
            invalid_counts = M - valid_counts
            rank = (rank - invalid_counts).clamp(min=0.0)

        if self.mode == "uniform":
            norm_sub = (rank / (valid_counts - 1.0 + self.eps)) - 0.5
        elif self.mode == "gaussian":
            u = (rank + 0.5) / (valid_counts + self.eps)
            u = u.clamp(min=self.eps, max=1.0 - self.eps)
            norm_sub = self.sqrt_two * torch.erfinv(2.0 * u - 1.0)
        else:
            raise ValueError(f"不支持的 mode: {self.mode}")

        if valid_mask is not None:
            norm_sub = norm_sub.masked_fill(~mask_4d.bool(), 0.0)

        out_seq[..., target_indices] = norm_sub.to(orig_dtype)
        return out_seq