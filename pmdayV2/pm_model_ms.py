import math
import sys
from argparse import Namespace
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
#import cvxpy as cp
from einops import rearrange
import torch.nn.functional as F
import random

import copy
import json
import os
from typing import Iterable, Optional, Sequence
from csrank import CSRankPreprocessor


sys.path.append('../')
from common_module.data_normalizer import DataNormalizer as DataNormalizer

######!!!注意: 在将模型scripted化时,要将checkpoint相差代码注释掉,否则会出错######
######!!!注意: 训练时,将注释打开######

# def apply_with_chunking(func, tensor, chunk_size=2000):
#     """分块应用函数，使用预分配内存"""
#     if tensor.shape[0] <= chunk_size:
#         return func(tensor)
    
#     chunks = tensor.split(chunk_size, dim=0)
#     output = torch.empty_like(tensor)  # 假设输出形状与输入相同
    
#     start = 0
#     for chunk in chunks:
#         chunk_out = func(chunk)
#         end = start + chunk_out.shape[0]
#         output[start:end] = chunk_out
#         start = end
    
#     return output


class PMQuantTaskHead(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 8,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        self.index_extractor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model)
            )

        # 1. 角色 Token
        self.role_index = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.role_stock = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 2. 引入指定的 AdaLN 条件调制层
        self.ada_ln = AdaLNConstraintProjectionForPatchTS(d_model=d_model)

        # 3. 宏观环境乘法门控
        self.macro_gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # 4. 截面博弈注意力交互主干
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.interaction_layers = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 5. MoE 路由与分配专家
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts),
        )
        expert_in_dim = d_model + 1
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_in_dim, expert_in_dim * 4),  # 升维
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_in_dim * 4, 1),             # 收缩成 Logit
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        si_rep: torch.Tensor,
        sm_rep: torch.Tensor,
        pre_w: torch.Tensor,
        target_exp: torch.Tensor,
        condition_emb: torch.Tensor,
        target_weights: torch.Tensor,
        stock_valid_mask: torch.Tensor = None,
    ):
        B, M, D = si_rep.shape

        # ==========================================================
        # Step 0: 维度对齐与安全检查
        # ==========================================================
        if sm_rep.dim() == 2:
            sm_rep = sm_rep.unsqueeze(1)  # 保证为 [B, 1, D]

        if condition_emb.dim() == 3:
            condition_emb = condition_emb.squeeze(1)  # 保证为 [B, D]

        if target_weights.dim() == 2:
            weight_expanded = target_weights.unsqueeze(-1)  # [B, M, 1]
        else:
            weight_expanded = target_weights

        # ==========================================================
        # Step 1: Benchmark Token 合成 与 AdaLN 股票条件调制
        # ==========================================================
        # 指数表征加权合成
        projected_si = self.index_extractor(si_rep)
        index_token = torch.sum(projected_si * weight_expanded, dim=1, keepdim=True)  # [B, 1, D]

        # 调用指定的 AdaLNConstraintProjectionForPatchTS 调制股票表征
        stock_tokens = self.ada_ln(si_rep, condition_emb)  # [B, M, D]

        # 注入身份/角色编码
        index_token = index_token + self.role_index
        stock_tokens = stock_tokens + self.role_stock

        # ==========================================================
        # Step 2: 序列拼接与宏观门控注入 (Concat & Macro FiLM)
        # ==========================================================
        # 序列排布: [Index, Stock_1, ..., Stock_M] -> [B, 1+M, D]
        sequence = torch.cat([index_token, stock_tokens], dim=1)

        # 根据宏观信息生成 0~1 的乘法门控: [B, 1, D]
        macro_gate = self.macro_gate_net(sm_rep)
        sequence = sequence * macro_gate

        # ==========================================================
        # Step 3: 全截面博弈交互 (Self-Attention)
        # ==========================================================
        sequence = self.interaction_layers(sequence)

        # 取出进化后的 Index Token 作为全局决策状态
        final_index_token = sequence[:, 0:1, :]  # [B, 1, D]

        # ==========================================================
        # Step 4: MoE 软路由权重生成
        # ==========================================================
        router_input = torch.cat([final_index_token, sm_rep], dim=-1)  # [B, 1, 2D]
        router_logits = self.router(router_input).squeeze(1)          # [B, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)             # [B, num_experts]

        # ==========================================================
        # Step 5: 专家网络的输入准备 (融入前日仓位 pre_w)
        # ==========================================================
        pre_w_expanded = pre_w.unsqueeze(-1)                          # [B, 1+M, 1]
        expert_input = torch.cat([sequence, pre_w_expanded], dim=-1)  # [B, 1+M, D+1]

        # ==========================================================
        # Step 6: MoE 专家计算与聚合
        # ==========================================================
        combined_expert_output = torch.zeros(
            (B, 1 + M, 1),
            dtype=expert_input.dtype,
            device=expert_input.device,
        )

        for i, expert in enumerate(self.experts):
            expert_out = expert(expert_input)                          # [B, 1+M, 1]
            weight_i = router_weights[:, i].view(B, 1, 1)             # [B, 1, 1]
            combined_expert_output = combined_expert_output + (expert_out * weight_i)

        # ==========================================================
        # Step 7: 输出最终 Logits
        # ==========================================================
        final_logits = combined_expert_output.squeeze(-1)              # [B, 1+M]

     
        return final_logits
    
class PortfolioCompetitionLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        # 截面注意力机制：让 M 只股票互相比较
        # batch_first=True 保证输入形状为 (B, M, D)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 经典的 Transformer FeedForward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, M, D) 已经被 AdaLN 调制过的全截面股票表征
        """
        # 1. 股票间互相打量 (Cross-Sectional Attention)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # 2. 特征非线性升华
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class CrossAttentionConstraintConditionerV2(nn.Module):
    def __init__(self, f_dim=10, d_model=128, nhead=4):
        super().__init__()
        self.f_dim = f_dim
        self.d_model = d_model

        # 1. CLS Token：总代理
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. 因子身份护照 (强制正交初始化)
        self.factor_id_embeds = nn.Parameter(torch.empty(1, f_dim, d_model))
        nn.init.orthogonal_(self.factor_id_embeds)

        # ==========================================
        # 👑 核心魔法：神圣的 Void Token (代表无约束状态)
        # ==========================================
        self.void_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 3. 目标值映射网络
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # 4. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, target_exp, active_mask):
        B = target_exp.shape[0]
        device = target_exp.device

        # 1. 构建 10 个真实因子的 Token
        val_embs = self.value_encoder(target_exp.unsqueeze(-1))
        factor_tokens = val_embs + self.factor_id_embeds.expand(B, -1, -1)

        # ==========================================
        # 2. 将 Void Token 拼接到序列末尾！
        # ==========================================
        # void_tokens 扩展为 (B, 1, d_model)
        expanded_void = self.void_token.expand(B, -1, -1)

        # key_values 序列变成了 11 个 Token: (B, 11, d_model)
        all_tokens = torch.cat([factor_tokens, expanded_void], dim=1)

        # 3. 准备 CLS Token
        query = self.cls_token.expand(B, -1, -1)

        # ==========================================
        # 4. 完美无瑕的 Mask 逻辑
        # ==========================================
        # 对 10 个真实因子： active_mask 为 0 时，屏蔽 (True)
        base_padding_mask = (active_mask == 0)  # (B, 10)

        # 对第 11 个 Void Token：永远不屏蔽！(False)
        void_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B, 1)

        # 拼接出最终的 11 维掩码 (B, 11)
        # 如果前 10 个全是 True，没关系，第 11 个是 False，绝不会报 NaN！
        final_padding_mask = torch.cat([base_padding_mask, void_mask], dim=1)

        # 5. Cross-Attention 吸收信息
        attn_output, _ = self.cross_attn(
            query=query,
            key=all_tokens,
            value=all_tokens,
            key_padding_mask=final_padding_mask
        )

        # 6. 提取与输出
        condition_emb = attn_output.squeeze(1)
        final_condition = self.output_mlp(condition_emb)

        return final_condition


class CrossAttentionConstraintConditioner(nn.Module):
    def __init__(self, f_dim=10, d_model=128, nhead=4):
        super().__init__()
        self.f_dim = f_dim
        self.d_model = d_model

        # 1. CLS Token：作为总代理，负责吸收所有生效的风控指令
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 2. 因子身份护照 (Factor ID Embeddings)
        # 代表 10 个因子的固有身份，就像位置编码一样绝对
        self.factor_id_embeds = nn.Parameter(torch.randn(1, f_dim, d_model))

        # 3. 目标值映射网络 (将标量数值变成高维特征)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # 4. 核心：交叉注意力机制
        # batch_first=True 极其重要
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # 5. 输出融合与归一化
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, target_exp, active_mask):
        """
        target_exp: (B, f_dim) 10 个因子的约束值
        active_mask: (B, f_dim) 1 表示有约束，0 表示无约束
        """
        B = target_exp.shape[0]

        # ==========================================
        # 第一步：构建 10 个因子的完整 Token (Identity + Value)
        # ==========================================
        # 1. 获取数值的嵌入特征 (B, f_dim, 1) -> (B, f_dim, d_model)
        val_embs = self.value_encoder(target_exp.unsqueeze(-1))

        # 2. 将数值特征与固有的因子 ID 身份相加
        # (B, f_dim, d_model) + (1, f_dim, d_model)
        factor_tokens = val_embs + self.factor_id_embeds.expand(B, -1, -1)

        # ==========================================
        # 第二步：准备 CLS Token 和 绝对掩码 (Key Padding Mask)
        # ==========================================
        # (1, 1, d_model) -> (B, 1, d_model)
        query = self.cls_token.expand(B, -1, -1)

        # 【核心黑科技】：Key Padding Mask
        # PyTorch 的 MHA 中，key_padding_mask 要求 True 表示被忽略 (屏蔽)
        # active_mask 是 1 表示激活，0 表示不激活。所以取反！
        # key_padding_mask 形状: (B, f_dim)
        key_padding_mask = (active_mask == 0)

        # 防护补丁：如果极其罕见地出现一个 Batch 全是 0 (没有约束)
        # 强行给全 0 的 batch 放开第 0 个因子的 mask，防止 Attention 出现 NaN
        # (这不会影响结果，因为你 Loss 里如果是 0约束，Loss直接乘0了)
        all_zero_mask = key_padding_mask.all(dim=1)
        if all_zero_mask.any():
            key_padding_mask[all_zero_mask, 0] = False

            # ==========================================
        # 第三步：Cross-Attention 吸收有效信息
        # ==========================================
        # Query 只有 1 个 token。Key/Value 是 10 个 factor_tokens。
        # 被 key_padding_mask 标记为 True 的因子，注意力得分在 softmax 之前会被强行置为 -inf。
        # 它们对 CLS Token 的更新贡献绝对、百分之百为 0！
        attn_output, _ = self.cross_attn(
            query=query,
            key=factor_tokens,
            value=factor_tokens,
            key_padding_mask=key_padding_mask
        )

        # ==========================================
        # 第四步：输出最终的宏观指令表征
        # ==========================================
        # 拿走 CLS Token 吸收完精华后的向量: (B, 1, d_model) -> (B, d_model)
        condition_emb = attn_output.squeeze(1)

        # 经过 MLP 提取并归一化
        final_condition = self.output_mlp(condition_emb)

        return final_condition

class DisentangledConstraintConditioner(nn.Module):
    def __init__(self, f_dim=10, d_model=256):
        """
        f_dim: 因子数量 (默认 10 个)
        d_model: 最终输出的隐层维度
        """
        super().__init__()
        self.f_dim = f_dim

        # 内部处理时，每个因子的表征维度。
        # 让 10 个因子的信息足够宽广，再用 MLP 压缩
        self.inner_dim = d_model // 2

        # ==========================================
        # 1. 状态编码器 (State Encoder) -> 解决 "0" 的语义冲突
        # ==========================================
        # num_embeddings=2 表示只有两个状态: 0(无约束) 和 1(有约束)
        # 为每个因子独立学习两套状态，因为不同因子"被放开"的含义可能在全局上不同
        self.state_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=2, embedding_dim=self.inner_dim)
            for _ in range(f_dim)
        ])

        # ==========================================
        # 2. 数值编码器 (Value Encoder)
        # ==========================================
        # 每个因子的目标暴露是一个标量 (1维)，把它升维到 inner_dim
        self.value_encoders = nn.ModuleList([
            nn.Linear(1, self.inner_dim)
            for _ in range(f_dim)
        ])

        # ==========================================
        # 3. 终极指令融合网络
        # ==========================================
        # 把 10 个因子的独立表征拼接后 (10 * inner_dim)，压缩成最终指令 D 维
        self.fusion_mlp = nn.Sequential(
            nn.Linear(f_dim * self.inner_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, target_exp, active_mask):
        """
        target_exp: (B, f_dim) 10 个因子的约束值 (浮点数)
        active_mask: (B, f_dim) 1 表示有约束，0 表示无约束 (整数/布尔值)
        """
        B = target_exp.shape[0]
        factor_embs = []

        # 对每一个因子进行独立的绝对隔离处理
        for i in range(self.f_dim):
            # 获取当前因子的目标值和 Mask
            # 形状都变成 (B, 1) 以便后续操作
            val_i = target_exp[:, i].unsqueeze(-1)
            mask_i = active_mask[:, i].long()  # 转为整数索引用于 Embedding

            # ----------------------------------------------------
            # 核心魔法 1：极其明确的【状态语义注入】
            # 这个 Embedding 清清楚楚地告诉网络："我现在到底有没有被约束！"
            # state_emb: (B, inner_dim)
            # ----------------------------------------------------
            state_emb = self.state_embeddings[i](mask_i)

            # ----------------------------------------------------
            # 核心魔法 2：绝对隔绝的【数值门控】
            # 如果 mask=0，val_emb 变成全 0 向量。
            # 此时网络绝不会把它误认为"中性约束=0"，因为上面的 state_emb
            # 已经大声广播了："这个全是 0 的数值是废弃特征，请无视！"
            # ----------------------------------------------------
            val_emb = self.value_encoders[i](val_i)
            # 乘法屏蔽，斩断暗物质污染
            val_emb = val_emb * mask_i.unsqueeze(-1).float()

            # 融合当前因子的状态与数值
            f_emb = state_emb + val_emb
            factor_embs.append(f_emb)

        # 把 10 个极其纯净的、没有语义歧义的因子特征拼接起来
        # concat_embs: (B, f_dim * inner_dim)
        concat_embs = torch.cat(factor_embs, dim=-1)

        # 降维成终极的风控指令 condition_emb: (B, d_model)
        condition_emb = self.fusion_mlp(concat_embs)

        return condition_emb

# ==========================================
# 模块一：行业编码器 (Industry Encoder)
# 输入: B * M * I (31维行业暴露)
# 输出: B * M * D (行业 ID Token)
# ==========================================
class IndustryIDEncoder(nn.Module):
    def __init__(self, num_industries=31, d_model=256):
        super().__init__()
        # 因为输入已经是 31 维的暴露度向量（支持硬one-hot，也支持软权重）
        # 我们使用一个小型的 MLP 把它映射到 d_model 的高维隐空间
        self.encoder = nn.Sequential(
            nn.Linear(num_industries, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)  # 归一化，保证和其它 Token 的量级一致
        )

    def forward(self, industry_exp):
        """
        industry_exp: (B, M, I) 例如 I=31
        返回: industry_token (B, M, D)
        """
        return self.encoder(industry_exp)


# ==========================================
# 模块 2：条件编码器 (带非线性的 MLP)
# ==========================================
class ConstraintConditioner(nn.Module):
    def __init__(self, f_dim, d_model):
        """
        f_dim: 因子数量 F
        d_model: 你的模型主特征维度 D
        """
        super().__init__()
        # 推荐使用隐藏层放大维度，充分混合因子特征
        hidden_dim = max(d_model // 2, f_dim * 4)

        self.mlp = nn.Sequential(
            nn.Linear(f_dim, hidden_dim),
            nn.GELU(),  # 现代架构首选的平滑非线性激活函数
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, target_exp):
        """
        target_exp: (B, F)
        返回: condition_emb (B, d_model)
        """
        return self.mlp(target_exp)

# ==========================================
# 模块 3: 基于风格因子的股票ID生成
# ==========================================
class BarraStockIDEncoder(nn.Module):
    """
    纯静态 Barra 因子 → 动态股票 ID 编码
    阶段1: 因子截取 -> B*M*L*10
    阶段2: 时序聚合 (长期性格) + 最新一天状态 (今日坐标) + MLP 映射 -> B*M*D
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()

        # 相对索引 (基于截取后的 10 维因子)
        self.static_idx = [0, 1, 4, 5, 6, 7]  # 6 个静态因子
        self.dynamic_idx = [2, 3, 8, 9]  # 4 个动态因子

        # 特征数量 = 6(静态均值) + 4(动态均值) + 4(动态标准差) + 10(最后一天最新状态) = 24
        n_in = 24

        self.norm = nn.LayerNorm(n_in)
        self.mlp = nn.Sequential(
            nn.Linear(n_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: B * M * L * 10
        factors = x

        # ==========================================
        # 1. 提取最新一天的状态 (极度关键！用于对齐约束目标)
        # ==========================================
        # latest_factors: B * M * 10
        latest_factors = factors[:, :, -1, :]

        # ==========================================
        # 2. 分离静态与动态因子，计算时序聚合 (股票长期性格)
        # ==========================================
        static_x = factors[..., self.static_idx]  # B * M * L * 6
        dynamic_x = factors[..., self.dynamic_idx]  # B * M * L * 4

        static_mean = static_x.mean(dim=2)  # B * M * 6
        dynamic_mean = dynamic_x.mean(dim=2)  # B * M * 4
        dynamic_std = dynamic_x.std(dim=2, correction=0)  # B * M * 4

        # ==========================================
        # 3. 拼接特征：长期性格 + 此时此刻
        # ==========================================
        # feat 维度: 6 + 4 + 4 + 10 = 24
        feat = torch.cat([static_mean, dynamic_mean, dynamic_std, latest_factors], dim=-1)

        # ==========================================
        # 4. MLP 映射为 d_model 维度的 Stock ID Token
        # ==========================================
        out = self.mlp(self.norm(feat))  # B * M * d_model

        # 强烈建议返回 (B, M, D) 而不是展平，这样后续极其方便加上 Condition_Embedding
        return out
        # return out.reshape(-1, out.shape[-1])  # (B*M) * D

# ==========================================
# 模块 4: 将行业和风格因子的ID融合
# ==========================================
class TwoFoldTokenFuser(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        # 拼接后的总维度是 3 * d_model
        # 我们使用一个漏斗状的 MLP，先稍微升维/保持维度以充分混合特征，再降回 d_model
        hidden_dim = d_model * 2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

        # 定义两个标量参数，用于控制残差的权重，初始化为一个小值（比如 0.5）
        self.alpha_style = nn.Parameter(torch.tensor(0.5))
        self.alpha_industry = nn.Parameter(torch.tensor(0.5))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, style_token, industry_token):
        """
        style_token:    (B, M, D) 股票的风格画像 (来自你之前的 static_ID_encoder)
        industry_token: (B, M, D) 股票的行业归属 (来自上方的 IndustryIDEncoder)
        """

        merged_features = torch.cat([
            style_token,
            industry_token,
        ], dim=-1)

        # 3. 跨特征融合降维 (Cross-Feature Fusion)
        # 形状变回 (B, M, D)
        mlp_out = self.fusion_mlp(merged_features)

        # 2. 【核心魔法】残差注入 (Residual Injection)！
        # 把老板的约束指令，直接通过加法硬加到最终特征上，打通梯度高速公路！
        final_out = mlp_out + self.alpha_style * style_token + self.alpha_industry * industry_token

        # 3. 最终归一化
        final_stock_token = self.final_norm(final_out)

        return final_stock_token

class AdaLNConstraintProjectionForPatchTS(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        # 基础归一化，在最后一个维度 D 上执行，对每个 Patch 独立归一化
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # 将约束指令映射为缩放和平移参数
        self.condition_to_gamma_beta = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU()
        )

    def forward(self, x_seq, condition_emb):
        """
        x_seq: (B, C, P, D) 股票数, 通道数, Patch数, 特征维度
        condition_emb: (D,) 或 (1, D) 或 (B, D) 全局约束指令
                       如果是 (B, D) 则要求这 B 只股票在这一天共享相同的指令行
        """
        # 1. 对 Patch 级特征进行归一化
        # (B, C, P, D)
        normalized_x = self.norm(x_seq)

        # 2. 约束参数映射
        # (1, 2*D) 或 (B, 2*D)
        gamma_beta = self.condition_to_gamma_beta(condition_emb)

        # 3. 切分为 gamma 和 beta
        # (1, D) 或 (B, D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # 4. 关键的维度对齐！
        # 如果 condition_emb 传来的是 (D,)，这里会变成 (1, 1, 1, D)
        # 如果 condition_emb 传来的是 (B, D) 且全是相同的值，这里变成 (B, 1, 1, D)
        # 我们需要在索引 1(通道) 和索引 2(Patch) 的位置插入广播维度

        # 无论输入情况如何，先确保它是 2D 的 (N, D)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(0)
            beta = beta.unsqueeze(0)

        # 插入股票 M 的维度
        gamma = gamma.unsqueeze(1)# 变成 (B, 1, D)
        beta = beta.unsqueeze(1) # 变成 (B, 1, D)

        # 5. 核心投影调制
        # 即使 N=1，PyTorch 也会自动将这一个指令广播给所有的 B 只股票！
        projected_x = normalized_x * (1 + gamma) + beta

        return projected_x

'''因子收益全局塔数据预处理 V2'''

class FactorReturnPreprocessorV2(nn.Module):
    def __init__(self, F=42, clip_threshold=5.0, eps=1e-8):
        super().__init__()
        self.clip_threshold = clip_threshold
        self.eps = eps

        # 定义一个针对 42 个因子各自独立的可学习权重 (通道缩放器)
        # 初始化为 1.0，让网络自己决定放大或缩小哪些因子
        self.factor_weight = nn.Parameter(torch.ones(1, 1, 1, F))

    def forward(self, factor_returns):
        """
        factor_returns: (B, 1, L, F)
        """
        # 1. 测算各自真实的绝对波动率 (寻找尺子)
        std = torch.std(factor_returns, dim=2, keepdim=True)
        factor_scale = torch.clamp(std, min=self.eps)

        # 2. 剥离量纲，提取纯粹时序形态 (放大到 1.0 附近)
        scaled_returns = factor_returns / factor_scale

        # 3. 极值截断 (防爆，干掉黑天鹅)
        clipped_returns = torch.clamp(scaled_returns, min=-self.clip_threshold, max=self.clip_threshold)
        clipped_returns = torch.nan_to_num(clipped_returns, nan=0.0)

        # ==========================================
        # 4. 【核心修复】: 物理级量纲还原与重塑
        # ==========================================

        # 步骤 A：把被截掉黑天鹅的、极其干净的波浪，乘回它原来的比例尺！
        # 此时数据不仅恢复了真实的大小关系(国家因子 > 行业因子)，而且绝对干净。
        restored_returns = clipped_returns * factor_scale

        # 步骤 B：解决梯度消失的痛点
        # 既然数据又变回了 1e-3 量级，为了让下游网络好受，
        # 我们用一个可学习的、针对每个因子的乘法权重 (factor_weight) 去整体放大它！
        # 网络会自动学到：为了梯度好传，整体乘个 100 ；同时通过微调 42 个权重，赋予不同因子重要性。
        final_features = restored_returns * self.factor_weight

        return final_features

'''门控专家池 的 MOE任务头'''
class GatedContextMoEHead(nn.Module):
    def __init__(self, d_model=128, num_experts=8, hidden_dim=64):
        super().__init__()

        # ==========================================
        # 1. 路由网络 (Router)
        # 输入: 风控指令 (D) + 宏观大盘环境 (D) = 2*D
        # 输出: 给 8 个专家的路由打分
        # ==========================================
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_experts)
        )

        # ==========================================
        # 2. 专家网络池 (Experts)
        # ==========================================
        # 每个专家包含两部分：
        # (1) 宏观门控生成器：把宏观特征变为 0~1 的缩放器
        # (2) 核心打分网络：只处理被门控缩放后的个股特征 + 前日仓位

        self.macro_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()  # 生成 0~1 的乘法缩放系数
            ) for _ in range(num_experts)
        ])

        # 核心网络的输入：被门控调制后的特征(D) + 前日仓位(1) = D + 1
        in_features = d_model + 1

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)  # 输出 1 维 Logit
            ) for _ in range(num_experts)
        ])

    def forward(self, rep_sm, rep_si, rep_sm_cash, condition_emb, prev_wx):
        """
        rep_sm:        [B, 1, D]   大盘/宏观表征
        rep_si:        [B, M, D]   所有个股表征 (已被 AdaLN 严加看管)
        rep_sm_cash:   [B, 1, D]   现金项表征
        condition_emb: [B, D]      当天的风控约束指令
        prev_wx:       [B, 1+M, 1] 前日权重 (第0位是现金)
        """
        B, M, D = rep_si.shape

        # ==========================================
        # 第一步：整合候选资产池 (Assets = Cash + Stocks)
        # 保证现金排在第 0 位，完全对齐 prev_wx
        # rep_assets: [B, 1+M, D]
        # ==========================================
        rep_assets = torch.cat([rep_sm_cash, rep_si], dim=1)

        # ==========================================
        # 第二步：生成路由权重 (Router)
        # ==========================================
        # 将宏观特征展平 [B, D]
        macro_flat = rep_sm.squeeze(1)

        # 路由依据：风控指令 + 宏观大环境
        router_input = torch.cat([condition_emb, macro_flat], dim=-1)

        # 路由打分与 Softmax 软路由
        route_logits = self.router(router_input)

        # route_weights: [B, num_experts] (返回给外部用于日志监控)
        route_weights = F.softmax(route_logits, dim=-1)

        # 增加一个维度以备稍后的广播乘法: [B, 1, num_experts]
        route_weights_expanded = route_weights.unsqueeze(1)

        # ==========================================
        # 第三步：各路专家独立进行“门控调制打分”
        # ==========================================
        expert_outputs = []

        for i in range(len(self.experts)):
            # 1. 生成属于该专家的宏观滤镜门控
            # gate: [B, 1, D] (对所有股票进行无差别的宏观洗礼)
            gate = self.macro_gates[i](macro_flat).unsqueeze(1)

            # 2. 乘法门控融合！
            # 这是架构的核心防御：用客观大势(gate)去激化个股的特性(rep_assets)，
            # 但绝不给大势单独绕过 AdaLN 创造 Logits 的机会！
            # gated_assets: [B, 1+M, D]
            gated_assets = rep_assets * gate

            # 3. 拼装前日仓位
            # expert_input: [B, 1+M, D+1]
            expert_input = torch.cat([gated_assets, prev_wx], dim=-1)

            # 4. 专家给出最终打分
            # [B, 1+M, 1] -> squeeze -> [B, 1+M]
            logit = self.experts[i](expert_input).squeeze(-1)
            expert_outputs.append(logit)

        # ==========================================
        # 第四步：动态加权融合 (MoE)
        # ==========================================
        # 堆叠专家打分: [B, 1+M, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        # [B, 1+M, num_experts] * [B, 1, num_experts] -> 沿最后一维求和 -> [B, 1+M]
        final_logits = (expert_outputs * route_weights_expanded).sum(dim=-1)

        # 返回打分，以及路由权重(用于监控和正则化分析)
        return final_logits, route_weights

class FullAttention(nn.Module):
    """
    使用 SDPA 优化的 Attention 操作
    """
    def __init__(self, scale, attention_dropout):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values):
        # SDPA 要求的形状是 [B, H, L, E]，而你传入的是 [B, L, H, E]
        # 因此需要进行 permute 转置
        queries = queries.permute(0, 2, 1, 3) # [B, H, L, E]
        keys = keys.permute(0, 2, 1, 3)       # [B, H, S, E]
        values = values.permute(0, 2, 1, 3)   # [B, H, S, D]

        # 如果没有指定 scale，SDPA 默认就是 1/sqrt(E)
        # 如果指定了，通过参数传入
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(queries.size(-1))

        CHUNK_SIZE = 60000
        if queries.shape[0] > CHUNK_SIZE:
            out_list = []
            for i in range(0, queries.shape[0], CHUNK_SIZE):
                out_chunk = F.scaled_dot_product_attention(
                    queries[i:i+CHUNK_SIZE], 
                    keys[i:i+CHUNK_SIZE], 
                    values[i:i+CHUNK_SIZE],
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                    scale=scale
                )
                out_list.append(out_chunk)
            out = torch.cat(out_list, dim=0)
        else:
            out = F.scaled_dot_product_attention(
                queries, 
                keys, 
                values, 
                attn_mask=None, 
                dropout_p=self.dropout_p if self.training else 0.0, 
                is_causal=False, 
                scale=scale)

        # 将形状还原回 [B, L, H, D] 以适配原有代码的 view 操作
        return out.permute(0, 2, 1, 3).contiguous()


class AttentionLayer(nn.Module):
    """
    The Multi-head Self-Attention (MSA) Layer
    """

    def __init__(self, d_model, n_heads, d_keys, d_values, dropout):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TwoStageAttentionLayer, self).__init__()
        self.time_attention = AttentionLayer(d_model, n_heads, d_keys=None, d_values=None, dropout=dropout)
        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff),nn.GELU(),nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,s,p,d = x.shape

        x = rearrange(x, 'b s p d -> (b s) p d')

        x = self.norm1(x + self.dropout(self.time_attention(x,x,x)))
        x = self.norm2(x + self.dropout(self.ffn1(x)))

        x = rearrange(x, '(b s) p d -> b s p d', b=b, s=s, p=p)

        return x
    

class StockAttentionLayer(nn.Module):
    """
    The Stock Attention Layer
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(StockAttentionLayer, self).__init__()
        self.stock_attention = AttentionLayer(d_model, n_heads, d_keys=None, d_values=None, dropout=dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),nn.GELU(),nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,m,d = x.shape

        x = self.norm1(x + self.dropout(self.stock_attention(x,x,x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


class TwoStageAttentionLayer_PSS(nn.Module):
    """
    The Two Stage Attention (TSA) Layer 二阶段注意力+股票方向自注意力
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, self_atten):
        super(TwoStageAttentionLayer_PSS, self).__init__()
        self.tsa_layer = TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout)
        self.atten = self_atten
        if self.atten :
            self.stock_attention = AttentionLayer(d_model, n_heads, d_keys=None, d_values=None, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                      nn.GELU(),
                                      nn.Linear(d_ff, d_model))

    def forward(self, x):
        # 先执行二阶段注意力层
        B, M, N, D = x.shape
        tsa_out = self.tsa_layer(x)  # [B,M,N,D]
        output = tsa_out

        if self.atten :
            tsa_out = tsa_out.permute(0, 2, 1, 3)  # [B,N,M,D]
            tsa_out = tsa_out.reshape(B * N, M, D)  # [B*N,M,D]

            # 在股票方向上做self-attention操作, 缺点在于计算量大(显存占用非常大)
            stock_enc = self.stock_attention(tsa_out, tsa_out, tsa_out)  # 股票方向自注意力 [B*N,M,D]

            output = tsa_out + self.dropout(stock_enc)  # 残差连接
            output = self.norm1(output)  # 层归一化
            output = output + self.dropout(self.MLP1(output))  # MLP
            output = self.norm2(output)  # 层归一化  [B*N,M,D]

            # 还原到原本的数据形状
            output = output.reshape(B, N, M, D)  # [B,N,M,D]
            output = output.permute(0, 2, 1, 3)  # [B,M,N,D]

        return output

class scale_block_tsa_pss(nn.Module):
    """
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter 'depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    """

    def __init__(self, d_model, n_heads, d_ff, depth, dropout, self_atten=False):
        super(scale_block_tsa_pss, self).__init__()
        self.encode_layers = nn.ModuleList()
        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer_PSS(d_model, n_heads, d_ff, dropout, self_atten))

    def forward(self, x):
        for layer in self.encode_layers:
            #print('x shape scale', x.shape)
            x = layer(x)
        return x


class Encoder_tsa_pss(nn.Module):
    """
    The Encoder of Crossformer.
    """

    def __init__(self, e_blocks, d_model, n_heads, d_ff, block_depth, dropout, self_atten=False):
        super(Encoder_tsa_pss, self).__init__()
        self.encode_blocks = nn.ModuleList()

        for i in range(0, e_blocks):
            self.encode_blocks.append(scale_block_tsa_pss(d_model, n_heads, d_ff, block_depth, dropout, self_atten))

    def forward(self, x):
        for block in self.encode_blocks:
            # x, encode_x = block(x, encode_x)
            x = checkpoint.checkpoint(block, x, use_reentrant=False)
        return x


class StockRep_TSA_PSS(nn.Module):
    """
    单股票表征模型，输入为[B,M,L,C]，其中B为batch size，L为预测长度，C为通道数
    输出为[B,M,d_model]
    """

    def __init__(self, seq_len, channels, d_model, layers, blocks,dropout, self_atten=True):
        super(StockRep_TSA_PSS, self).__init__()
        self.channels = channels  # 通道数
        self.seq_len = seq_len  # 输入序列长度
        self.d_model = d_model  # 隐藏层维度
        self.elayers = layers  # encoder层数
        self.atten = self_atten

        self.fusion = nn.Sequential(nn.Linear(self.channels, self.d_model * 4),
                                    nn.GELU(),
                                    nn.Linear(self.d_model * 4, d_model))

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.seq_len, d_model))

        self.pre_norm = nn.LayerNorm(d_model)  # PP任务用

        # Encoder
        if self.atten:
            '''因子信息输入模块'''
            self.industry_encoder = IndustryIDEncoder(num_industries=31, d_model=self.d_model)
            self.static_ID_encoder = BarraStockIDEncoder(d_model=d_model, dropout=0.1)
            self.stock_dna = TwoFoldTokenFuser(d_model=d_model, dropout=0.1)
            
            self.encoder = Encoder_tsa_pss(e_blocks=self.elayers, d_model=d_model, n_heads=8, d_ff=d_model * 4,
                                            block_depth=blocks, dropout=dropout, self_atten=self.atten)
        else:
            self.encoder = Encoder_tsa_pss(e_blocks=self.elayers, d_model=d_model, n_heads=8, d_ff=d_model * 4,
                                           block_depth=blocks, dropout=dropout, self_atten=self.atten)
        
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(nn.Linear(d_model, int(d_model ** 0.5)),
                                  nn.GELU(),
                                  nn.Linear(int(d_model ** 0.5), 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputSeq, factor_step,  Industry_exp_ID):
        """
        输入为[B,M,L,C]，其中B为batch size，L为输入序列长度，C为通道数

        """

        '''引入股票池的所有个股行业作独立编码'''
        Ind_ID = self.industry_encoder(Industry_exp_ID)  # B*M*D 基于个股对31个行业作embedding
        factor_ID = self.static_ID_encoder(factor_step)  # B*M*D 基于风格因子的ID编码
        Si_ID = self.stock_dna(factor_ID, Ind_ID)

        #print ('before fusion', inputSeq.shape)
        inputSeq = self.fusion(inputSeq)  # [B, M, L, D]
        #print('after fusion', inputSeq.shape)

        inputSeq += self.pos_embedding
        inputSeq = self.pre_norm(inputSeq)

        B, M, L, D = inputSeq.shape
        inputSeq = inputSeq + Si_ID.unsqueeze(2).expand(-1, -1, L, -1)

        #print('before encoder', inputSeq.shape)

        # 编码器
        enc_out = self.encoder(inputSeq)  # [B, M, N, D]

        score = self.score(self.norm(enc_out)).squeeze(-1)  # [B,M,N]
        attn = torch.softmax(score/self.temperature.clamp(min=1e-4), dim=-1)  # [B,M,N]
        encode_x = torch.sum(enc_out * attn.unsqueeze(-1), dim=-2)  # [B,M,D]

        return encode_x #BMD

    def forward_MKT(self, x_seq):
        """
        输入为[B,M,L,C]，其中B为batch size，L为输入序列长度，C为通道数
        """

        x_seq = self.fusion(x_seq)  # [B, M, L, D]

        x_seq += self.pos_embedding
        x_seq = self.pre_norm(x_seq)

        # 编码器
        enc_out = self.encoder(x_seq)  # [B, M, N, D]

        score = self.score(self.norm(enc_out)).squeeze(-1)  # [B,M,N]
        attn = torch.softmax(score/self.temperature.clamp(min=1e-4), dim=-1)  # [B,M,N]
        encode_x = torch.sum(enc_out * attn.unsqueeze(-1), dim=-2)  # [B,M,D]

        return encode_x #BMD


'''=========================== 5分钟数据分支 ==========================='''

class MinutePatchFusion(nn.Module):
    """
    将连续patch_size根5分钟bar拼接为一个patch token
    x: [B, M, L, C] -> [B, M, L//patch_size, D]
    """

    def __init__(self, channels: int, d_model: int, patch_size: int = 48, dropout: float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = patch_size * channels
        self.fusion = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        B, M, L, C = x.shape
        assert L % self.patch_size == 0, f"分钟序列长度{L}必须能被patch_size={self.patch_size}整除"
        x = x.reshape(B, M, L // self.patch_size, self.patch_size * C)
        return self.fusion(x)


class MinuteTSALayer(nn.Module):
    """
    时间方向自注意力(同一股票的各patch之间) + 可选的股票方向自注意力(同一patch的各股票之间)
    input/output: [B, M, N, D]
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, stock_attn=True):
        super().__init__()
        self.time_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                       dropout=dropout, activation="gelu", batch_first=True,
                                                       norm_first=True)
        self.stock_attn = stock_attn
        if stock_attn:
            self.stock_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                            dropout=dropout, activation="gelu", batch_first=True,
                                                            norm_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, valid=None):
        B, M, N, D = x.shape
        if valid is None:
            valid = torch.ones((B, M, N), dtype=torch.bool, device=x.device)
        def safe_padding(v):
            padding = ~v
            padding = padding.clone()
            padding[padding.all(dim=-1), 0] = False
            return padding
        x = self.time_encoder(x.reshape(B * M, N, D),
                              src_key_padding_mask=safe_padding(valid.reshape(B * M, N)))
        x = x.reshape(B, M, N, D) * valid.unsqueeze(-1)
        if self.stock_attn:
            x = x.permute(0, 2, 1, 3).reshape(B * N, M, D)
            stock_valid = valid.permute(0, 2, 1).reshape(B * N, M)
            x = self.stock_encoder(x, src_key_padding_mask=safe_padding(stock_valid))
            x = x.reshape(B, N, M, D).permute(0, 2, 1, 3)
        return self.norm(x) * valid.unsqueeze(-1)


class MinuteStockRep(nn.Module):
    """
    5分钟数据的单股票表征(沿用T1V2的分钟编码结构)
    输入: [B, M, L_day, 48, C] (C为分钟数据集的原始字段数)
    输出: [B, M, out_dim]
    """

    # 分钟数据集中不作为模型输入的字段(存在才剔除)
    ID_FIELDS = ["ts_code", "gen_natural_id"]
    CALENDAR_FIELDS = ["gen_year", "gen_month", "gen_day", "gen_week", "gen_minute"]

    def __init__(self, scaler_file, seq_days, patch_size, d_model, out_dim, layers, dropout,
                 stock_attn=True, drop_calendar=True, n_heads=8):
        super().__init__()
        with open(scaler_file, "r", encoding="utf-8") as f:
            ms_fields = list(json.load(f).keys())
        mask_channels = [f for f in self.ID_FIELDS + (self.CALENDAR_FIELDS if drop_calendar else []) if f in ms_fields]

        self.normalizer = DataNormalizer(scaler_file=scaler_file,
                                         priceFieldList=["open", "high", "low", "close"],
                                         base_price_field="close",
                                         mask_channels=mask_channels,
                                         extra_orignal_fields=[],
                                         log1p_fields=["vol", "amount"],
                                         signed_log1p_fields=[],
                                         sample_norm="time_stock",
                                         log1p_negative_policy="error",
                                         )
        self.close_index = self.normalizer.src_field_list.index("close")
        channels = int(self.normalizer.keep_mask.sum().item())

        assert (seq_days * 48) % patch_size == 0, "seq_days*48 必须能被 ms_patch_size 整除"
        self.patch_size = patch_size
        self.num_patches = seq_days * 48 // patch_size

        self.fusion = MinutePatchFusion(channels=channels, d_model=d_model, patch_size=patch_size, dropout=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList([
            MinuteTSALayer(d_model, n_heads, d_model * 4, dropout, stock_attn) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(nn.Linear(d_model, int(d_model ** 0.5)),
                                   nn.GELU(),
                                   nn.Linear(int(d_model ** 0.5), 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, m_seq):
        B, M, Ld, bars, C = m_seq.shape
        x = m_seq.reshape(B, M, Ld * bars, C).float()

        # 收盘价<=0的bar视为停牌/缺失数据
        valid = x[..., self.close_index] > 0  # [B, M, L]
        x = self.normalizer(x, valid_mask=valid)  # [B, M, L, C']

        x = self.fusion(x)  # [B, M, N, D]
        x = self.pre_norm(x + self.pos_embedding)

        patch_valid = valid.reshape(B, M, self.num_patches, self.patch_size).any(dim=-1)
        x = x * patch_valid.unsqueeze(-1)

        for layer in self.encoder:
            x = layer(x, patch_valid)

        # patch内全部无效的patch不参与注意力池化
        patch_valid = valid.reshape(B, M, self.num_patches, self.patch_size).any(dim=-1)  # [B, M, N]
        score = self.score(self.norm(x)).squeeze(-1) / self.temperature.clamp(min=1e-4)  # [B, M, N]
        score = score.float().masked_fill(~patch_valid, -1e4)
        attn = torch.softmax(score, dim=-1).to(x.dtype)
        rep = torch.sum(x * attn.unsqueeze(-1), dim=-2)  # [B, M, D]
        rep = rep * patch_valid.any(dim=-1, keepdim=True).to(rep.dtype)  # 整段无分钟数据的股票置0

        return self.out_proj(rep) * patch_valid.any(dim=-1, keepdim=True).to(rep.dtype)


class MinuteGatedFusion(nn.Module):
    """
    把分钟表征以门控残差的方式注入日线表征: out = rep_day + gate * proj(rep_ms)
    proj最后一层零初始化 => 初始时 out == rep_day, 模型与纯日线模型完全等价, 可直接加载日线模型参数热启动
    """

    def __init__(self, d_model, ms_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(ms_dim),
                                  nn.Linear(ms_dim, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, rep_day, rep_ms, return_parts=False):
        delta = self.proj(rep_ms)
        g = self.gate(torch.cat([rep_day, delta], dim=-1))
        out = rep_day + g * delta
        if return_parts:
            return out, g, delta
        return out


def _is_rank0():
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
        or torch.distributed.get_rank() == 0


@torch.no_grad()
def rep_stats(name, x):
    """
    表征统计: x [B, M, D]
    cs_std: 同一维度在股票截面上的标准差(按batch和维度取平均), 越接近0说明各股票表征越趋同
    """
    x = x.detach().float()
    bad = (~torch.isfinite(x)).sum().item()
    return (f"{name:<14s} shape={tuple(x.shape)} mean={x.mean().item():+.4f} std={x.std().item():.4f} "
            f"min={x.min().item():+.4f} max={x.max().item():+.4f} cs_std={x.std(dim=1).mean().item():.4f} "
            f"nan/inf={bad} | stock0[:6]={[round(v, 4) for v in x[0, 0, :6].tolist()]}")


class PortfolioModel_SMV2(nn.Module):
    def __init__(self, args: Namespace):
        super(PortfolioModel_SMV2, self).__init__()
        self.M = args.M  # 资产数
        self.d_model = args.d_model
        self.time_step = args.time_step
        self.use_prev_w = args.use_prev_w > 0  # 是否使用上一时刻的权重作为输入

        # 5分钟数据分支
        self.use_minute = getattr(args, 'use_minute', 0) > 0
        if self.use_minute:
            self.representation_ms = MinuteStockRep(scaler_file=args.ms_scaler_info_path,
                                                    seq_days=args.ms_time_step,
                                                    patch_size=args.ms_patch_size,
                                                    d_model=args.ms_d_model,
                                                    out_dim=args.ms_d_model,
                                                    layers=args.ms_layers,
                                                    dropout=args.dropout,
                                                    stock_attn=args.ms_stock_attn > 0,
                                                    drop_calendar=args.ms_drop_calendar > 0)
            self.ms_fusion = MinuteGatedFusion(d_model=self.d_model, ms_dim=args.ms_d_model)
        #self.data_normalizer = DataNormalizer(args.scaler_info_path, args.priceRelatedField, args.mask_channel)
        #self.channels = args.channels - 50 - 62 - 1
        # 定义单股票特征编码器
        self.representation_si = StockRep_TSA_PSS(seq_len=self.time_step,
                                                  channels= 302, #self.channels,
                                                  d_model=self.d_model,
                                                  layers=args.layers,
                                                  blocks=args.blocks,
                                                  dropout=args.dropout, self_atten=True)
        self.representation_sm = StockRep_TSA_PSS(seq_len=self.time_step,
                                                  channels=42,
                                                  d_model=self.d_model,
                                                  layers=args.layers,
                                                  blocks=args.blocks,
                                                  dropout=args.dropout, self_atten=False)
        
        self.pre_norm = nn.LayerNorm(self.d_model)
        '''
        self.shrink_layer = nn.Sequential(nn.Linear(self.d_model, self.d_model * 4),
                                          nn.GELU(),
                                          nn.Linear(self.d_model * 4, 1))'''

        # 因子约束模块：

        self.preprocessor = FactorReturnPreprocessorV2(F=42, clip_threshold=5.0)

        self.conditioner = CrossAttentionConstraintConditionerV2(f_dim=10, d_model=self.d_model)
        # 定义输出层，将权重图进行softmax
        self.quant_head = PMQuantTaskHead(
            d_model=self.d_model,
            num_experts=8,
            n_heads=8,
            num_layers=3,
            dropout=args.dropout,
        )
        self.csrank_preprocessor = CSRankPreprocessor(rank_mode="gaussian")

        # 调试: 每debug_rep次前向打印一次日线/分钟表征统计, 0为关闭
        self.debug_rep = getattr(args, 'debug_rep', 0)
        self._fwd_count = 0
      

    # def forward(self, input_seq, benchmark_weights):
    def forward(self, prev_w, raw_seq,  target_exp, active_mask, target_weights, ms_seq=None):
        """
        :param prev_w: [B, 1+M] 表示1+M个资产的前一时刻的权重
        :param raw_seq: [B, M, L, C] 表示M个股票在过去L天的股票数据
        :param target_exp: [B, F] 表示目标期望收益
        :param ms_seq: [B, M, ms_time_step, 48, C_ms] 回看窗口最后ms_time_step天的5分钟数据(use_minute时必须提供)
        :return: wt_next: [B, 1+M] 表示1+M个资产的下一时刻的权重
        """

        #input_seq=input_seq[:,:,:,1:] #剔除股票ID
        # =========================================================
        # 1. 数据切分
        # =========================================================

        input_seq = raw_seq[..., :311].clone().float()
        factor_step = raw_seq[..., 311:321]
        global_seq = raw_seq[:, :1, :, 353:]
        Industry_exp_ID = raw_seq[:, :, -1, 322:353]

        input_seq = self.csrank_preprocessor(
                    input_seq=input_seq, 
                    base_price_index=22, 
                    eps=1e-6
                )
        #inputSeq = self.data_normalizer(input_seq)

        

        #print("inut_seq double check : ", input_seq.shape)

        #rep_si = self.representation_si(input_seq, factor_step, target_exp, Industry_exp_ID)
        rep_si = checkpoint.checkpoint(self.representation_si, input_seq, factor_step,  Industry_exp_ID,
                               use_reentrant=False)

        show = self.debug_rep > 0 and self._fwd_count % self.debug_rep == 0 and _is_rank0()
        self._fwd_count += 1
        if show:
            print(f"==== 表征检查 forward#{self._fwd_count - 1} ({'train' if self.training else 'eval'}) ====")
            print(rep_stats('rep_day', rep_si))

        # 5分钟数据表征, 门控注入日线个股表征
        if self.use_minute:
            assert ms_seq is not None, "use_minute=1 时必须提供5分钟数据 ms_seq"
            rep_ms = checkpoint.checkpoint(self.representation_ms, ms_seq, use_reentrant=False)  # [B, M, D_ms]
            rep_day = rep_si
            rep_si, gate, delta = self.ms_fusion(rep_day, rep_ms, return_parts=True)
            ms_valid = (ms_seq[..., self.representation_ms.close_index] > 0).any(dim=(-1, -2))
            rep_si = torch.where(ms_valid.unsqueeze(-1), rep_si, rep_day)
            if show:
                with torch.no_grad():
                    inj = (gate * delta).float()
                    ratio = inj.norm(dim=-1) / rep_day.float().norm(dim=-1).clamp_min(1e-8)  # 分钟注入量/日线表征 的范数比
                    cos = F.cosine_similarity(rep_day.float(), delta.float(), dim=-1)
                print(rep_stats('rep_ms', rep_ms))
                print(rep_stats('ms_delta', delta))
                print(rep_stats('rep_fused', rep_si))
                print(f"{'fusion':<14s} gate_mean={gate.float().mean().item():.4f} "
                      f"inject/day_norm={ratio.mean().item():.6f} cos(day,delta)={cos.mean().item():+.4f}")

        rep_si = self.pre_norm(rep_si)  # [B, M, D]

        # 1. 预处理分离
        Barra_return_to_backbone = self.preprocessor(global_seq) # B*1*L*F
        rep_sm = self.representation_sm.forward_MKT(Barra_return_to_backbone)  # [B,1, D]
        rep_sm = self.pre_norm(rep_sm)  # [B,1,D]

   
    
        '''构造持仓因子约束条件编码 --- 推理任务头要输入'''
        # 配置头前向推断 (外部 competition_layer 已由头内部完整交互取代)
        condition_emb = self.conditioner(target_exp, active_mask)
        logits = self.quant_head(
            si_rep=rep_si,
            sm_rep=rep_sm,
            pre_w=prev_w,
            target_exp = target_exp,
            condition_emb = condition_emb,
            target_weights = target_weights,

        )

        # 5. 生成组合权重
        next_w = F.softmax(logits, dim=-1)  # [B, 1+M]

        return next_w

