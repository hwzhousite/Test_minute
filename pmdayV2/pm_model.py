from argparse import Namespace
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import cvxpy as cp
from common_module.data_normalizer import DataNormalizer as DataNormalizer

def optimize(raw_weight, benchmark_weight, w_old, lambda_tc=0.005, lambda_te=2, solver=cp.OSQP):

    if isinstance(raw_weight, torch.Tensor):
        raw_weight = raw_weight.cpu().numpy()

    if isinstance(w_old, torch.Tensor):
        w_old = w_old[1:].cpu().numpy()

    if isinstance(benchmark_weight, torch.Tensor):
        w_b = benchmark_weight.cpu().numpy()

    M = raw_weight.shape[0]

    w = cp.Variable(M)

    objective = cp.Minimize(cp.sum_squares(w - raw_weight) + lambda_tc * cp.norm1(w - w_old) + lambda_te * cp.sum_squares(w - w_b))

    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver, warm_start=True,)

    if w.value is None:
        raise RuntimeError("QP failed.")

    return torch.from_numpy(w.value).float()
    

class TemporalPatchFusion(nn.Module):
    def __init__(self, channels: int, d_model: int, patch_size: int = 6, dropout: float = 0.0):
        super().__init__()

        self.channels = channels
        self.d_model = d_model
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
        """
        x: [B, M, L, C]

        return:
            out:        [B, M, ceil(L / patch_size), D]
            patch_mask: [B, M, ceil(L / patch_size)]
        """
        B, M, L, C = x.shape
        num_patches = L // self.patch_size

        # [B,M,L,C] -> [B,M,num_patches,patch_size,C]
        x = x.reshape(B, M, num_patches, self.patch_size, C,)

        # [B,M,num_patches,6,C] -> [B,M,num_patches,6*C]
        x = x.reshape(B, M, num_patches, self.patch_size * C, )

        # [B,M,num_patches,6*C] -> [B,M,num_patches,D]
        out = self.fusion(x)

        return out


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer 二阶段注意力+股票方向自注意力
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        time_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(
            encoder_layer=time_attention,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        stock_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stock_encoder = nn.TransformerEncoder(
            encoder_layer=stock_attention,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, x):
        # 先执行二阶段注意力层
        B, M, N, D = x.shape
        x = x.reshape(B * M, N, D)  # [B*M,N,D]

        # time_mask = mask.reshape(B * M, N) if mask is not None else None  # [B*M,N]
        ta_out = self.time_encoder(x)  # [B*M,N,D]
        ta_out = ta_out.reshape(B, M, N, D)  # [B,M,N,D]

        ta_out = ta_out.permute(0, 2, 1, 3)  # [B,N,M,D]
        ta_out = ta_out.reshape(B * N, M, D)  # [B*N,M,D]

        # stock_mask = mask.permute(0, 2, 1).reshape(B * N, M) if mask is not None else None  # [B*N,M]
        sa_out = self.stock_encoder(ta_out)  # 股票方向自注意力 [B*N,M,D]

        # 还原到原本的数据形状
        sa_out = sa_out.reshape(B, N, M, D)  # [B,N,M,D]
        sa_out = sa_out.permute(0, 2, 1, 3)  # [B,M,N,D]

        return sa_out


class Encoder(nn.Module):
    """
    The Encoder of Crossformer.
    """

    def __init__(self, blocks, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.encode_blocks = nn.ModuleList([
            TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(blocks)
        ])

    def forward(self, x):
        for block in self.encode_blocks:
            # x = checkpoint.checkpoint(block, x, use_reentrant=False)
            x = block(x)
        return x


class StockRep(nn.Module):
    """
    单股票表征模型，输入为[B,M,L,C]，其中B为batch size，L为预测长度，C为通道数
    输出为[B,M,d_model]
    """

    def __init__(self, seq_len, patch_size, channels, d_model, layers, dropout):
        super().__init__()
        self.channels = channels  # 通道数
        self.seq_len = seq_len  # 输入序列长度
        self.patch_size = patch_size  # 补丁大小
        self.d_model = d_model  # 隐藏层维度
        self.elayers = layers  # encoder层数

        # self.fusion = nn.Sequential(nn.Linear(self.channels, self.d_model * 4),
        #                             nn.GELU(),
        #                             nn.Linear(self.d_model * 4, d_model))
        self.fusion = TemporalPatchFusion(channels=self.channels, d_model=self.d_model, patch_size=self.patch_size, dropout=dropout)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.seq_len//self.patch_size, d_model))

        self.pre_norm = nn.LayerNorm(d_model)  # PP任务用

        # Encoder
        self.encoder = Encoder(blocks=self.elayers, d_model=d_model, n_heads=8, d_ff=d_model * 4, dropout=dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Sequential(nn.Linear(d_model, int(d_model ** 0.5)),
                                  nn.GELU(),
                                  nn.Linear(int(d_model ** 0.5), 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, m_seq):
        """
        输入为[B,M,L,C]，其中B为batch size，L为输入序列长度，C为通道数
        """
        B, M, L, C = m_seq.shape

        x_seq = self.fusion(m_seq)  # [B, M, N, D]

        x_seq += self.pos_embedding
        x_seq = self.pre_norm(x_seq)

        # 编码器
        enc_out = self.encoder(x_seq)  # [B, M, N, D]

        score = self.score(self.norm(enc_out)).squeeze(-1)  # [B,M,N]
        attn = torch.softmax(score/self.temperature.clamp(min=1e-4), dim=-1)  # [B,M,N]
        encode_x = torch.sum(enc_out * attn.unsqueeze(-1), dim=-2)  # [B,M,D]

        return encode_x #BMD


class PortfolioModel(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.M = args.M  # 资产数
        self.d_model = args.d_model
        self.time_step = args.time_step

        self.minute_normalizer = DataNormalizer(scaler_file=args.ms_scaler_info_path,
                                                priceFieldList=["open","high","low","close"],
                                                base_price_field="close",
                                                mask_channels=["ts_code", "gen_natural_id"],
                                                extra_orignal_fields=[],
                                                log1p_fields=["vol","amount"],
                                                signed_log1p_fields=[],
                                                sample_norm="time_stock",
                                                log1p_negative_policy="error",
                                            )

        # 定义单股票特征编码器
        self.representation_si = StockRep(seq_len=self.time_step*48,  # 5分钟数据一天有48个时间步
                                          patch_size=48,  # 每个patch包含48个时间步
                                          channels=14-2,  # 14个特征中去掉ts_code和gen_natural_id
                                          d_model=self.d_model,
                                          layers=args.layers,
                                          dropout=args.dropout)
        
        self.head = nn.Sequential(nn.LayerNorm(self.d_model),
                                          nn.Linear(self.d_model, self.d_model * 4),
                                          nn.GELU(),
                                          nn.Linear(self.d_model * 4, 1))

    def forward(self, m_seq, dev, benchmark_weights, prev_w):
        """
        :param prev_w: [B, 1+M] 表示1+M个资产的前一时刻的权重
        :param input_seq: [B, M, L, C] 表示M个股票在过去L天的股票数据
        :param benchmark_weights: [B, M] 表示基准投资组合的权重
        :return: wt_next: [B, 1+M] 表示1+M个资产的下一时刻的权重
        """

        device = m_seq.device

        B,M,L,m,_ = m_seq.shape
        m_seq = m_seq.reshape(B, M, L * m, -1)
        # padding_mask = m_seq[..., -1]
        mseq = self.minute_normalizer(m_seq)

        rep_si = self.representation_si(mseq)  # [B, M, d_model]
        rep = self.head(rep_si).squeeze(-1)  # [B, M]

        active_w = dev * torch.tanh(rep)
        active_w_mean = active_w.mean(dim=1, keepdim=True)
        active_w_centered = active_w - active_w_mean

        raw_portfolio_weights = benchmark_weights + active_w_centered
        
        if self.training:
            final_weights = raw_portfolio_weights
        else:
            weights = []
            
            for b in range(B):
                weight = optimize(raw_portfolio_weights[b], benchmark_weights[b], prev_w[b])
                weights.append(weight)

            final_weights = torch.stack(weights, dim=0).to(device)

        cash_features = torch.zeros(B, 1, device=device)  # [B, 1]
        wt_next = torch.cat([cash_features, final_weights], dim=-1)

        return rep, wt_next
