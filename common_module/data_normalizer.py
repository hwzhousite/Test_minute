import copy
import json
import os

import torch
import torch.nn as nn


# 高置信度的非负长尾字段。订单买卖金额、成交量和成交笔数通过前缀补全。
DEFAULT_LOG1P_FIELDS = {
    'f_p_adj_factor',
    'f_v_volume', 'f_v_amount', 'f_v_avg_3M',
    'f_v_trades_total', 'f_v_trades',
    'f_i_mv_total', 'f_i_mv_circ',
    'f_i_shares_total', 'f_i_shares_float', 'f_i_shares_free',
    'f_i_oper_rev_ttm', 'f_i_oper_rev_lyr',
    'f_v_call_open_volume', 'f_v_call_open_trades',
    'f_v_call_close_volume', 'f_v_call_close_trades',
    'f_i_buy_amount', 'f_i_buy_volume',
    'f_i_sell_amount', 'f_i_sell_volume',
    'f_i_large_buy_amount', 'f_i_large_buy_volume',
    'f_i_large_sell_amount', 'f_i_large_sell_volume',
    'f_i_margin_balance', 'f_i_margin_buy', 'f_i_margin_repay',
    'f_i_seclending_balance', 'f_i_seclending_volume',
    'f_i_seclending_sell', 'f_i_seclending_repay',
    'f_i_margin_total', 'f_i_margin_balance_volume',
    'f_i_seclending_sell_amount', 'f_i_seclending_repay_amount',
    'f_i_vma_1M', 'f_i_vma_5d', 'f_i_vma_22d', 'f_i_vma_60d',
    'f_i_tapi_16d', 'f_i_tapi_6d', 'f_i_vstd_10d',
}

DEFAULT_LOG1P_PREFIXES = ('f_v_buy_', 'f_v_sell_')


# 可能跨过0，并且绝对规模具有长尾的字段。
DEFAULT_SIGNED_LOG1P_FIELDS = {
    'f_i_pe', 'f_i_pb', 'f_i_pe_ttm',
    'f_i_pcf_ocf', 'f_i_pcf_ocf_ttm',
    'f_i_pcf_ncf', 'f_i_pcf_ncf_ttm',
    'f_i_ps', 'f_i_ps_ttm', 'f_i_price_div_dps',
    'f_i_net_profit_ttm', 'f_i_net_profit_lyr', 'f_i_net_assets',
    'f_i_cash_flow_oper_ttm', 'f_i_cash_flow_oper_lyr',
    'f_i_cash_flow_total_ttm', 'f_i_cash_flow_total_lyr',
    'f_i_obv', 'f_i_obv_obv', 'f_i_pvt',
    'f_i_wvad_wvad', 'f_i_wvad_mawvad',
    'f_i_wad', 'f_i_mawad',
    'f_i_vmacd', 'f_i_vmacd_dea', 'f_i_vmacd_macd', 'f_i_sobv',
}

DEFAULT_SIGNED_LOG1P_PREFIXES = (
    'f_v_diff_',
    'f_v_net_inflow',
    'f_v_inflow_open',
    'f_v_inflow_close',
    'f_v_inflow_large',
)


def _build_default_transform_fields(field_list):
    """根据字段名生成默认log1p与signed_log1p集合。"""
    log1p_fields = {
        field for field in field_list
        if field in DEFAULT_LOG1P_FIELDS
        or field.startswith(DEFAULT_LOG1P_PREFIXES)
    }

    signed_log1p_fields = {
        field for field in field_list
        if field in DEFAULT_SIGNED_LOG1P_FIELDS
        or (
            field.startswith(DEFAULT_SIGNED_LOG1P_PREFIXES)
            and '_rate' not in field
        )
    }
    return log1p_fields, signed_log1p_fields


class DataNormalizer(nn.Module):
    def __init__(
        self,
        scaler_file,
        priceFieldList,
        mask_channels,
        extra_orignal_fields=None,
        base_price_field='f_p_close_adj',
        sample_norm="time_stock",
        eps=1e-8,
        log1p_fields=None,
        signed_log1p_fields=None,
        log1p_negative_policy="signed",
        strict_transform_fields=False,
    ):
        """
        Args:
            scaler_file: 字段scaler配置文件；其key顺序必须与输入通道一致。
            priceFieldList: 需要除以最后一天基准价的价格量纲字段。
            mask_channels: 标准化完成后从输出中删除的字段。
            log1p_fields: 非负长尾字段；None表示使用本文件内置清单。
            signed_log1p_fields: 可正可负长尾字段；None表示使用内置清单。
            log1p_negative_policy: log1p字段遇到负值时的策略：
                "signed"（默认，保留符号）、"clamp"（截为0）或"error"。
            strict_transform_fields: True时，用户清单中不存在的字段会报错。

        处理顺序：价格相对化 -> 长尾变换 -> ENUM -> 样本内均值方差
        -> 删除mask_channels。
        """
        super().__init__()

        assert os.path.exists(scaler_file)

        self.scaler = json.load(open(scaler_file, "r", encoding="utf-8"))
        self.src_field_list = list(self.scaler.keys())

        self.priceFieldList = list(priceFieldList)
        self.sample_norm = sample_norm
        self.eps = eps

        valid_negative_policies = {"signed", "clamp", "error"}
        if log1p_negative_policy not in valid_negative_policies:
            raise ValueError(
                f"log1p_negative_policy must be one of "
                f"{valid_negative_policies}, got {log1p_negative_policy!r}"
            )
        self.log1p_negative_policy = log1p_negative_policy

        self.keep_original_list = copy.deepcopy(priceFieldList)
        self.keep_original_list += extra_orignal_fields if extra_orignal_fields else []

        self.price_index = [self.src_field_list.index(f) for f in self.priceFieldList]

        self.base_price_index = self.src_field_list.index(base_price_field)

        default_log1p, default_signed_log1p = _build_default_transform_fields(
            self.src_field_list
        )
        requested_log1p = (
            default_log1p if log1p_fields is None else set(log1p_fields)
        )
        requested_signed_log1p = (
            default_signed_log1p
            if signed_log1p_fields is None
            else set(signed_log1p_fields)
        )

        unknown_fields = (
            requested_log1p | requested_signed_log1p
        ) - set(self.src_field_list)
        if strict_transform_fields and unknown_fields:
            raise KeyError(
                f"transform fields not found in scaler: {sorted(unknown_fields)}"
            )

        # 价格字段、保留原值字段和ENUM字段不参与长尾变换。
        candidate_log1p = requested_log1p & set(self.src_field_list)
        candidate_signed_log1p = requested_signed_log1p & set(self.src_field_list)
        overlap = candidate_log1p & candidate_signed_log1p
        if overlap:
            raise ValueError(
                f"fields cannot be in both log1p and signed_log1p: "
                f"{sorted(overlap)}"
            )

        self.enum_index = []
        self.mms_index = []

        enum_min = []
        enum_scale = []

        for i, field in enumerate(self.src_field_list):
            info = self.scaler[field]

            if field in self.keep_original_list:
                continue

            if info[0] == "ENUM":
                self.enum_index.append(i)
                enum_min.append(info[1])
                enum_scale.append(info[2] - info[1] + 1)

            elif info[0] == "MMS":
                self.mms_index.append(i)

        mms_fields = {self.src_field_list[i] for i in self.mms_index}
        excluded_fields = set(self.keep_original_list) | set(self.priceFieldList)
        self.log1p_fields = sorted(
            candidate_log1p & mms_fields - excluded_fields
        )
        self.signed_log1p_fields = sorted(
            candidate_signed_log1p & mms_fields - excluded_fields
        )

        self.register_buffer(
            "log1p_index",
            torch.tensor(
                [self.src_field_list.index(f) for f in self.log1p_fields],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "signed_log1p_index",
            torch.tensor(
                [self.src_field_list.index(f) for f in self.signed_log1p_fields],
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.register_buffer(
            "enum_min",
            torch.tensor(enum_min, dtype=torch.float32)
            if len(enum_min) else torch.empty(0)
        )

        self.register_buffer(
            "enum_scale",
            torch.tensor(enum_scale, dtype=torch.float32)
            if len(enum_scale) else torch.empty(0)
        )
        
        keep_mask = torch.ones(len(self.src_field_list), dtype=torch.bool)

        for field in mask_channels:
            keep_mask[self.src_field_list.index(field)] = False

        self.register_buffer("keep_mask", keep_mask)

    @staticmethod
    def _signed_log1p(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _normalize_price(self, x, valid_mask=None):
        if self.base_price_index is None or len(self.price_index) == 0:
            return x

        if valid_mask is None:
            base = x[:, :, -1, self.base_price_index]
        else:
            # 以每只股票最后一个有效时间步的基准价为基准; 全部无效时用1.0占位(输出随后会被置0)
            L = x.shape[2]
            steps = torch.arange(L, device=x.device).view(1, 1, L)
            last_valid = torch.where(valid_mask, steps, torch.zeros_like(steps)).amax(dim=2)  # [B,M]
            base = torch.gather(x[..., self.base_price_index], 2, last_valid.unsqueeze(-1)).squeeze(-1)
            base = torch.where(valid_mask.any(dim=2), base, torch.ones_like(base))
        base = base.unsqueeze(2).unsqueeze(3)

        x[..., self.price_index] = x[..., self.price_index] / (base + self.eps) - 1.0
        return x

    def _normalize_enum(self, x):
        if len(self.enum_index) == 0:
            return x

        tmp = x[..., self.enum_index]
        tmp = (tmp - self.enum_min) / self.enum_scale
        x[..., self.enum_index] = tmp
        return x

    def _transform_long_tail(self, x):
        """在均值方差标准化之前压缩长尾。"""
        if self.log1p_index.numel() > 0:
            tmp = x[..., self.log1p_index]

            if self.log1p_negative_policy == "signed":
                # 合法的非负值与普通log1p完全相同；异常负值不会产生NaN。
                tmp = self._signed_log1p(tmp)
            elif self.log1p_negative_policy == "clamp":
                tmp = torch.log1p(tmp.clamp_min(0.0))
            else:  # error
                if torch.any(tmp < 0):
                    raise ValueError(
                        "negative value found in a log1p field; use "
                        "log1p_negative_policy='signed' or clean the data"
                    )
                tmp = torch.log1p(tmp)

            x[..., self.log1p_index] = tmp

        if self.signed_log1p_index.numel() > 0:
            tmp = x[..., self.signed_log1p_index]
            x[..., self.signed_log1p_index] = self._signed_log1p(tmp)

        return x

    def _normalize_mms(self, x, valid_mask=None):
        if len(self.mms_index) == 0:
            return x

        tmp = x[..., self.mms_index]

        if valid_mask is not None:
            # 只用有效时间步统计均值方差
            dims = {"time": (2,), "time_stock": (1, 2), "stock": (1,)}.get(self.sample_norm)
            if dims is None:
                raise ValueError(self.sample_norm)
            w = valid_mask.unsqueeze(-1).to(tmp.dtype)
            cnt = w.sum(dim=dims, keepdim=True).clamp_min(1.0)
            mean = (tmp * w).sum(dim=dims, keepdim=True) / cnt
            std = (((tmp - mean) ** 2) * w).sum(dim=dims, keepdim=True).div(cnt).sqrt()
            x[..., self.mms_index] = (tmp - mean) / (std + self.eps)
            return x

        if self.sample_norm == "time":
            mean = tmp.mean(dim=2, keepdim=True)
            std = tmp.std(dim=2, keepdim=True, unbiased=False)

        elif self.sample_norm == "time_stock":
            mean = tmp.mean(dim=(1,2), keepdim=True)
            std = tmp.std(dim=(1,2), keepdim=True, unbiased=False)

        elif self.sample_norm == "stock":
            mean = tmp.mean(dim=1, keepdim=True)
            std = tmp.std(dim=1, keepdim=True, unbiased=False)

        else:
            raise ValueError(self.sample_norm)

        tmp = (tmp - mean) / (std + self.eps)
        x[..., self.mms_index] = tmp
        return x

    def forward(self, data_batch, valid_mask=None):
        """
        data_batch: [B,M,L,C]
        valid_mask: [B,M,L] 可选, True表示该时间步为有效数据(非停牌/非填充);
                    给定时, 基准价取最后一个有效时间步, 均值方差只在有效时间步上统计, 无效时间步输出置0
        """
        x = data_batch.clone()

        x = self._normalize_price(x, valid_mask)
        x = self._transform_long_tail(x)
        x = self._normalize_enum(x)
        x = self._normalize_mms(x, valid_mask)

        x = x[..., self.keep_mask]
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x
