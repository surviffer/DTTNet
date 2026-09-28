import torch.nn as nn
import torch

from src.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from src.dp_tdf.bandsequence import BandSequenceModelModule

from src.layers import (get_norm)
from src.dp_tdf.abstract import AbstractModel
from src.dp_tdf.mr_frontend import MRFrontend

class DPTDFNet(AbstractModel):
    def __init__(self, num_blocks, l, g, k, bn, bias, bn_norm, bandsequence, block_type, mr_frontend=None, **kwargs):

        super(DPTDFNet, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.num_blocks = num_blocks #U-Net主框架encoder和decoder共有多少个block
        self.l = l #
        self.g = g  #通道数增量
        self.k = k  #应该是卷积核大小
        self.bn = bn
        self.bias = bias    #卷积层/线性层要不要带偏置项（bias）

        self.n = num_blocks // 2    #encoder和decoder各一半
        scale = (2, 2)  #上采样和下采样都按2倍缩放

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        # fusion_mode: baseline | fixed | learned_static | dynamic | mid_only
        mr_cfg = dict(mr_frontend or {})
        enabled = mr_cfg.pop("enabled", mr_frontend is not None)
        fusion_mode = mr_cfg.pop("fusion_mode", "dynamic")
        fusion_scale_init = mr_cfg.pop("fusion_scale_init", 0.1)
        mr_cfg.pop("align_mode", None)  # 旧配置项，频率对齐已改为按物理频率插值
        self.fusion_mode = fusion_mode if enabled else "baseline"
        self.use_mr_frontend = self.fusion_mode != "baseline"

        if self.use_mr_frontend:
            self.mr_frontend = MRFrontend(
                dim_c_in=self.dim_c_in,
                g=g,
                bn_norm=bn_norm,
                bias=bias,
                dim_f=self.dim_f,
                n_fft_mid=self.n_fft_mid,
                n_fft_short=self.n_fft_short,
                n_fft_long=self.n_fft_long,
                fusion_mode=self.fusion_mode,
                **mr_cfg
            )
            # 残差缩放系数，三种多分辨率模式共用；从预训练 baseline 微调时设为 0
            self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))

        f = self.dim_f  #当前频率大小，下采样f=f/2，上采样f=f*2
        c = g   #first_conv后的通道数
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        """
            输入为 multi_stft() 返回的 dict，包含 short/mid/long
            baseline 也可以直接输入中窗 tensor，多分辨率模型不允许
        """
        if not isinstance(x, dict):
            # 多分辨率模型收到单窗输入会静默丢掉前端，直接报错
            assert not self.use_mr_frontend, "多分辨率模型必须使用 multi_stft() 的 dict 输入"
            x = {"mid": x}

        # 中窗主路：复用原始first_conv
        f_base = self.first_conv(x["mid"])
        if self.use_mr_frontend:
            f_fused = self.mr_frontend(x["short"], f_base, x["long"], x_mid=x["mid"])
            x = f_base + self.fusion_scale * f_fused
        else:
            x = f_base

        x = x.transpose(-1, -2)
        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")

            # 跳连乘法统一在 FP32 中计算；FP16 下乘积可能超出可表示范围，
            # 所有模式都做同样的 clamp 后再转回 FP16。BF16 不需要 clamp。
            skip = ds_outputs[-i - 1]
            out_dtype = skip.dtype
            with torch.autocast(device_type=x.device.type, enabled=False):
                x = x.float() * skip.float()
                if out_dtype == torch.float16:
                    fp16_max = torch.finfo(torch.float16).max
                    x = x.clamp(min=-fp16_max, max=fp16_max)
                x = x.to(out_dtype)
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x

    def fusion_diagnostics(self):
        """最近一次前向的融合权重，baseline 返回 None"""
        if not self.use_mr_frontend or self.mr_frontend.last_alpha is None:
            return None
        return {
            "alpha": self.mr_frontend.last_alpha,   # (B, 3): short, mid, long
            "fusion_scale": self.fusion_scale.detach(),
        }
