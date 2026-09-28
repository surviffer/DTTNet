import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import get_norm

# baseline 不经过本模块，由 DPTDFNet 直接关闭前端
FUSION_MODES = ("fixed", "learned_static", "dynamic", "mid_only")


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn_norm, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            get_norm(bn_norm, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class MRBranch(nn.Module):
    """
    单路浅层卷积分支
    第一版建议：2层 3x3 Conv
    """
    def __init__(self, channels, bn_norm, num_layers=2, bias=False):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvBNAct(channels, channels, kernel_size=3, bn_norm=bn_norm, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WeightNet(nn.Module):
    """
    输入相关的片段级动态权重
    输入：三路特征
    输出：(B, 3)，依次为 short / mid / long
    """
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(channels * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, f_s, f_m, f_l):
        # GAP: (B, C, F, T) -> (B, C)
        z = torch.cat([f.mean(dim=(-1, -2)) for f in (f_s, f_m, f_l)], dim=1)   # (B, 3C)
        z = F.relu(self.fc1(z))
        return torch.softmax(self.fc2(z), dim=1)


class FreqResampler(nn.Module):
    """
    按物理频率把 n_fft_src 的频谱重采样到中窗前 dim_f 个 bin
    中窗第 k 个 bin 的频率 = k * sr / n_fft_mid，
    在源谱中的位置 p_k = k * n_fft_src / n_fft_mid（一般不是整数），做线性插值
    """
    def __init__(self, n_fft_src, n_fft_mid, dim_f):
        super().__init__()
        pos = torch.arange(dim_f, dtype=torch.float64) * n_fft_src / n_fft_mid
        idx0 = pos.floor().long()
        w = (pos - idx0).float()

        # 线性插值需要 idx0 + 1，因此保留到 idx0[-1] + 1
        self.num_src_bins = int(idx0[-1].item()) + 2
        assert self.num_src_bins <= n_fft_src // 2 + 1, "中窗 f_max 超出源谱的奈奎斯特频率"

        self.n_fft_src = n_fft_src
        self.n_fft_mid = n_fft_mid
        self.dim_f = dim_f
        self.register_buffer("idx0", idx0, persistent=False)
        self.register_buffer("w", w.view(1, 1, -1, 1), persistent=False)

    def crop(self, x):
        # (B, C, F_src, T) -> (B, C, num_src_bins, T)，先去掉 f_max 以上的频点
        return x[:, :, :self.num_src_bins]

    def forward(self, x):
        # (B, C, num_src_bins, T) -> (B, C, dim_f, T)
        assert x.shape[2] == self.num_src_bins
        x0 = x.index_select(2, self.idx0)
        x1 = x.index_select(2, self.idx0 + 1)
        w = self.w.to(x.dtype)
        return x0 * (1 - w) + x1 * w


class MRFrontend(nn.Module):
    """
    多分辨率前端
    - 中窗路直接使用 DTT 原始 first_conv 之后的特征 f_base
    - 短窗、长窗先裁到中窗最高频率，各自做 1x1 stem，再按物理频率对齐到中窗频点
    - 三路浅层卷积分支
    - 按 fusion_mode 融合：
        fixed          固定 (1/3, 1/3, 1/3)
        learned_static 输入无关的可学习全局权重
        dynamic        输入相关的片段级动态权重（DRFF）
        mid_only       等容量对照：结构和参数量与 fixed 完全相同，
                       但 short/long 两路的 stem 输入换成中窗谱，固定 1/3 权重
    """
    def __init__(
        self,
        dim_c_in,
        g,
        bn_norm,
        dim_f,
        n_fft_mid,
        n_fft_short,
        n_fft_long,
        fusion_mode="dynamic",
        num_branch_layers=2,
        weight_hidden_dim=128,
        bias=False,
    ):
        super().__init__()
        assert fusion_mode in FUSION_MODES, f"未知 fusion_mode: {fusion_mode}"
        self.fusion_mode = fusion_mode

        # 所有模式都创建两路 stem，保证 mid_only 与 fixed 的可训练结构完全相同
        self.stem_short = self._stem(dim_c_in, g, bn_norm, bias)
        self.stem_long = self._stem(dim_c_in, g, bn_norm, bias)
        if fusion_mode != "mid_only":
            # FreqResampler 只有不可训练的 buffer，不影响参数量
            self.resample_short = FreqResampler(n_fft_short, n_fft_mid, dim_f)
            self.resample_long = FreqResampler(n_fft_long, n_fft_mid, dim_f)

        # 三路浅层卷积分支：结构相同，参数独立
        self.branch_short = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)
        self.branch_mid = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)
        self.branch_long = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)

        if fusion_mode == "dynamic":
            self.weight_net = WeightNet(g, hidden_dim=weight_hidden_dim)
        elif fusion_mode == "learned_static":
            # 全零 logits，softmax 后初始权重为 (1/3, 1/3, 1/3)
            self.static_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("fixed_alpha", torch.full((3,), 1.0 / 3), persistent=False)

        # 最近一次前向的融合权重 (B, 3)，仅用于日志和评估
        self.last_alpha = None

    @staticmethod
    def _stem(dim_c_in, g, bn_norm, bias):
        return nn.Sequential(
            nn.Conv2d(dim_c_in, g, kernel_size=1, bias=bias),
            get_norm(bn_norm, g),
            nn.ReLU()
        )

    def _alpha(self, f_s, f_m, f_l):
        b = f_m.shape[0]
        if self.fusion_mode == "dynamic":
            return self.weight_net(f_s, f_m, f_l)
        if self.fusion_mode == "learned_static":
            return torch.softmax(self.static_logits, dim=0).expand(b, 3)
        return self.fixed_alpha.expand(b, 3)

    def forward(self, x_short, f_mid_base, x_long, x_mid=None):
        """
        参数：
        x_short: 短窗原始谱图，shape=(B, dim_c_in, F_s, T)，未裁剪；mid_only 时为 None
        f_mid_base: 中窗 first_conv 后特征，shape=(B, g, dim_f, T)
        x_long: 长窗原始谱图，shape=(B, dim_c_in, F_l, T)，未裁剪；mid_only 时为 None
        x_mid: 中窗原始谱图，shape=(B, dim_c_in, dim_f, T)，仅 mid_only 使用

        返回：
        f_fused: shape=(B, g, dim_f, T)
        """
        if self.fusion_mode == "mid_only":
            # 与 fixed 走同样的 stem -> 分支，只是输入换成中窗谱（无需频率对齐）
            assert x_mid is not None, "mid_only 需要中窗原始谱 x_mid"
            f_s0 = self.stem_short(x_mid)
            f_l0 = self.stem_long(x_mid)
        else:
            # 同一 hop_length 且 center=True 时三路帧数必然一致
            assert x_short.shape[-1] == f_mid_base.shape[-1] == x_long.shape[-1], \
                "三路 STFT 时间帧数不一致，检查 hop_length 和 center"
            f_s0 = self.resample_short(self.stem_short(self.resample_short.crop(x_short)))
            f_l0 = self.resample_long(self.stem_long(self.resample_long.crop(x_long)))

        f_s = self.branch_short(f_s0)
        f_m = self.branch_mid(f_mid_base)
        f_l = self.branch_long(f_l0)

        alpha = self._alpha(f_s, f_m, f_l)
        self.last_alpha = alpha.detach()

        a = alpha.to(f_m.dtype).view(-1, 3, 1, 1, 1)
        return a[:, 0] * f_s + a[:, 1] * f_m + a[:, 2] * f_l
