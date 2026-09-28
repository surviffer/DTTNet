from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from src.dp_tdf.mr_frontend import FreqResampler

ROOT = Path(__file__).resolve().parents[2]
MODES = ["baseline", "fixed", "learned_static", "dynamic", "mid_only"]
DIM_T = 32  # 小尺寸，CPU 上也能跑


def build(mode, stem="vocals"):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{stem}.yaml")
    cfg.dim_t = DIM_T
    cfg.bn_norm = "BN"
    cfg.mr_frontend.enabled = mode != "baseline"
    if mode != "baseline":
        cfg.mr_frontend.fusion_mode = mode
    return hydra.utils.instantiate(cfg)


def wav(batch=2, scale=1.0, model=None):
    length = model.chunk_size if model is not None else 1024 * (DIM_T - 1)
    return torch.randn(batch, 2, length) * scale


@pytest.mark.parametrize("n_src", [4096, 8192])
@pytest.mark.parametrize("dim_f", [2048, 864])
def test_resampler_matches_physical_frequency(n_src, dim_f):
    r = FreqResampler(n_src, 6144, dim_f)
    # 以频率值本身作为“频谱”，插值结果应等于中窗各 bin 的频率
    f_src = torch.arange(r.num_src_bins, dtype=torch.float32) * 44100 / n_src
    out = r(r.crop(f_src.view(1, 1, -1, 1).repeat(1, 1, 1, 3)))
    f_mid = torch.arange(dim_f, dtype=torch.float32) * 44100 / 6144
    torch.testing.assert_close(out[0, 0, :, 0], f_mid, atol=1e-2, rtol=0)


@pytest.mark.parametrize("stem", ["vocals", "bass"])
def test_same_output_shape_all_modes(stem):
    torch.manual_seed(0)
    x = wav()
    shapes = set()
    for mode in MODES:
        m = build(mode, stem).eval()
        with torch.no_grad():
            shapes.add(tuple(m(m.multi_stft(x)).shape))
    assert len(shapes) == 1


def test_baseline_skips_short_long():
    m = build("baseline")
    specs = m.multi_stft(wav(batch=1))
    assert specs["short"] is None and specs["long"] is None
    assert not hasattr(m, "mr_frontend")
    assert m.fusion_diagnostics() is None


def test_mr_model_rejects_single_window_input():
    m = build("dynamic").eval()
    with pytest.raises(AssertionError):
        m(m.stft(wav(batch=1)))


@pytest.mark.parametrize("mode", ["fixed", "learned_static", "dynamic", "mid_only"])
def test_alpha_properties(mode):
    torch.manual_seed(0)
    m = build(mode).eval()
    with torch.no_grad():
        m(m.multi_stft(wav()))
        a1 = m.fusion_diagnostics()["alpha"].clone()
        m(m.multi_stft(wav(scale=5.0)))
        a2 = m.fusion_diagnostics()["alpha"].clone()

    assert a1.shape == (2, 3)
    assert (a1 >= 0).all()
    torch.testing.assert_close(a1.sum(1), torch.ones(2))
    if mode in ("fixed", "mid_only"):
        torch.testing.assert_close(a1, torch.full_like(a1, 1 / 3))
    if mode == "learned_static":
        torch.testing.assert_close(a1, torch.full_like(a1, 1 / 3))  # 全零 logits 初始化
        torch.testing.assert_close(a1, a2)                          # 与输入无关
    if mode == "dynamic":
        assert not torch.allclose(a1, a2)                           # 随输入变化


def test_mid_only_has_same_trainable_structure_as_fixed():
    # mid_only 是等容量对照：可训练参数的名称和形状必须与 fixed 完全一致
    def shapes(m):
        return {k: tuple(p.shape) for k, p in m.named_parameters() if p.requires_grad}
    assert shapes(build("mid_only")) == shapes(build("fixed"))


@pytest.mark.parametrize("mode", ["fixed", "learned_static", "dynamic", "mid_only"])
def test_all_branches_receive_grad(mode):
    torch.manual_seed(0)
    m = build(mode).train()
    m(m.multi_stft(wav())).abs().mean().backward()
    for name in ["branch_short", "branch_mid", "branch_long", "stem_short", "stem_long"]:
        for p in getattr(m.mr_frontend, name).parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
    if mode == "learned_static":
        assert m.mr_frontend.static_logits.grad.abs().sum() > 0


def test_zero_fusion_scale_equals_baseline():
    # fusion_scale 为 0 时，多分辨率模型与共享主干权重的 baseline 输出完全一致
    torch.manual_seed(0)
    base = build("baseline").eval()
    mr = build("dynamic").eval()
    mr.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        mr.fusion_scale.zero_()
        x = wav()
        torch.testing.assert_close(mr(mr.multi_stft(x)), base(base.multi_stft(x)))


def test_istft_length():
    m = build("dynamic").eval()
    x = wav(batch=1, model=m)
    with torch.no_grad():
        out = m.istft(m(m.multi_stft(x)))
    assert out.shape[-1] == x.shape[-1]


def test_optimizer_groups_skip_decay_for_logits_and_scale():
    m = build("learned_static")
    opt = m.configure_optimizers()
    no_decay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert id(m.mr_frontend.static_logits) in no_decay
    assert id(m.fusion_scale) in no_decay
    all_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.window_mid) not in all_params
