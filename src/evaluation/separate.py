from pathlib import Path

import torch
import numpy as np
import math
import os
from src.utils.utils import split_nparray_with_overlap, join_chunks

# 单窗 ONNX 推理（separate_with_onnx*）和旧版 separate_with_ckpt 已移除：
# 它们只调用中窗 model.stft，会静默绕过多分辨率前端。


def check_checkpoint_meta(model, checkpoint):
    '''
    校验 checkpoint 元数据与外部配置一致；旧 checkpoint 没有元数据时跳过
    '''
    meta = checkpoint.get("drff_meta")
    if meta is None:
        return
    mode = getattr(model, "fusion_mode", "baseline")
    assert meta["fusion_mode"] == mode, f"checkpoint 是 {meta['fusion_mode']}，配置是 {mode}"
    assert meta["target_name"] == model.target_name, \
        f"checkpoint 是 {meta['target_name']}，配置是 {model.target_name}"
    assert list(meta["n_fft"]) == [model.n_fft_short, model.n_fft_mid, model.n_fft_long], \
        f"checkpoint STFT 参数 {meta['n_fft']} 与配置不一致"
    assert meta["dim_f"] == model.dim_f and meta["hop_length"] == model.hop_length, \
        "checkpoint 的 dim_f / hop_length 与配置不一致"


def separate_with_ckpt_TDF(batch_size, model, ckpt_path: Path, mix, device, double_chunk, overlap_add):
    '''
    Args:
        batch_size: the inference batch size
        model: the model to be used
        ckpt_path: the path to the checkpoint
        mix: (c, t)
        device: the device to be used
        double_chunk: whether to use double chunk size
        overlap_add: overlap-add 配置，None 表示关闭
    Returns:
        target_wav_hat: (c, t)
        alphas: 每个片段的 [alpha_short, alpha_mid, alpha_long, fusion_scale]，baseline 为空列表
    '''
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_checkpoint_meta(model, checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)
    if double_chunk:
        inf_ck = model.inference_chunk_size
    else:
        inf_ck = model.chunk_size

    if overlap_add is None:
        return no_overlap_inference(model, mix, device, batch_size, inf_ck)

    if not os.path.exists(overlap_add.tmp_root):
        os.makedirs(overlap_add.tmp_root)
    return overlap_inference(model, mix, device, batch_size, inf_ck, overlap_add.overlap_rate, overlap_add.tmp_root, overlap_add.samplerate)


def _collect_alphas(model, alphas):
    # 按片段顺序追加融合权重；baseline 没有前端，不追加
    diag = model.fusion_diagnostics()
    if diag is None:
        return
    fs = float(diag["fusion_scale"])
    alphas.extend(a.tolist() + [fs] for a in diag["alpha"].float().cpu())


def no_overlap_inference(model, mix, device, batch_size, inf_ck):
    true_samples = inf_ck - 2 * model.overlap

    right_pad = true_samples + model.overlap - ((mix.shape[-1]) % true_samples)
    mixture = np.concatenate((np.zeros((model.audio_ch, model.overlap), dtype='float32'),
                              mix,
                              np.zeros((model.audio_ch, right_pad), dtype='float32')),
                             1)
    num_chunks = mixture.shape[-1] // true_samples
    mix_waves_batched = [mixture[:, i * true_samples: i * true_samples + inf_ck] for i in
                         range(num_chunks)]
    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size)

    target_wav_hats = []
    alphas = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_specs = model.multi_stft(mixture_wav.to(device))
            spec_hat = model(mix_specs)
            _collect_alphas(model, alphas)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat) # (b, c, t)

        target_wav_hat = np.vstack(target_wav_hats)[:, :, model.overlap:-model.overlap] # (sum(b), c, t)
        target_wav_hat = np.concatenate(target_wav_hat, axis=-1)[:, :mix.shape[-1]]
    return target_wav_hat, alphas


def overlap_inference(model, mix, device, batch_size, inf_ck, overlap_rate, tmp_root, samplerate):
    '''
    Args:
        mix: (c, t)
    '''
    hop_length = math.ceil((1 - overlap_rate) * inf_ck)
    overlap_size = inf_ck - hop_length
    step_t = mix.shape[1]
    mix_waves_batched = split_nparray_with_overlap(mix.T, hop_length, overlap_size)

    mix_waves_batched = torch.tensor(mix_waves_batched, dtype=torch.float32).split(batch_size) # [(b, c, t)]

    target_wav_hats = []
    alphas = []

    with torch.no_grad():
        model.eval()
        for mixture_wav in mix_waves_batched:
            mix_specs = model.multi_stft(mixture_wav.to(device))
            spec_hat = model(mix_specs)
            _collect_alphas(model, alphas)
            target_wav_hat = model.istft(spec_hat)
            target_wav_hat = target_wav_hat.cpu().detach().numpy()
            target_wav_hats.append(target_wav_hat) # (b, c, t)

        target_wav_hat = np.vstack(target_wav_hats) # (sum(b), c, t)
        target_wav_hat = np.transpose(target_wav_hat, (0, 2, 1)) # (sum(b), t, c)
        target_wav_hat = join_chunks(tmp_root, target_wav_hat, samplerate, overlap_size) # (t, c)
    return target_wav_hat[:step_t].T, alphas # (c, t)
