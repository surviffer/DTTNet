import os
import subprocess
from abc import ABCMeta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils.utils import sdr, simplified_msseval


class AbstractModel(LightningModule):
	__metaclass__ = ABCMeta

	def __init__(self, target_name,
				 lr, optimizer,
				  dim_f, dim_t, n_fft, n_fft_short, n_fft_long, hop_length, overlap,
				 audio_ch,
				 weight_decay=0.01,
				  **kwargs):
		super().__init__()
		self.target_name = target_name	#当前模型要分离的目标源名字 如vocal、drums等
		self.lr = lr	#学习率learning rate
		self.optimizer = optimizer	#优化器种类 如Adam
		self.weight_decay = weight_decay	#AdamW 权重衰减，显式配置
		self.seed = None	#由 src/train.py 写入，保存到 checkpoint 元数据
		self.dim_c_in = audio_ch * 2	#输入通道数（*2是因为短时傅里叶之后包含实部和虚部：左声道实部&左声道虚部&右声道实部&右声道虚部）
		self.dim_c_out = audio_ch * 2	#输出通道数
		self.dim_f = dim_f	#模型真正保留的频率维大小（通常小于n_bins）
		self.dim_t = dim_t	#时间维长度  单位：帧

		self.n_fft_mid = n_fft	#中窗长
		self.n_fft_short = n_fft_short	#短窗长
		self.n_fft_long = n_fft_long	#长窗长
		self.n_fft = self.n_fft_mid  # STFT窗长（每个STFT窗口的采样点数量）

		self.n_bins_mid = self.n_fft_mid // 2 + 1
		self.n_bins_short = self.n_fft_short // 2 + 1
		self.n_bins_long = self.n_fft_long // 2 + 1
		self.n_bins = self.n_bins_mid  # 有效的频率点个数（奈奎斯特频率） stft之后的max频率

		self.hop_length = hop_length	#滑窗每次往前走多少（两帧之间采样点数量）
		self.audio_ch = audio_ch	#声道数

		#chunk：一段连续的音频片段，它是模型实际处理的输入单元。（切割原信号为很多小片）
		self.chunk_size = hop_length * (self.dim_t - 1)	#训练的chunk，单次数据量。多个chunk组成一个batch
		self.inference_chunk_size = hop_length * (self.dim_t*2 - 1)	#比训练的chunk更长，希望看到更多一点上下文
		self.overlap = overlap	# 推理时分块之间的重叠长度

		# 固定Hann窗，给STFT/ISTFT用，不参与梯度更新（requires_grad=False）
		#self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
		self.window_mid = nn.Parameter(torch.hann_window(window_length=self.n_fft_mid, periodic=True),requires_grad=False)
		self.window_short = nn.Parameter(torch.hann_window(window_length=self.n_fft_short, periodic=True),requires_grad=False)
		self.window_long = nn.Parameter(torch.hann_window(window_length=self.n_fft_long, periodic=True),requires_grad=False)
		self.window = self.window_mid

		self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)
		#固定的0张量，istft时把模型输出的裁剪后频谱补回到完整频谱纬度（dim_f->n_bins）。因为模型输出只预测中窗域前 dim_f 个频率 bin，所以 istft() 前需要补回完整的中窗频率维
		self.inference_chunk_shape = (self.stft(torch.zeros([1, audio_ch, self.inference_chunk_size]))).shape
		#初始化时，先拿一个全0的假输入过一次stft，把推理chunk对应的频谱shape存下来，提前知道推理时一块音频进频谱后长什么样。


	def configure_optimizers(self):
		# 权重衰减只作用于卷积/线性层权重；bias、BN、static_logits、fusion_scale 不衰减
		# （static_logits 被衰减会把权重拉回 1/3，使 Learned static 偏向 Fixed average）
		decay, no_decay = [], []
		for name, p in self.named_parameters():
			if not p.requires_grad:	# 窗函数、freq_pad
				continue
			if p.ndim <= 1 or name.endswith(("static_logits", "fusion_scale")):
				no_decay.append(p)
			else:
				decay.append(p)
		groups = [
			{"params": decay, "weight_decay": self.weight_decay},
			{"params": no_decay, "weight_decay": 0.0},
		]
		# 根据配置决定训练时用哪种优化器
		if self.optimizer == 'rmsprop':
			print("Using RMSprop optimizer")
			return torch.optim.RMSprop(groups, self.lr)
		elif self.optimizer == 'adamW':
			print("Using AdamW optimizer")
			return torch.optim.AdamW(groups, self.lr)
		raise ValueError(f"Unknown optimizer {self.optimizer}")

	def comp_loss(self, pred_detail, target_wave):
		#把模型输出从频谱域转回波形域，再计算 L1 损失。
		pred_detail = self.istft(pred_detail)	# 模型输出的是频谱，先ISTFT回波形

		comp_loss = F.l1_loss(pred_detail, target_wave)		# 用波形域的 L1 loss 做训练目标

		self.log("train/comp_loss", comp_loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)	#把损失记录到日志

		return comp_loss


	def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
		#定义训练时一个batch怎么跑。*args, **kwargs 用于接收框架自动传入的参数。
		# 通常 Lightning 调用时会传入 (batch, batch_idx)。这里通过 args[0] 获取第一个参数，即 batch。
		mix_wave, target_wave = args[0] # (batch, c, 261120)
		# input 1
		mix_specs = self.multi_stft(mix_wave)
		# forward
		t_est_stft = self(mix_specs) # (batch, c, 1044, 256)
		if not torch.isfinite(t_est_stft).all():
			raise FloatingPointError(f"non-finite model output at step {self.global_step}")

		loss = self.comp_loss(t_est_stft, target_wave)
		if not torch.isfinite(loss):
			raise FloatingPointError(f"non-finite loss at step {self.global_step}")

		self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
		self._log_fusion("train")

		return {"loss": loss}

	def on_after_backward(self):
		# FP16 下 GradScaler 会主动跳过 inf 梯度的步，只记录次数；FP32/BF16 下直接报错
		bad = [name for name, p in self.named_parameters()
			   if p.grad is not None and not torch.isfinite(p.grad).all()]
		if not bad:
			return
		if self.trainer is not None and str(self.trainer.precision) in ("16", "16-mixed"):
			self.log("train/nonfinite_grad_steps", 1.0, on_step=False, on_epoch=True, reduce_fx="sum")
		else:
			raise FloatingPointError(f"non-finite grad in {bad[:5]} at step {self.global_step}")

	def fusion_diagnostics(self):
		# 子类（DPTDFNet）覆盖；baseline 返回 None
		return None

	def _log_fusion(self, stage):
		diag = self.fusion_diagnostics()
		if diag is None:
			return
		a = diag["alpha"].float().mean(0)
		for i, name in enumerate(("short", "mid", "long")):
			self.log(f"{stage}/alpha_{name}", a[i], sync_dist=True, on_step=False, on_epoch=True)
		self.log(f"{stage}/fusion_scale", diag["fusion_scale"].float(), sync_dist=True, on_step=False, on_epoch=True)

	def on_save_checkpoint(self, checkpoint):
		# 记录模型变体、STFT 参数、seed、git commit 和 Hydra 输出目录，推理时用于校验
		try:
			repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
		except Exception:
			commit = "unknown"
		checkpoint["drff_meta"] = {
			"fusion_mode": getattr(self, "fusion_mode", "baseline"),
			"target_name": self.target_name,
			"n_fft": [self.n_fft_short, self.n_fft_mid, self.n_fft_long],
			"dim_f": self.dim_f,
			"dim_t": self.dim_t,
			"hop_length": self.hop_length,
			"seed": self.seed,
			"git_commit": commit,
			"run_dir": os.getcwd(),	# Hydra 完整配置在 run_dir/.hydra/config.yaml
		}


	# Validation SDR is calculated on whole tracks and not chunks since
	# short inputs have high possibility of being silent (all-zero signal)
	# which leads to very low sdr values regardless of the model.
	# A natural procedure would be to split a track into chunk batches and
	# load them on multiple gpus, but aggregation was too difficult.
	# So instead we load one whole track on a single device (data_loader batch_size should always be 1)
	# and do all the batch splitting and aggregation on a single device.
	def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:	#验证集上按照整首歌的方式评估模型
		mix_chunk_batches, target = args[0]	#mix_chunk_batches是一个列表，每个其中的每个元素是一个batch

		# remove data_loader batch dimension
		# [(b, c, time)], (c, all_times)
		mix_chunk_batches, target = [batch[0] for batch in mix_chunk_batches], target[0]

		# process whole track in batches of chunks
		target_hat_chunks = []
		for batch in mix_chunk_batches:
			# input
			mix_specs = self.multi_stft(batch)  # (batch, c*2, 1044, 256)
			pred_detail = self(mix_specs) # (batch, c, 1044, 256), irm
			self._log_fusion("val")
			pred_detail = self.istft(pred_detail)	#得到每个chunk的波形

			target_hat_chunks.append(pred_detail[..., self.overlap:-self.overlap])	#减少chunk边界伪影，存入target_hat_chunks
		target_hat_chunks = torch.cat(target_hat_chunks) # (b*len(ls),c,t) 拼接（总块数，c，有效长度）

		# concat all output chunks (c, all_times)
		target_hat = target_hat_chunks.transpose(0, 1).reshape(self.audio_ch, -1)[..., :target.shape[-1]]	#交换前两维，后两维合并，截取与目标相同的长度

		ests = target_hat.detach().cpu().numpy()  # (c, all_times)
		references = target.cpu().numpy()
		#↑将估计波形和target转化为numpy数组
		score = sdr(ests, references)

		# (src, t, c)
		SDR = simplified_msseval(np.expand_dims(references.T, axis=0), np.expand_dims(ests.T, axis=0), chunk_size=44100)
		# self.log("val/sdr", score, sync_dist=True, on_step=False, on_epoch=True, logger=True)

		return {'song': score, 'chunk': SDR}

	def validation_epoch_end(self, outputs) -> None:	#把整轮验证里所有歌曲的结果汇总，得到最终验证指标。
		avg_uSDR = torch.Tensor([x['song'] for x in outputs]).mean()	#把每首歌的 song-level SDR 求平均
		self.log("val/usdr", avg_uSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

		chunks = [x['chunk'][0, :] for x in outputs]
		# concat np array
		chunks = np.concatenate(chunks, axis=0)
		median_cSDR = np.nanmedian(chunks.flatten(), axis=0)
		# 把所有 chunk 的 SDR 拼起来，取中位数cSDR
		median_cSDR = float(median_cSDR)
		self.log("val/csdr", median_cSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

	def _stft_impl(self, x, n_fft, window):
		"""
        通用STFT实现
        输入x: (B, C, T)
        输出(B, C*2, F, T_frames)
        """
		dim_b = x.shape[0]
		x = x.reshape([dim_b * self.audio_ch, -1])
		x = torch.stft(
			x,
			n_fft=n_fft,
			hop_length=self.hop_length,
			window=window,
			center=True,
			return_complex=True,
		)
		x = torch.view_as_real(x)
		x = x.permute([0, 3, 1, 2])
		x = x.reshape([dim_b, self.audio_ch, 2, x.shape[-2], -1]).reshape(
			[dim_b, self.audio_ch * 2, x.shape[-2], -1]
		)

		return x

	def stft(self, x):
		"""
        为了兼容原始代码，stft默认仍然表示中窗STFT
        输出频率维仍然裁到self.dim_f，作为主干输入域
        """
		x = self._stft_impl(x, self.n_fft_mid, self.window_mid)
		return x[:, :, :self.dim_f]

	def stft_short(self, x):
		"""
        短窗 STFT
        第一版不裁频率维，后面交给前端模块统一对齐
        """
		return self._stft_impl(x, self.n_fft_short, self.window_short)

	def stft_long(self, x):
		"""
        长窗 STFT
        第一版不裁频率维，后面交给前端模块统一对齐
        """
		return self._stft_impl(x, self.n_fft_long, self.window_long)

	def multi_stft(self, x):
		"""
        返回三路输入
        约定：
        - mid是主干输入域
        - short / long 作为辅助前端输入
        """
		specs = {"mid": self.stft(x)}	# 中窗，保留原始 DTT 输入域
		if getattr(self, "use_mr_frontend", False) and getattr(self, "fusion_mode", None) != "mid_only":
			specs["short"] = self.stft_short(x)	# 由前端按物理频率裁剪、对齐
			specs["long"] = self.stft_long(x)
		else:
			# baseline 和 mid_only 不需要短窗、长窗
			specs["short"] = specs["long"] = None
		return specs

	def istft(self, x):
		'''
		Args:
		x: (batch, c*2, 2048, 256)
		'''
		dim_b = x.shape[0]

		x = torch.cat(
		[x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])],
		-2
		)  # (batch, c*2, 3073, 256)

		x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape(
		[dim_b * self.audio_ch, 2, self.n_bins, -1]
		)  # (batch*c, 2, 3073, 256)

		x = x.permute([0, 2, 3, 1]).contiguous()  # (batch*c, 3073, 256, 2)
		x = torch.view_as_complex(x)  # (batch*c, 3073, 256)

		x = torch.istft(
		x,
		n_fft=self.n_fft,
		hop_length=self.hop_length,
		window=self.window,
		center=True,
		)  # (batch*c, 261120)

		return x.reshape([dim_b, self.audio_ch, -1])  # (batch, c, 261120)
