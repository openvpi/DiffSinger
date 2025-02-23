import math
from math import sqrt

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams

@torch.jit.script
def RoPE(Q, K):
	# Split
	w = Q.size(-1) // 4
	Q1r, Q1i, Q2r, Q2i = torch.split(Q, [w, w, w, w], dim=-1)
	K1r, K1i, K2r, K2i = torch.split(K, [w, w, w, w], dim=-1)
	# Calc
	Pt = torch.arange(0, Q.size(-3), device=Q.device).reshape(1, -1, 1, 1)
	Df = (2 ** -torch.arange(0, w, device=Q.device)).reshape(1, 1, 1, -1)
	U = Pt * Df / 16
	Ur, Ui = torch.sin(U), torch.cos(U)
	# Mul
	Q1r, Q1i = (Q1r * Ur - Q1i * Ui, Q1r * Ui + Q1i * Ur)
	Q2r, Q2i = (Q2r * Ur - Q2i * Ui, Q2r * Ui + Q2i * Ur)
	K1r, K1i = (K1r * Ur - K1i * Ui, K1r * Ui + K1i * Ur)
	K2r, K2i = (K2r * Ur - K2i * Ui, K2r * Ui + K2i * Ur)
	return torch.cat([Q1r, Q1i], dim=-1), torch.cat([Q2r, Q2i], dim=-1), torch.cat([K1r, K1i], dim=-1), torch.cat([K2r, K2i], dim=-1)

class NeoformerMHA(nn.Module):
	def __init__(self, feat=512, heads=4, feat_cond=None):
		super().__init__()
		if feat_cond == None:
			feat_cond = feat
		self.head_feat = feat // heads
		self.heads = heads
		self.Q_emb = nn.Linear(feat, feat)
		self.K_emb = nn.Linear(feat, feat)
		self.V_emb = nn.Linear(feat_cond, feat)
		self.scale = math.sqrt(feat // 2)
		self.s2_invern = nn.Parameter(torch.rand(heads), requires_grad=True)
	def __forward(self, x, cond):
		x = x.transpose(-1, -2)
		Q = self.Q_emb(x).reshape(x.size(0), x.size(1), self.heads, self.head_feat) # (B, S, H, F)
		K = self.K_emb(x).reshape(x.size(0), x.size(1), self.heads, self.head_feat) # (B, S, H, F)
		V = self.V_emb(cond.transpose(-1, -2)).reshape(x.size(0), x.size(1), self.heads, self.head_feat) # (B, S, H, F)
		Q1, Q2, K1, K2 = RoPE(Q, K)
		Attn1 = F.softmax(torch.einsum("bqhf,bkhf->bhqk", Q1, K1) / self.scale, dim=-1)
		Attn2 = F.softmax(torch.einsum("bqhf,bkhf->bhqk", Q2, K2) / self.scale, dim=-1)
		Attn = Attn1 - torch.einsum("bhqk,h->bhqk", Attn2, self.s2_invern)
		R = torch.einsum("bhqk,bkhf->bqhf", Attn, V)
		y = R.flatten(start_dim=-2, end_dim=-1) # (B, S, H, F) -> (B, S, F)
		return (x + y).transpose(-1, -2)
	def forward(self, x, cond=None):
		if cond == None:
			cond = x
		if self.training:
			return cp.checkpoint(self.__forward, x, cond)
		else:
			return self.__forward(x, cond)

class NeoformerMLP(nn.Module):
	def __init__(self, feat=512, kernel_size=5):
		super().__init__()
		self.mainer = nn.Sequential(
			nn.Conv1d(feat, feat * 2, 1),
			nn.SiLU(),
			nn.Conv1d(feat * 2, feat * 2, kernel_size, groups=feat * 2, padding="same")
		)
		self.gate = nn.Sequential(
			nn.Conv1d(feat, feat * 2, 1),
			nn.SiLU()
		)
		self.deposer = nn.Conv1d(feat * 2, feat, 1)
	def forward(self, x):
		return x + self.deposer(self.mainer(x) * self.gate(x))

class NeoformerCell(nn.Module):
	def __init__(self, feat=512, feat_cond=256, heads=4, kernel_size=5, is_first=False):
		super().__init__()
		if is_first:
			self.norm1 = nn.Identity()
		else:
			self.norm1 = nn.LayerNorm(feat)
		self.self_attn = NeoformerMHA(feat, heads)
		self.mlp1 = NeoformerMLP(feat, kernel_size)
		self.norm2 = nn.LayerNorm(feat)
		self.cross_attn = NeoformerMHA(feat, heads, feat_cond)
		self.mlp2 = NeoformerMLP(feat, kernel_size)
	def forward(self, x, cond):
		x = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
		x = self.self_attn(x)
		x = self.mlp1(x)
		x = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
		x = self.cross_attn(x, cond)
		x = self.mlp2(x)
		return x

class NeoformerBackbone(nn.Module):
	def __init__(self, feat=512, layers=5, feat_cond=256, kernel_size=21, cond_latent=1024, heads=4):
		super().__init__()
		self.cond_reposer = nn.Sequential(
			nn.Conv1d(feat_cond, cond_latent, 1),
			nn.SiLU()
		)
		self.cells = nn.ModuleList([ NeoformerCell(feat, cond_latent, heads, kernel_size, i == 0) for i in range(0, layers) ])
	def forward(self, x, cond):
		cond = self.cond_reposer(cond)
		for cell in self.cells:
			x = cell(x, cond)
		return x

class DiffsingerNeoformer(nn.Module):
	def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, num_heads=4, kernel_size=5, cond_latent=1024):
		super().__init__()
		self.in_dims = in_dims
		self.n_feats = n_feats
		self.input_projection = nn.Conv1d(in_dims * n_feats, num_channels, 1)
		self.diffusion_embedding = nn.Sequential(
			SinusoidalPosEmb(num_channels),
			nn.Linear(num_channels, num_channels * 4),
			nn.GELU(),
			nn.Linear(num_channels * 4, hparams['hidden_size']),
		)
		self.backbone = NeoformerBackbone(
			feat=num_channels,
			layers=num_layers,
			feat_cond=hparams['hidden_size'],
			kernel_size=kernel_size,
			cond_latent=cond_latent,
			heads=num_heads
		)
		self.output_projection = nn.Conv1d(num_channels, in_dims * n_feats, 1)
	def forward(self, x, n_step, cond):
		x = x.reshape(x.size(0), self.in_dims * self.n_feats, x.size(-1))
		x = self.input_projection(x)
		cond = cond + self.diffusion_embedding(n_step).unsqueeze(-1)
		x = self.backbone(x, cond)
		x = F.silu(x)
		x = self.output_projection(x)
		x = x.reshape(x.size(0), self.n_feats, self.in_dims, x.size(-1))
		return x
