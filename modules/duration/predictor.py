"""Duration predictor built from sliding-window attention blocks.

``x -> in_proj -> optional within-word positions -> N x pre-norm sliding-window
blocks -> out_norm -> out_proj``, followed by the allocation tail: the phonemes
of a word are turned into a distribution over that word's frame budget, and
inference returns the integer frame counts of the split, which sum to the budget
exactly. With ``use_allocation`` off the stack predicts absolute durations in
the linear domain instead.

The exported graph stays batch size 1 as in the other duration/variance graphs,
but every operation is batch-generic, so training runs on batches.
"""

import torch
import torch.nn as nn

from modules.duration.attention import SlidingWindowBlock
from modules.duration.word_groups import (
    allocate_word_frames,
    word_log_softmax,
    word_membership,
    word_position_ids,
)

__all__ = ["DurationPredictorV2"]


class DurationPredictorV2(nn.Module):
    """Sliding-window attention duration predictor with an optional word allocator.

    Args:
        in_dims (int): Input dimension, i.e. the width of the conditioning.
        hidden_size (int): Internal width of the attention stack.
        num_blocks (int, optional): Number of pre-norm attention blocks.
        num_heads (int, optional): Number of attention heads.
        radius (int, optional): Attention window radius in items.
        ffn_mult (int, optional): Feed-forward expansion factor.
        ffn_act (str, optional): Feed-forward activation.
        dropout (float, optional): Dropout rate of the blocks.
        use_pos_embed (bool, optional): Add the forward/reverse within-word
            position embeddings (requires ``ph2word`` at every call site). The
            embeddings saturate at ``radius``, the reach of the attention.
        use_allocation (bool, optional): Split the frame budget of every word
            instead of predicting absolute durations.
        offset (float, optional): Offset of the log-domain output, used when
            ``use_allocation`` is off.
        loss_type (str, optional): Underlying loss type, used when
            ``use_allocation`` is off.
    """

    def __init__(self, in_dims, hidden_size, num_blocks=4, num_heads=4, radius=8,
                 ffn_mult=4, ffn_act='gelu', dropout=0.0, use_pos_embed=True,
                 use_allocation=True, offset=1.0, loss_type='mse'):
        """Initialize the module; the class docstring documents every argument."""
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if loss_type not in ['mse', 'huber']:
            raise NotImplementedError(loss_type)
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_allocation = use_allocation
        self.use_pos_embed = use_pos_embed
        self.max_pos = radius
        self.offset = offset
        self.loss_type = loss_type
        # Declared here so that consumers never have to know which architecture
        # they are holding.
        self.needs_word_div = use_allocation or use_pos_embed
        self.needs_word_dur = use_allocation
        self.in_proj = nn.Linear(in_dims, hidden_size)
        self.blocks = nn.ModuleList(
            SlidingWindowBlock(
                hidden_size, num_heads, radius, ffn_mult, ffn_act, dropout
            )
            for _ in range(num_blocks)
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        # A plain nn.Linear on purpose: the output projection is a matrix and so
        # belongs to the matrix parameter group like the rest of the stack,
        # unlike the AdamWLinear used by the convolutional predictors.
        self.out_proj = nn.Linear(hidden_size, 1)
        if use_pos_embed:
            self.pos_embed = nn.Embedding(self.max_pos + 1, hidden_size)
            self.reverse_pos_embed = nn.Embedding(self.max_pos + 1, hidden_size)
        else:
            self.pos_embed = None
            self.reverse_pos_embed = None

    @classmethod
    def from_hparams(cls, in_dims, dur_hparams: dict) -> 'DurationPredictorV2':
        """Build from the flat ``dur_prediction_args`` configuration block.

        Only the stack shape is read from the configuration: naming ``arch:
        'attn'`` is what selects the allocation output and the within-word
        positions, so neither of them is a separate key.

        Args:
            in_dims (int): Input dimension, i.e. the width of the conditioning.
            dur_hparams (dict): The ``dur_prediction_args`` block of the configuration.

        Returns:
            DurationPredictorV2: The configured module.
        """
        radius = dur_hparams.get('dur_radius', 8)
        return cls(
            in_dims=in_dims,
            hidden_size=dur_hparams['hidden_size'],
            num_blocks=dur_hparams.get('dur_num_blocks', 4),
            num_heads=dur_hparams.get('dur_num_heads', 4),
            radius=radius,
            ffn_mult=dur_hparams.get('dur_ffn_mult', 4),
            dropout=dur_hparams['dropout'],
            offset=dur_hparams['log_offset'],
            loss_type=dur_hparams['loss_type'],
        )

    def out2dur(self, xs):
        """Convert the log-domain stack output (B, Tmax, 1) into a duration (B, Tmax)."""
        if self.loss_type in ['mse', 'huber']:
            # NOTE: calculate loss in log domain
            dur = xs.squeeze(-1).exp() - self.offset  # (B, Tmax)
        else:
            raise NotImplementedError(self.loss_type)
        return dur

    def forward(self, xs, x_masks=None, infer=True, ph2word=None, word_budget=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (BoolTensor, optional): Batch of masks indicating padded part (B, Tmax).
            infer (bool): Whether inference
            ph2word (Tensor, optional): Word index per phoneme [B, Tmax], 0 for padding.
                Required when the within-word positions or the allocation output are used.
            word_budget (Tensor, optional): Frame budget of every word [B, T_w], read at
                the word each phoneme belongs to. Required when the allocation output
                is used.
        Returns:
            Tensor: Predicted durations in the linear domain [B, Tmax]. With the allocation
            output enabled these are the number of frames per phoneme, integer in inference.
        """
        if x_masks is None:
            non_pad_mask = torch.ones(xs.shape[:2], dtype=torch.bool, device=xs.device)
        else:
            non_pad_mask = ~x_masks.bool()
        needs_words = self.use_allocation or self.use_pos_embed
        if needs_words and ph2word is None:
            raise ValueError(
                'the duration predictor needs the word ids of the phonemes; pass '
                'ph2word (training) or pass word_div to rebuild them (inference and export)'
            )
        if self.use_allocation and word_budget is None:
            raise ValueError(
                'the allocation output needs the frame budget of every word; pass a '
                '[B, T_w] word_budget (training) or word_dur (inference and export)'
            )
        members = word_membership(ph2word, x_masks=x_masks) if needs_words else None

        hidden = self.in_proj(xs)
        if self.pos_embed is not None:
            forward_pos, reverse_pos = word_position_ids(members, self.max_pos)
            hidden = hidden + self.pos_embed(forward_pos) + self.reverse_pos_embed(reverse_pos)
        hidden = hidden * non_pad_mask[:, :, None]
        for block in self.blocks:
            hidden = block(hidden, non_pad_mask)
        logits = self.out_proj(self.out_norm(hidden)).squeeze(-1)  # [B, Tmax]
        logits = logits * non_pad_mask

        if self.use_allocation:
            # The allocation is discrete and the frame counts have to be exact, so
            # it always runs in float32, also under mixed precision.
            logits = logits.float()
            prob = word_log_softmax(logits, members).exp() * non_pad_mask
            # The budget arrives with one entry per word and must stay [B, T_w]:
            # expanded per phoneme it would be indexed with its own length and
            # collapse every word onto the budget of the first one.
            budget = word_budget.gather(1, ph2word.clamp(min=1) - 1)
            if infer:
                dur_pred = allocate_word_frames(prob, budget, ph2word, x_masks=x_masks)
            else:
                # differentiable allocation: the shares scaled by the frame budget
                dur_pred = prob * budget
        else:
            dur_pred = self.out2dur(logits.unsqueeze(-1))
            if infer:
                dur_pred = dur_pred.clamp(min=0.)  # avoid negative value
        return dur_pred
