"""Duration predictor built from dense-matrix sequence mixers.

See ``modules/duration/attention.py`` for why every learned parameter of this
stack is a 2-D matrix or a 1-D gain, and ``modules/duration/word_groups.py`` for
the word structure it predicts over.

Structure
---------
``x -> in_proj -> optional within-word positions -> N x pre-norm sliding-window
block -> out_norm -> out_proj`` followed by one of two output tails:

* ``use_allocation=True`` (the default of the shipped configurations) reads the
  word structure of the score: the phonemes of a word are turned into a
  distribution over that word's frame budget, and inference returns the integer
  frame counts of the split, which sum to the budget exactly.
* ``use_allocation=False`` returns absolute phoneme durations in the linear
  domain, exactly like the convolutional predictors do.

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
            position embeddings (requires ``ph2word`` at every call site).
        max_pos (int, optional): Clamp for those position embeddings.
        use_allocation (bool, optional): Split the frame budget of every word
            instead of predicting absolute durations.
        offset (float, optional): Offset of the log-domain output, used when
            ``use_allocation`` is off.
        loss_type (str, optional): Underlying loss type, used when
            ``use_allocation`` is off.
    """

    def __init__(self, in_dims, hidden_size, num_blocks=4, num_heads=4, radius=8,
                 ffn_mult=4, ffn_act='gelu', dropout=0.0, use_pos_embed=True,
                 max_pos=8, use_allocation=True, offset=1.0, loss_type='mse'):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if use_pos_embed and max_pos < 0:
            raise ValueError(f"max_pos must be non-negative, got {max_pos}")
        if loss_type not in ['mse', 'huber']:
            raise NotImplementedError(loss_type)
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_allocation = use_allocation
        self.use_pos_embed = use_pos_embed
        self.max_pos = max_pos
        self.offset = offset
        self.loss_type = loss_type
        self.in_proj = nn.Linear(in_dims, hidden_size)
        self.blocks = nn.ModuleList(
            SlidingWindowBlock(
                hidden_size, num_heads, radius, ffn_mult, ffn_act, dropout
            )
            for _ in range(num_blocks)
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        # A plain nn.Linear on purpose: the output projection is a matrix, so it
        # belongs to the matrix parameter group together with the rest of the
        # stack, unlike the AdamWLinear used by the convolutional predictors.
        # The bias is kept because it shifts the absolute durations when
        # ``use_allocation`` is off. With allocation on it is inert, as the split
        # of a word budget only depends on the differences between the scores of
        # that word, which the softmax already removes.
        self.out_proj = nn.Linear(hidden_size, 1)
        if use_pos_embed:
            self.pos_embed = nn.Embedding(max_pos + 1, hidden_size)
            self.reverse_pos_embed = nn.Embedding(max_pos + 1, hidden_size)
        else:
            self.pos_embed = None
            self.reverse_pos_embed = None

    @property
    def needs_word_div(self) -> bool:
        """Whether the exported graph has to rebuild the word ids from ``word_div``."""
        return self.use_allocation or self.use_pos_embed

    @classmethod
    def from_hparams(cls, in_dims, dur_hparams: dict) -> 'DurationPredictorV2':
        """Build from the flat ``dur_prediction_args`` configuration block.

        Every argument that the convolutional predictors do not share is read
        with a default, so that a configuration written before this class existed
        keeps building.
        """
        return cls(
            in_dims=in_dims,
            hidden_size=dur_hparams['hidden_size'],
            num_blocks=dur_hparams.get('dur_num_blocks', 4),
            num_heads=dur_hparams.get('dur_num_heads', 4),
            radius=dur_hparams.get('dur_radius', 8),
            ffn_mult=dur_hparams.get('dur_ffn_mult', 4),
            ffn_act=dur_hparams.get('dur_ffn_act', 'gelu'),
            dropout=dur_hparams['dropout'],
            use_pos_embed=dur_hparams.get('dur_pos_in_group', True),
            max_pos=dur_hparams.get('dur_max_group_pos', 8),
            use_allocation=dur_hparams.get('use_allocation', False),
            offset=dur_hparams['log_offset'],
            loss_type=dur_hparams['loss_type'],
        )

    def out2dur(self, xs):
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
            # The allocation below is a discrete, order-preserving computation, so
            # it is always run in float32, also under mixed-precision training,
            # where the shares would otherwise carry reduced-precision noise into
            # the frame counts.
            logits = logits.float()
            prob = word_log_softmax(logits, members).exp() * non_pad_mask
            # `word_budget` arrives per word, one entry per word of the utterance,
            # and is expanded here. Passing it already expanded per phoneme would
            # index it with its own length and silently collapse every word onto
            # the budget of the first one, so it must stay [B, T_w].
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
