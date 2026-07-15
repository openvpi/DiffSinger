# 外部噪声输入（seed / 可复现）导出改动说明

## 目的

把扩散/RF 采样的**初始噪声**从模型内部 `torch.randn` 改为**外部输入张量 `noise`**。
导出后的 ONNX 多一个输入口 `noise`；喂相同 `noise` → 输出逐元素一致（可复现 / 可序列化 / 跨机一致），
从而上层（插件）可用 seed 做真正的 retake，不再依赖张量缓存兜稳定性。

**向后兼容**：`noise` 在 `forward` 里默认 `None`，不传则保持原内部随机行为（训练 / eager 推理不受影响）。
本改动**不需要重训**——纯导出/推理路径改动。

## noise I/O 契约

| 模型 | 输入名 | 形状 `[1, num_feats, out_dims, n_frames]` | 说明 |
|---|---|---|---|
| acoustic | `noise` | `[1, 1, audio_num_mel_bins, T]`（如 `[1,1,128,T]`） | float32 |
| pitch | `noise` | `[1, 1, repeat_bins, T]`（如 `[1,1,64,T]`） | float32 |
| variance | `noise` | `[1, num_variances, repeat_bins, T]`（如 `[1,3,r,T]`） | float32 |

仅**帧轴（axis 3）动态**；axis 1/2 导出为定值，便于下游从 InputMetadata 直接读出通道形状。

## 改动清单（5 文件）

**采样器 `forward` 加 `noise=None` 入参，内部 `if noise is None: noise = torch.randn(...)`：**
- `deployment/modules/rectified_flow.py` — `RectifiedFlowONNX.forward`
- `deployment/modules/diffusion.py` — `GaussianDiffusionONNX.forward`
  （Pitch/MultiVariance 版继承此 forward，自动覆盖）

**toplevel 包装方法透传 `noise`（被 `view_as_*` 赋为 `model.forward` 后导出）：**
- `deployment/modules/toplevel.py`：`forward_shallow_diffusion` / `forward_diffusion` /
  `forward_shallow_reflow` / `forward_reflow` / `forward_pitch_reflow` / `forward_variance_reflow`

**导出器把 `noise` 声明为 ONNX 输入（example_inputs 末尾 + input_names + dynamic_axes）：**
- `deployment/exporters/acoustic_exporter.py` — 复用已有的 `noise` dummy（backbone trace 用的那个，形状已对）
- `deployment/exporters/variance_exporter.py` — pitch 段 + variance 段，同样复用已有 `noise` dummy

> 注：`noise` 是 `forward` 的**最后一个位置参数**，导出元组按位置追加即可；
> shallow（变深度）与非 shallow 两条 forward 签名不同，但都靠 `diffusion_inputs` 自适应，追加 `(steps, noise)` 对两者都正确。

## 朋友需要做的验证（有 ckpt + 训练环境）

1. 跑正常导出流程（`python deployment/scripts/export.py ...` 或你们的导出入口）。
2. 加载导出的 ONNX，确认输入里多了 `noise`、且 `RandomNormalLike` 已从图里消失（噪声改由输入提供）。
3. 喂固定 `noise` 跑两遍 → 输出逐元素一致即成功。
4. 若 TorchScript 在 `torch.jit.script` 处对 `noise` 报类型问题：本改动刻意**完全沿用现有 `x_end=None` 的无注解写法**
   （`x_end` 同样是 `=None` 默认 + `if x_end is None`，且已在 example_inputs 提供 Tensor），理论上等价；
   若仍有问题，把 `noise` 注解为 `Optional[torch.Tensor]` 并 `from typing import Optional` 即可。
5. merge 步骤：`noise` 是图中无人内部产出的输入，合并后应自动留在顶层。确认未被优化器消除。

## （可选 / 需重训）acoustic 的 note 级 retake

实测结论：**acoustic 噪声重摇在时间上不局部化**（卷积感受野 ~1s，单步即近全局），
所以仅靠 `noise` 输入，acoustic 只能做 **part 级整体重摇**；pitch/variance 因自带 `retake` 口可到 note/帧级。

`retake` 是**训练能力**（`training/variance_task.py` 的 `random_retake_masks` + `toplevel.py` 里学习到的
`pitch_retake_embed` / `variance_retake_scaling`）。要让 acoustic 也支持 note 级 retake，需把这套机制**移植到 acoustic 并重训**：

1. acoustic 模型加一个 `mel_retake_embed`（仿 `pitch_retake_embed`），把 retake 状态作为条件加入。
2. 训练时用 `random_retake_masks` 随机遮罩，retake=false 帧喂入已知 mel、并令模型重建之。
3. 导出时给 acoustic 增加 `retake`（bool, `[1,T]`）+ 已知 mel 输入。

这是一次真正的训练改造（非导出改动），成本/质量风险都高于 variance 版；
建议先用 part 级 Timbre 跑起来，确有需要再投入。
