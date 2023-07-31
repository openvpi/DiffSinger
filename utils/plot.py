import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors='r', label='pred')
    plt.vlines(dur_gt, 0, 10, colors='b', label='gt')
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i-1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.text((dur_gt[i-1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color='black', linewidth=2, linestyle=':')
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    fig.legend()
    fig.tight_layout()
    return fig


def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None):
    if isinstance(curve_gt, torch.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, torch.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, torch.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure()
    if curve_base is not None:
        plt.plot(curve_base, color='g', label='base')
    plt.plot(curve_gt, color='b', label='gt')
    if curve_pred is not None:
        plt.plot(curve_pred, color='r', label='pred')
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig
