"""
Copyright 2025 Maviss Design Co., Ltd. All rights reserved.
Modification, and/or distribution of this software is prohibited.
However you can make similar function on your responsibility.
"""

import matplotlib.pyplot as plt
import numpy as np

from samact.Results.Dynamics import Dynamics
from samact.Results.StaticParameters import StaticParameters


def ShowFiringRate(dynamics: list[Dynamics]):
    """
    Visualize firing rates of neurons for each layer.

    This function computes the firing rate of each neuron as the total
    number of spikes divided by the total number of time steps, and
    displays the result as vertically stacked heatmaps for each layer.

    Args:
        dynamics: List of dynamic simulation results for each layer.
    """
    layers: list[np.ndarray] = []
    labels: list[str] = []

    for value in dynamics:
        _, firingTimes = value.gainMap.shape
        layers.append(
            value.gainMap.sum(axis=1)[:, np.newaxis] / firingTimes
        )
        labels.append(value.name)

    fig, ax = plt.subplots(figsize=(6, 6))

    # 色スケール
    vmin = 0.0
    vmax = 1.0
    cmap = 'hot'

    # x方向に並べて表示
    x_start = 0
    x_gap = 1  # 各層の横幅

    for i, layer in enumerate(layers):
        h = layer.shape[0]  # ニューロン数（縦）
        extent = [x_start, x_start + x_gap, 0, h]
        ax.imshow(
            layer,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.text(
            x_start + 0.5,
            h + 2,
            labels[i],
            ha='center',
            va='bottom',
        )
        x_start += x_gap

    # 軸設定
    ax.set_xlim(0, x_start)
    ax.set_ylim(0, max(layer.shape[0] for layer in layers) + 10)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron Index')
    ax.set_xticks([])

    # カラーバー
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Firing Rate')

    plt.title('Firing Rate')
    plt.tight_layout()
    plt.show()


def ShowPotentials(
    dynamics: list[Dynamics],
    statics: list[StaticParameters],
):
    """
    Visualize internal membrane potentials relative to theta.

    This function subtracts theta from the internal potential values
    for each neuron and visualizes the result as a heatmap over time.

    Args:
        dynamics: List of dynamic simulation results.
        statics: List of corresponding static parameters.
    """
    for dynamic, static in zip(dynamics, statics):
        theta = static.theta
        pot = dynamic.potentialMap - theta[:, np.newaxis]

        plt.imshow(
            pot,
            vmin=-1_000_000_000,
            vmax=1_000_000_000,
            cmap='hot',
        )
        plt.colorbar(label='Value')
        plt.ylabel('Neuron')
        plt.xlabel('t')
        plt.title(
            f'Internal Potential of {dynamic.name}',
            pad=20,
        )
        plt.show()