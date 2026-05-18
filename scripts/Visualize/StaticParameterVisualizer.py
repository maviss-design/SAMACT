"""
Copyright 2025 Maviss Design Co., Ltd. All rights reserved.
Modification, and/or distribution of this software is prohibited.
However you can make similar function on your responsibility.
"""

import matplotlib.pyplot as plt
import numpy as np

from samact.Results.StaticParameters import StaticParameters


def ShowWeightMap(statics: list[StaticParameters]):
    """
    Visualize the ratio between weights and theta as a heatmap.

    This function computes |weight / theta| for each neuron–synapse pair
    and displays the result as a heatmap, where the color scale is clipped
    between the 25th and 75th percentiles to reduce the influence of outliers.

    Args:
        statics: List of static parameter snapshots for each layer.
    """
    for static in statics:
        ratio = np.abs(static.weight / static.theta[:, np.newaxis])
        q25 = np.percentile(ratio, 25)
        q75 = np.percentile(ratio, 75)

        # ヒートマップの描画
        plt.figure(figsize=(10, 8))
        plt.imshow(ratio, aspect='equal', cmap='hot', vmin=q25, vmax=q75)
        plt.colorbar(label='Ratio Weight/Theta')
        plt.xlabel('Synapse')
        plt.ylabel('Neuron')
        plt.title(f'Weight Ratio Heatmap of {static.name}', pad=20)
        plt.show()