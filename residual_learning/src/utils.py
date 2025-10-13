#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils.py

This module contains general-purpose utility functions, such as math helpers
and plotting functions, used across the project.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def plot_curves(train_history: list, val_history: list, val_epochs: list, model_type: str, encoder_type: str):
    """
    Plots training loss and validation positional error on a twin-axis plot.

    Args:
        train_history: A list of the average training loss from each epoch.
        val_history: A list of the validation positional error from validation epochs.
        val_epochs: A list of the epoch numbers corresponding to the validation runs.
        model_type: The type of model (e.g., 'svgp').
        encoder_type: The type of encoder (e.g., 'transformer').
    """
    if not train_history or not val_history:
        print("Warning: History lists are empty. Skipping plot generation.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot training loss on the primary y-axis
    epochs_train = range(1, len(train_history) + 1)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average One-Step Training Loss', color=color)
    ax1.plot(epochs_train, train_history, color=color, marker='o', linestyle='-', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a second y-axis for the validation positional error
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Simulation: Final Positional Error [m]', color=color)
    ax2.plot(val_epochs, val_history, color=color, marker='s', linestyle='--', label='Validation Positional Error')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a marker for the best validation score
    if val_history:
        best_epoch_idx = np.argmin(val_history)
        best_epoch = val_epochs[best_epoch_idx]
        best_error = val_history[best_epoch_idx]
        ax2.plot(best_epoch, best_error, 'o', color='gold', markersize=15, markeredgecolor='black', label=f'Best Model (Error: {best_error:.3f}m)')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.suptitle(f'Training & Validation Performance for {model_type.upper()} with {encoder_type.upper()} Encoder', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = f'training_curve_{model_type}_{encoder_type}.png'
    plt.savefig(save_path)
    print(f"\nSaved training curve to {save_path}")
    plt.show()