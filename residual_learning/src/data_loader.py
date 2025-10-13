#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data_loader.py

This module is responsible for all data loading and pre-processing tasks.
It defines the PyTorch Dataset class and the main function for building
session-aware training samples from a DataFrame.
"""

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .simulation import simulate_step
from .config import K_DELTA
from .utils import wrap_to_pi

# ----------------------- DATASET HELPERS --------------------------

class WindowDataset(Dataset):
    """
    A simple PyTorch Dataset that serves windowed data samples.
    """
    def __init__(self, X, Y):
        # Expects X and Y to be PyTorch Tensors
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        # Returns the sample, the target, and the original index
        return self.X[idx], self.Y[idx], idx

# ----------------------- BUILD DATA (SESSION-AWARE) -----------------------

def build_data(df_all_sessions: pd.DataFrame, H: int, dt: float):
    """
    Builds windowed training samples (X, Y) from a complete DataFrame,
    ensuring that no sample window crosses the boundary between different
    recording sessions.

    Args:
        df_all_sessions (pd.DataFrame): The full dataframe containing data from all
                                        sessions. Must include a 'session_id' column.
        H (int): The history window size.
        dt (float): The time step between data points.

    Returns:
        A tuple of (torch.Tensor, torch.Tensor) for features X and labels Y.
    """
    # Group the dataframe by the session_id to treat each recording separately
    grouped = df_all_sessions.groupby('session_id')
    
    all_X = []
    all_Y = []

    print("Building session-aware training data...")
    # Iterate over each session (each original rosbag) with a progress bar
    for session_id, session_df in tqdm(grouped, desc="Processing Sessions"):
        
        # Ensure the session is indexed from 0 for easy lookup
        session_df = session_df.reset_index(drop=True)
        
        # Convert the session's DataFrame into a list of dictionaries for the simulator
        meas, ctrl = [], []
        for k in range(len(session_df)):
            meas.append({
                'pos_x': session_df['pos_x'].iloc[k],
                'pos_y': session_df['pos_y'].iloc[k],
                'yaw':   session_df['yaw'].iloc[k],
                'speed': session_df['speed'].iloc[k],
                'a_x':   session_df['acceleration'].iloc[k],
                # Get the previous step's steering command for the lag model
                'delta_prev': np.deg2rad(session_df['steer_deg'].iloc[k-1] if k > 0 else session_df['steer_deg'].iloc[0]) * K_DELTA
            })
            ctrl.append({
                'acc': session_df['acceleration'].iloc[k],
                'delta_cmd': np.deg2rad(session_df['steer_deg'].iloc[k]) * K_DELTA
            })

        N_session = len(session_df)
        # Skip this session if it's too short to create even one training sample
        if N_session <= H:
            continue

        # Calculate the number of samples that can be created from this session
        num_samples_in_session = N_session - H
        
        X_session = np.zeros((num_samples_in_session, H, 6), dtype=np.float32)
        Y_session = np.zeros((num_samples_in_session, 1), dtype=np.float32)

        # Create each sample one by one
        for i in range(num_samples_in_session):
            # Create the history window for the features X
            for j in range(H):
                k = i + j
                X_session[i, j] = [
                    meas[k]['a_x'], meas[k]['speed'],
                    math.sin(meas[k]['yaw']), math.cos(meas[k]['yaw']),
                    ctrl[k]['acc'], ctrl[k]['delta_cmd']
                ]
            
            # --- Calculate the target residual Y ---
            # 1. Get the state at the end of the history window
            state0 = meas[i + H - 1].copy()
            control0 = ctrl[i + H - 1].copy()
            
            # 2. Use the kinematic model to predict the yaw rate for the next step
            next_state_base = simulate_step(state0, control0, dt)
            predicted_rz = next_state_base['r_z']

            # 3. Get the TRUE yaw rate from the real data at the next step
            # We must calculate it from the yaw angles to be precise
            true_next_yaw = meas[i + H]['yaw']
            current_yaw = meas[i + H - 1]['yaw']
            true_rz = wrap_to_pi(true_next_yaw - current_yaw) / dt

            # 4. The label is the difference (the residual)
            Y_session[i, 0] = true_rz - predicted_rz

        # Add the samples from this session to our master lists
        all_X.append(X_session)
        all_Y.append(Y_session)

    if not all_X:
        raise ValueError("No valid training samples could be created from the provided data. Check history window size (`--hist`) and the length of individual sessions.")

    # --- Concatenate the results from all sessions into single large tensors ---
    final_X = torch.from_numpy(np.concatenate(all_X, axis=0))
    final_Y = torch.from_numpy(np.concatenate(all_Y, axis=0))
    
    print(f"Successfully built {final_X.shape[0]} total training samples from {len(grouped)} sessions.")
    
    return final_X, final_Y