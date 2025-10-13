#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/simulation.py

This module contains all functions related to the kinematic bicycle model
simulation. It includes both single-instance and batched (PyTorch) versions
of the simulation step, as well as the high-level evaluation function that
runs a full closed-loop simulation to assess model performance.
"""

import math
import numpy as np
import torch
from .config import L, K_DELTA, TAU_DELTA, CD
from .utils import wrap_to_pi

# ----------------------- SIMULATION HELPERS --------------------------

def simulate_step(state: dict, control: dict, dt: float) -> dict:
    """
    Single-step kinematic bicycle update (for data building and evaluation).
    Operates on single data points (dictionaries) on the CPU.
    """
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev'])
    x     = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y     = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt
    psi   = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v     = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt
    r_z   = (psi - state['yaw']) / dt
    a_x   = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}


def simulate_step_torch(states: dict, controls: dict, dt: float) -> dict:
    """
    Vectorized, PyTorch-based version of simulate_step for batched processing
    during the training rollout loss calculation.
    """
    delta = states['delta_prev'] + (dt / TAU_DELTA) * (controls['delta_cmd'] - states['delta_prev'])
    x     = states['pos_x'] + states['speed'] * torch.cos(states['yaw']) * dt
    y     = states['pos_y'] + states['speed'] * torch.sin(states['yaw']) * dt
    unwrapped_psi = states['yaw'] + (states['speed'] / L) * torch.tan(delta) * dt
    psi   = torch.atan2(torch.sin(unwrapped_psi), torch.cos(unwrapped_psi))
    v     = states['speed'] + (controls['acc'] - CD * states['speed']**2) * dt
    r_z   = (psi - states['yaw']) / dt
    a_x   = (v - states['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}


def feat_tensor(state: dict, control: dict, device) -> torch.Tensor:
    """
    Creates a feature tensor for a single time step (non-batched).
    Used during the validation simulation.
    """
    return torch.tensor([
        state['a_x'], state['speed'],
        math.sin(state['yaw']), math.cos(state['yaw']),
        control['acc'], control['delta_cmd']
    ], dtype=torch.float32, device=device)


def feat_tensor_torch(states: dict, controls: dict) -> torch.Tensor:
    """
    Creates a feature tensor for batched inputs.
    Used during the training rollout loss calculation.
    """
    return torch.stack([
        states['a_x'], states['speed'],
        torch.sin(states['yaw']), torch.cos(states['yaw']),
        controls['acc'], controls['delta_cmd']
    ], dim=1)


# ----------------------- SIMULATION-BASED EVALUATION -----------------------

def evaluate_simulation_performance(model: torch.nn.Module, val_meas: list, val_ctrl: list, H: int, dt: float, device: torch.device) -> float:
    """
    Runs a full closed-loop simulation on a validation trajectory and returns
    the final positional error. This is the key metric for judging model quality.

    Args:
        model: The trained model to evaluate.
        val_meas: A list of measurement dictionaries for the validation trajectory.
        val_ctrl: A list of control dictionaries for the validation trajectory.
        H (int): The history window size.
        dt (float): The time step.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        The final positional error in meters.
    """
    model.eval()
    N = len(val_meas)
    sim_states = []
    state = val_meas[0].copy() # Start from the first true state of the validation trajectory
    history_features = torch.zeros((H, 6), device=device)

    with torch.no_grad():
        for k in range(N):
            sim_states.append(state.copy())
            if k >= N - 1: continue

            # Create feature vector from the current simulated state
            current_features = feat_tensor(state, val_ctrl[k], device)
            # Update history buffer
            history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)
            
            rz_residual = 0.0
            # Only start predicting after the history buffer is full
            if k >= H:
                # The model predicts the residual
                rz_residual = model(history_features.unsqueeze(0)).item()
            
            # Get the next state from the base kinematic model
            next_state_base = simulate_step(state, val_ctrl[k], dt)
            
            # Apply the learned residual correction to the yaw rate
            corrected_rz = next_state_base['r_z'] + rz_residual
            next_yaw_corr = wrap_to_pi(state['yaw'] + corrected_rz * dt)
            
            # The next state is a combination of the base model and the correction
            state = next_state_base.copy()
            state['yaw'] = next_yaw_corr
            state['r_z'] = corrected_rz

    # Compare the final position of the simulation to the ground truth
    real_pos_x = np.array([s['pos_x'] for s in val_meas])
    real_pos_y = np.array([s['pos_y'] for s in val_meas])
    sim_pos_x = np.array([s['pos_x'] for s in sim_states])
    sim_pos_y = np.array([s['pos_y'] for s in sim_states])
    
    # Calculate final positional error
    pos_error = np.sqrt((real_pos_x[-1] - sim_pos_x[-1])**2 + (real_pos_y[-1] - sim_pos_y[-1])**2)
    return float(pos_error)