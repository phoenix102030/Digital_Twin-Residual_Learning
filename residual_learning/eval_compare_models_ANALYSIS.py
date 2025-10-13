#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multi_scenario.py

Performs a comprehensive, multi-scenario evaluation for multiple models.
This script is designed to generate all necessary results for RQ1 and RQ2.

Key Features:
- Iterates through all .csv files in a specified scenario directory.
- Generates per-scenario plots for detailed analysis (trajectory, error, etc.).
- Saves the raw time-series data for each model and scenario to a CSV file.
- Produces detailed and summary CSVs with final metrics.
- Performs statistical analysis (ANOVA) on the aggregated results.
- NEW: Captures parameter count and inference time for analysis.

Usage (example for RQ1):
  python eval_multi_scenario.py \
    --scenario_dir ./test_data/ \
    --output_dir ./evaluation_results/ \
    --ckpts ckpt_k1.pt ckpt_k5.pt ckpt_k10.pt \
    --labels "k=1" "k=5" "k=10" \
    --hist 10 \
    --plots
"""

from pathlib import Path
import argparse
import math
import time
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt

try:
    from scipy import stats as spstats
    SCIPY_OK = True
except ImportError:
    print("Warning: SciPy not found. Statistical analysis will not be available.")
    SCIPY_OK = False

# --- (Constants and Helper functions are unchanged) ---
L, K_DELTA, TAU_DELTA, CD = 1.0, 1.38, 0.028, 3.0e-5
def wrap_to_pi(angle: float) -> float: return math.atan2(math.sin(angle), math.cos(angle))
def simulate_step(state: Dict, control: Dict, dt: float) -> Dict:
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev'])
    x = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt
    psi = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt
    r_z = (psi - state['yaw']) / dt; a_x = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}
def compute_rz_diff(yaw: np.ndarray, dt: float) -> np.ndarray:
    yaw_unw = np.unwrap(yaw.astype(np.float64)); r = np.zeros_like(yaw_unw, dtype=np.float64)
    r[1:] = np.diff(yaw_unw) / dt; r[0] = r[1] if len(r) > 1 else 0
    return r.astype(np.float32)

# --- (Encoder and Model definitions are unchanged, ensure MobileViT is included) ---
class MobileViTEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(d_model), torch.nn.SiLU(),
            torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(d_model), torch.nn.SiLU())
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        x_conv = x.permute(0, 2, 1); local_features = self.conv_block(x_conv).permute(0, 2, 1)
        global_features = self.transformer_encoder(local_features); fused_features = global_features + local_features
        h = fused_features.mean(dim=1); return self.final_norm(self.outp(h))
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__(); self.proj = torch.nn.Linear(input_dim, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        h = self.proj(x); h = self.encoder(h); h = h.mean(dim=1); h = self.outp(h); return self.final_norm(h)
class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0.0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x): h,_ = self.lstm(x); return self.final_norm(h[:, -1, :])
class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0.0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x): h,_ = self.gru(x); return self.final_norm(h[:, -1, :])
def build_encoder(enc_type: str, d_model: int, args: dict) -> torch.nn.Module:
    enc_type = enc_type.lower()
    if enc_type in ['transformer', 'mobilevit']:
        encoder_args = {'nhead': args.get('tf_nhead', 4), 'd_hid': args.get('tf_d_hid', 128), 'nlayers': args.get('tf_nlayers', 2), 'dropout': args.get('tf_dropout', 0.1)}
        return TransformerEncoder(6, d_model, **encoder_args) if enc_type == 'transformer' else MobileViTEncoder(6, d_model, **encoder_args)
    elif enc_type in ['lstm', 'gru']:
        encoder_args = {'nlayers': args.get('rnn_nlayers', 2), 'dropout': args.get('rnn_dropout', 0.1)}
        return LSTMEncoder(6, d_model, **encoder_args) if enc_type == 'lstm' else GRUEncoder(6, d_model, **encoder_args)
    raise ValueError(f"Unsupported encoder: {enc_type}")
# --- (Model Heads, Checkpoint Loading, Data Prep, Metrics are unchanged) ---
class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks=1; q=gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0),batch_shape=torch.Size([num_tasks])); vs=gpytorch.variational.VariationalStrategy(self,inducing_points,q,learn_inducing_locations=True); mvs=gpytorch.variational.IndependentMultitaskVariationalStrategy(vs,num_tasks=num_tasks); super().__init__(mvs)
        bs=torch.Size([num_tasks]); self.mean_module=gpytorch.means.ConstantMean(batch_shape=bs); self.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,batch_shape=bs)+gpytorch.kernels.RBFKernel(batch_shape=bs),batch_shape=bs)
    def forward(self,x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x),self.covar_module(x))
class ResidualModel_SVGP(torch.nn.Module):
    def __init__(self,encoder,y_mean,y_std,d_model,n_inducing,device):
        super().__init__();self.encoder=encoder.to(device);self.gp=SVGPLayer(torch.randn(n_inducing,d_model,device=device)).to(device);self.lik=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device);self.y_mean=torch.tensor(y_mean,dtype=torch.float32,device=device);self.y_std=torch.tensor(y_std,dtype=torch.float32,device=device)
    def forward(self,X):
        self.eval();
        with torch.no_grad(),gpytorch.settings.fast_pred_var(): z=self.encoder(X);post=self.lik(self.gp(z));mu_n=post.mean.squeeze(-1);return mu_n*self.y_std+self.y_mean
class LoadedModel:
    def __init__(self, model_type, encoder_type, model, y_mean, y_std, device):
        self.model_type=model_type;self.encoder_type=encoder_type;self.model=model;self.y_mean=y_mean;self.y_std=y_std;self.device=device
def load_checkpoint(ckpt_path:Path,device:torch.device)->LoadedModel:
    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False);mtype=ckpt.get('model_type','svgp').lower();etype=ckpt.get('encoder_type','gru').lower();args=ckpt.get('args',{});dmod=args.get('dmod',32);nind=args.get('ind',150);y_mean=float(np.array(ckpt['y_mean']).reshape(-1)[0]);y_std=float(np.array(ckpt['y_std']).reshape(-1)[0]);enc=build_encoder(etype,dmod,args)
    model=ResidualModel_SVGP(enc,y_mean,y_std,dmod,nind,device).to(device)
    if 'encoder' in ckpt: model.encoder.load_state_dict(ckpt['encoder'],strict=False)
    if 'gp' in ckpt: model.gp.load_state_dict(ckpt['gp'])
    if 'lik' in ckpt: model.lik.load_state_dict(ckpt['lik'])
    return LoadedModel(mtype,etype,model,y_mean,y_std,device)
def load_data_for_scenario(csv_path: Path):
    df = pd.read_csv(csv_path).sort_values('time').reset_index(drop=True)
    dt = float(np.mean(np.diff(df['time'].values.astype(np.float64))))
    df['r_z'] = compute_rz_diff(df['yaw'].values, dt)
    return df_to_lists(df), dt
def df_to_lists(df:pd.DataFrame):
    meas,ctrl=[],[];steer=np.deg2rad(df['steer_deg'].values.astype(np.float32))*K_DELTA
    for k in range(len(df)):
        meas.append({'pos_x':float(df['pos_x'].iloc[k]),'pos_y':float(df['pos_y'].iloc[k]),'yaw':float(df['yaw'].iloc[k]),'speed':float(df['speed'].iloc[k]),'a_x':float(df['acceleration'].iloc[k]),'r_z':float(df['r_z'].iloc[k]),'delta_prev':float(steer[k-1] if k>0 else steer[0])})
        ctrl.append({'acc':float(df['acceleration'].iloc[k]),'delta_cmd':float(steer[k])})
    return meas,ctrl
def pos_errors(xm,ym,xs,ys): e=np.hypot(xm-xs,ym-ys);return float(e[-1]),float(np.mean(e)),float(np.max(e)),e
def auec(e_curve:np.ndarray,dt:float)->float: return float(np.sum(e_curve)*dt)
def drift_slope(t:np.ndarray,e_curve:np.ndarray)->float: return float(np.polyfit(t,e_curve,1)[0])
def run_anova(samples_dict:Dict[str,np.ndarray])->Optional[Dict]:
    if not SCIPY_OK: print("Skipping ANOVA: SciPy is not installed.");return None
    sample_groups=[arr for arr in samples_dict.values() if len(arr)>1]
    if len(sample_groups)<2: print("Skipping ANOVA: Need at least two groups with data.");return None
    f_val,p_val=spstats.f_oneway(*sample_groups);return {'f_statistic':f_val,'p_value':p_val}

# --- EVALUATION ---
def run_closed_loop(meas_val: list, ctrl_val: list, dt: float,
                    lm: Optional[LoadedModel], H: int) -> Dict:
    N = len(meas_val); state = meas_val[0].copy()
    xs_sim, ys_sim, yaws_sim, vs_sim = [], [], [], []
    
    # MODIFIED: Initialize param_count and inference time
    param_count = 0
    total_inference_time = 0.0

    if lm is not None:
        param_count = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
        hist = torch.zeros((H,6), dtype=torch.float32, device=lm.device)
        
    with torch.no_grad():
        for k in range(N):
            xs_sim.append(state['pos_x']); ys_sim.append(state['pos_y'])
            yaws_sim.append(state['yaw']); vs_sim.append(state['speed'])
            
            if k >= N-1: break
            
            next_base = simulate_step(state, ctrl_val[k], dt)
            
            if (lm is None) or (k < H-1):
                state = next_base; continue
                
            # NEW: Timing logic for the inference step
            start_time = time.perf_counter()
            feat = torch.tensor([state['a_x'], state['speed'], math.sin(state['yaw']), math.cos(state['yaw']),
                                 ctrl_val[k]['acc'], ctrl_val[k]['delta_cmd']], dtype=torch.float32, device=lm.device).unsqueeze(0)
            hist = torch.cat([hist[1:], feat], dim=0)
            r_res = lm.model(hist.unsqueeze(0)).squeeze(0).item()
            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)
            
            r_corr = next_base['r_z'] + r_res
            next_yaw_corr = wrap_to_pi(state['yaw'] + r_corr * dt)
            state = next_base.copy(); state['yaw']=next_yaw_corr; state['r_z']=r_corr
            
    xm=np.array([m['pos_x'] for m in meas_val]); ym=np.array([m['pos_y'] for m in meas_val])
    xs=np.asarray(xs_sim); ys=np.asarray(ys_sim)
    e_final, e_mean, e_max, e_curve = pos_errors(xm, ym, xs, ys)
    yaw_m = np.unwrap(np.array([m['yaw'] for m in meas_val]))
    yaw_s = np.unwrap(np.asarray(yaws_sim))
    yaw_rmse = np.sqrt(np.mean((yaw_m - yaw_s)**2))
    
    # MODIFIED: Add new metrics to the returned dictionary
    return dict(pos_final=e_final, pos_mean=e_mean, pos_max=e_max, yaw_rmse=yaw_rmse,
                e_curve=e_curve, x_sim=xs, y_sim=ys, yaws_sim=yaws_sim, 
                param_count=param_count, inference_time_s=total_inference_time)

# --- PLOTTING & DATA SAVING ---
def save_and_plot_scenario_results(output_dir: Path, scenario_name: str, ground_truth: dict, baseline_result: dict, model_results: dict):
    """
    Generates plots AND saves the underlying time-series data to CSV files.
    """
    # --- Save Time-Series Data ---
    all_models_data = {'BASELINE': baseline_result, **model_results}
    for model_label, result_data in all_models_data.items():
        num_steps = len(ground_truth['time'])
        df_data = {
            'time': ground_truth['time'],
            'gt_x': ground_truth['x'][:num_steps], 'gt_y': ground_truth['y'][:num_steps], 'gt_yaw': ground_truth['yaw'][:num_steps],
            'pred_x': result_data['x_sim'][:num_steps], 'pred_y': result_data['y_sim'][:num_steps],
            'pred_yaw': np.unwrap(result_data['yaws_sim'][:num_steps]),
            'pos_error': result_data['e_curve'][:num_steps]
        }
        df = pd.DataFrame(df_data)
        output_path = output_dir / f"timeseries_data_{scenario_name}_{model_label}.csv"
        df.to_csv(output_path, index=False)

    # --- Generate Plots ---
    # Trajectory Plot
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth['x'], ground_truth['y'], 'k-', lw=3, label='Ground Truth')
    plt.plot(baseline_result['x_sim'], baseline_result['y_sim'], 'g--', lw=2, label='Baseline')
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_results)))
    for i, (label, result) in enumerate(model_results.items()):
        plt.plot(result['x_sim'], result['y_sim'], color=colors[i], lw=2, label=label)
    plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(fontsize=12)
    plt.xlabel('x position (m)', fontsize=14); plt.ylabel('y position (m)', fontsize=14)
    plt.title(f'Trajectory Comparison: {scenario_name.capitalize()}', fontsize=16)
    plt.tight_layout(); plt.savefig(output_dir / f'trajectory_comparison_{scenario_name}.png', dpi=300); plt.close()

    # Heading Angle Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    t = ground_truth['time']
    ax.plot(t, np.rad2deg(ground_truth['yaw']), 'k-', lw=3, label='Ground Truth')
    ax.plot(t, np.rad2deg(np.unwrap(baseline_result['yaws_sim'])), 'g--', lw=2, label='Baseline')
    for i, (label, result) in enumerate(model_results.items()):
        ax.plot(t, np.rad2deg(np.unwrap(result['yaws_sim'])), color=colors[i], lw=2, label=label)
    ax.set_xlabel('Time (s)', fontsize=14); ax.set_ylabel('Heading Angle (degrees)', fontsize=14)
    ax.set_title(f'Heading Angle Comparison: {scenario_name.capitalize()}', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize=12); fig.tight_layout()
    plt.savefig(output_dir / f'heading_comparison_{scenario_name}.png', dpi=300); plt.close()

    # Position Error Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, baseline_result['e_curve'], 'g--', lw=2, label='Baseline')
    for i, (label, result) in enumerate(model_results.items()):
        ax.plot(t, result['e_curve'], color=colors[i], lw=2, label=label)
    ax.set_xlabel('Time (s)', fontsize=14); ax.set_ylabel('Position Error (m)', fontsize=14)
    ax.set_title(f'Position Error Comparison: {scenario_name.capitalize()}', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize=12); fig.tight_layout()
    plt.savefig(output_dir / f'error_comparison_{scenario_name}.png', dpi=300); plt.close()

# --- MAIN ---
def main():
    ap = argparse.ArgumentParser(description="Run multi-scenario evaluation for multiple models.")
    ap.add_argument("--scenario_dir", required=True, type=Path, help="Directory containing scenario CSV files.")
    ap.add_argument("--output_dir", required=True, type=Path, help="Directory to save results (plots and data).")
    ap.add_argument("--ckpts", required=True, type=Path, nargs='+', help="Paths to model checkpoint files.")
    ap.add_argument("--labels", required=True, type=str, nargs='+', help="Labels for each model.")
    ap.add_argument("--hist", type=int, default=10, help="History window size.")
    ap.add_argument("--plots", action="store_true", help="Generate per-scenario plots and save data.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.ckpts) != len(args.labels):
        raise ValueError("Number of checkpoints must match the number of labels.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scenario_files = sorted(list(args.scenario_dir.glob("*.csv")))
    if not scenario_files:
        raise FileNotFoundError(f"No CSV files found in directory: {args.scenario_dir}")
    
    models = {label: load_checkpoint(ckpt, device) for label, ckpt in zip(args.labels, args.ckpts)}

    all_results = []
    for scenario_path in scenario_files:
        scenario_name = scenario_path.stem
        print(f"\n===== Running Scenario: {scenario_name} =====")
        (meas_val, ctrl_val), dt = load_data_for_scenario(scenario_path)
        
        baseline_result = run_closed_loop(meas_val, ctrl_val, dt, lm=None, H=args.hist)
        model_results_scenario = {label: run_closed_loop(meas_val, ctrl_val, dt, lm=model, H=args.hist) for label, model in models.items()}

        t_val = np.arange(len(baseline_result['e_curve'])) * dt
        
        # MODIFIED: Add new metrics for the baseline
        all_results.append(dict(scenario=scenario_name, model='BASELINE', 
                                pos_final=baseline_result['pos_final'], pos_mean=baseline_result['pos_mean'],
                                yaw_rmse=baseline_result['yaw_rmse'],
                                AUEC=auec(baseline_result['e_curve'], dt),
                                drift_b=drift_slope(t_val, baseline_result['e_curve']),
                                param_count=baseline_result['param_count'],
                                inference_time_s=baseline_result['inference_time_s']))
        
        # MODIFIED: Add new metrics for each model
        for label, result in model_results_scenario.items():
            all_results.append(dict(scenario=scenario_name, model=label,
                                    pos_final=result['pos_final'], pos_mean=result['pos_mean'],
                                    yaw_rmse=result['yaw_rmse'],
                                    AUEC=auec(result['e_curve'], dt),
                                    drift_b=drift_slope(t_val, result['e_curve']),
                                    param_count=result['param_count'],
                                    inference_time_s=result['inference_time_s']))
        
        if args.plots:
            ground_truth_data = {
                'x': [m['pos_x'] for m in meas_val], 'y': [m['pos_y'] for m in meas_val],
                'yaw': np.unwrap([m['yaw'] for m in meas_val]), 'time': t_val
            }
            save_and_plot_scenario_results(args.output_dir, scenario_name, ground_truth_data, baseline_result, model_results_scenario)
            print(f"Saved plots and time-series data for scenario: {scenario_name}")

    detailed_df = pd.DataFrame(all_results)
    detailed_df.to_csv(args.output_dir / "detailed_results.csv", index=False)
    print(f"\nSaved detailed per-scenario results to {args.output_dir / 'detailed_results.csv'}")

    summary_df = detailed_df.groupby('model').mean(numeric_only=True).reset_index()
    summary_df.to_csv(args.output_dir / "summary_results.csv", index=False)
    print(f"Saved summary (average) results to {args.output_dir / 'summary_results.csv'}")
    
    print("\n=== SUMMARY RESULTS (Averaged Across All Scenarios) ===")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()