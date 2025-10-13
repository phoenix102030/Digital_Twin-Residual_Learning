#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_compare_models_ANALYSIS.py

Performs a comparative evaluation of multiple models trained with different strategies.
This script is designed to handle the experimental design for RQ1 and RQ2.

Generates:
  - Statistical analysis (ANOVA) to compare model groups.
  - Figures for visual comparison & CSV files with detailed metrics.
  - Reports computational cost (parameters and inference time).
  - NEW: Adds a scatter plot for accuracy vs. computational cost.
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
    print("Warning: SciPy not found. Statistical analysis (ANOVA, p-values) will not be available.")
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
    def predict_dist_norm(self,X):
        self.eval();
        with torch.no_grad(),gpytorch.settings.fast_pred_var(): z=self.encoder(X);post=self.lik(self.gp(z));return post.mean.squeeze(-1),post.variance.squeeze(-1)
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
def load_csv_and_split(csv_path:Path,val_split:float=0.2):
    df=pd.read_csv(csv_path).sort_values('time').reset_index(drop=True);need={'time','pos_x','pos_y','yaw','speed','acceleration','steer_deg'};miss=need-set(df.columns)
    if miss: raise ValueError(f"CSV missing: {miss}")
    t=df['time'].values.astype(np.float64);dt=float(np.mean(np.diff(t)));df['r_z']=compute_rz_diff(df['yaw'].values,dt);n=len(df);n_val=int(max(1,n*val_split))
    return df.iloc[:n-n_val].reset_index(drop=True),df.iloc[n-n_val:].reset_index(drop=True),dt
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
                    lm: Optional[LoadedModel], H: int) -> Dict[str, float]:
    N = len(meas_val); state = meas_val[0].copy()
    xs_sim, ys_sim = [], []
    total_time = 0.0; param_count = 0
    if lm is not None:
        param_count = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
        hist = torch.zeros((H,6), dtype=torch.float32, device=lm.device)
    with torch.no_grad():
        for k in range(N):
            xs_sim.append(state['pos_x']); ys_sim.append(state['pos_y'])
            if k >= N-1: break
            next_base = simulate_step(state, ctrl_val[k], dt)
            if (lm is None) or (k < H-1):
                state = next_base; continue
            feat = torch.tensor([state['a_x'], state['speed'], math.sin(state['yaw']), math.cos(state['yaw']),
                                 ctrl_val[k]['acc'], ctrl_val[k]['delta_cmd']], dtype=torch.float32, device=lm.device).unsqueeze(0)
            hist = torch.cat([hist[1:], feat], dim=0)
            start_time = time.perf_counter()
            r_res = lm.model(hist.unsqueeze(0)).squeeze(0).item()
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
            r_corr = next_base['r_z'] + r_res
            next_yaw_corr = wrap_to_pi(state['yaw'] + r_corr * dt)
            state = next_base.copy(); state['yaw']=next_yaw_corr; state['r_z']=r_corr
    xm = np.array([m['pos_x'] for m in meas_val]); ym = np.array([m['pos_y'] for m in meas_val])
    xs = np.asarray(xs_sim); ys = np.asarray(ys_sim)
    e_final, e_mean, e_max, e_curve = pos_errors(xm, ym, xs, ys)
    avg_inference_ms = (total_time / max(1, N-H)) * 1000.0 if lm is not None else 0.0
    return dict(pos_final=e_final, pos_mean=e_mean, pos_max=e_max, e_curve=e_curve, x_sim=xs, y_sim=ys,
                param_count=param_count, avg_inference_ms=avg_inference_ms)

# --- PLOTTING ---
def plot_accuracy_vs_cost(summary_df: pd.DataFrame, out="accuracy_vs_cost.png"):
    """
    NEW: Creates a scatter plot of model accuracy vs. computational cost.
    """
    df_plot = summary_df[summary_df['model'] != 'BASELINE'].copy()
    if df_plot.empty:
        print("No models to plot for accuracy vs cost.")
        return
        
    plt.figure(figsize=(8, 6))
    plt.scatter(df_plot['avg_inference_ms'], df_plot['pos_final'], s=100, alpha=0.7)
    
    # Add labels to each point
    for i, row in df_plot.iterrows():
        plt.text(row['avg_inference_ms'] + 0.01, row['pos_final'], row['model'], fontsize=12)
        
    plt.xlabel('Average Inference Time (ms/step)')
    plt.ylabel('Final Position Error (m)')
    plt.title('Model Performance: Accuracy vs. Inference Speed')
    plt.grid(True, alpha=0.3)
    
    # Add arrow and text to indicate ideal region
    plt.annotate('More Efficient', xy=(0.2, 0.05), xycoords='axes fraction', 
                 xytext=(0.5, 0.05), arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, ha='right')
    plt.annotate('More Accurate', xy=(0.05, 0.5), xycoords='axes fraction', 
                 xytext=(0.05, 0.8), arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, va='bottom', rotation=90)

    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()

def plot_traj(xy_meas,xy_base,results_dict,out="traj_overlay.png",title="Trajectory Comparison"):
    plt.figure(figsize=(9,4));plt.plot(xy_meas[0],xy_meas[1],'k-',lw=2,label='Ground Truth');plt.plot(xy_base[0],xy_base[1],'g--',lw=1.5,label='Baseline')
    for i,(label,result) in enumerate(results_dict.items()): plt.plot(result['x_sim'],result['y_sim'],f'C{i}-',lw=1.5,label=label)
    plt.axis('equal');plt.grid(True,alpha=0.3);plt.legend();plt.xlabel('x [m]');plt.ylabel('y [m]');plt.title(title);plt.tight_layout();plt.savefig(out,dpi=220);plt.close()
def plot_error_over_time(t,base_curve,results_dict,out="error_over_time.png",title="Position Error vs Time"):
    plt.figure(figsize=(13,5));plt.plot(t,base_curve,label='Baseline')
    for i,(label,result) in enumerate(results_dict.items()): plt.plot(t[:len(result['e_curve'])],result['e_curve'],label=label)
    plt.xlabel('Time [s]');plt.ylabel('Position error [m]');plt.grid(True,alpha=0.3);plt.legend();plt.title(title);plt.tight_layout();plt.savefig(out,dpi=220);plt.close()
def mean_ci(x,alpha=0.95):
    x=np.asarray(x);n=len(x);m=float(np.mean(x));s=float(np.std(x,ddof=1)) if n>1 else 0.0
    tcrit=spstats.t.ppf((1+alpha)/2.0,df=n-1) if n>1 and SCIPY_OK else 1.96
    ci=tcrit*(s/math.sqrt(max(1,n)));return m,ci
def plot_box(samples_dict,out="box_pos_final.png",ylabel="final pos error [m]",title="Final Position Error"):
    labels=list(samples_dict.keys());data=[samples_dict[k] for k in labels];plt.figure(figsize=(2+1.5*len(labels),4))
    plt.boxplot(data,labels=labels,showmeans=True,meanline=True);plt.ylabel(ylabel);plt.title(title);plt.grid(axis='y',alpha=0.2);plt.tight_layout();plt.savefig(out,dpi=220);plt.close()
def plot_mean_ci(samples_dict,out="meanCI_pos_final.png",title="Mean ± 95% CI (final pos error)"):
    labels=list(samples_dict.keys());means=[];cis=[]
    for g in labels: m,ci=mean_ci(samples_dict[g],0.95);means.append(m);cis.append(ci)
    x=np.arange(len(labels));plt.figure(figsize=(2+1.5*len(labels),4));plt.bar(x,means,yerr=cis,capsize=6)
    plt.xticks(x,labels);plt.ylabel('final pos error [m]');plt.title(title);plt.grid(axis='y',alpha=0.2);plt.tight_layout();plt.savefig(out,dpi=220);plt.close()

# --- MAIN ---
def main():
    ap = argparse.ArgumentParser(description="Compare multiple models with statistical analysis & plots.")
    ap.add_argument("--csv", required=True, type=Path, help="Path to the data CSV file.")
    ap.add_argument("--ckpts", required=True, type=Path, nargs='+', help="Paths to model checkpoint files.")
    ap.add_argument("--labels", required=True, type=str, nargs='+', help="Labels for each model.")
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--horizon_s", type=float, default=1.5, help="Windowed horizon (seconds) for error distribution.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    if len(args.ckpts) != len(args.labels):
        raise ValueError("Number of checkpoints must match the number of labels.")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, df_val, dt = load_csv_and_split(args.csv, args.val_split)
    meas_val, ctrl_val = df_to_lists(df_val)

    results = {}
    for label, ckpt_path in zip(args.labels, args.ckpts):
        print(f"\n--- Evaluating Model: {label} ---")
        model_loaded = load_checkpoint(ckpt_path, device)
        print(f"Loaded {label}: {model_loaded.model_type.upper()} + {model_loaded.encoder_type.upper()}")
        run_output = run_closed_loop(meas_val, ctrl_val, dt, lm=model_loaded, H=args.hist)
        results[label] = run_output

    print("\n--- Evaluating Baseline ---")
    base_result = run_closed_loop(meas_val, ctrl_val, dt, lm=None, H=args.hist)
    
    summary_rows = []
    t_val = df_val['time'].values[:len(base_result['e_curve'])]
    
    base_metrics = {k: v for k, v in base_result.items() if not isinstance(v, np.ndarray)}
    base_metrics['AUEC'] = auec(base_result['e_curve'], dt)
    base_metrics['drift_b'] = drift_slope(t_val, base_result['e_curve'])
    summary_rows.append(dict(model="BASELINE", **base_metrics))

    for label, result in results.items():
        metrics = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        metrics['AUEC'] = auec(result['e_curve'], dt)
        metrics['drift_b'] = drift_slope(t_val, result['e_curve'])
        summary_rows.append(dict(model=label, **metrics))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("comparison_summary.csv", index=False)
    print("\nSaved overall metrics to comparison_summary.csv")

    steps = max(1, int(round(args.horizon_s / dt)))
    def horizon_samples(e_curve):
        if len(e_curve) <= steps: return np.array([e_curve[-1]])
        return np.asarray(e_curve[steps:], dtype=float)

    samples_dict = {'Baseline': horizon_samples(base_result['e_curve'])}
    for label, result in results.items():
        samples_dict[label] = horizon_samples(result['e_curve'])
    
    anova_results = run_anova(samples_dict)
    if anova_results:
        print("\n=== ANOVA Results (Final Position Error) ===")
        print(f"F-statistic: {anova_results['f_statistic']:.4f}, P-value: {anova_results['p_value']:.4g}")
        if anova_results['p_value'] < 0.05: print("Result is statistically significant (p < 0.05).")
        else: print("Result is not statistically significant (p >= 0.05).")
        pd.DataFrame([anova_results]).to_csv("anova_results.csv", index=False)
        print("Saved ANOVA results to anova_results.csv")

    if args.plots:
        print("\nGenerating plots...")
        plot_error_over_time(t_val, base_result['e_curve'], results, out="error_over_time.png")
        plot_box(samples_dict, out="box_pos_final.png", title=f"Final Position Error at {args.horizon_s:.2f}s Horizon")
        plot_mean_ci(samples_dict, out="meanCI_pos_final.png", title=f"Mean ± 95% CI (H={args.horizon_s:.2f}s)")
        plot_traj((df_val['pos_x'].values, df_val['pos_y'].values), (base_result['x_sim'], base_result['y_sim']), results, out="traj_overlay.png")
        # NEW: Call the new plotting function
        plot_accuracy_vs_cost(summary_df, out="accuracy_vs_cost.png")
        print("Saved plots: error_over_time.png, box_pos_final.png, meanCI_pos_final.png, traj_overlay.png, accuracy_vs_cost.png")

    print("\n=== CLOSED-LOOP METRICS (from comparison_summary.csv) ===")
    print(summary_df.to_string(formatters={'param_count': '{:,}'.format}))

if __name__ == "__main__":
    main()
