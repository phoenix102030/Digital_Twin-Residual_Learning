#!/usr/bin/env python3
import math
import argparse
import time
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from tqdm import tqdm
import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt

# ----------------------- BICYCLE PARAMETERS (Unchanged) -----------------------
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]

# ----------------------- HELPERS (Unchanged) --------------------------
def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def simulate_step(state: dict, control: dict, dt: float) -> dict:
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev'])
    x     = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y     = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt
    psi   = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v     = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt
    r_z   = wrap_to_pi(psi - state['yaw']) / dt
    a_x   = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}

def feat_tensor(state: dict, control: dict, device) -> torch.Tensor:
    return torch.tensor([
        state['a_x'], state['speed'],
        math.sin(state['yaw']), math.cos(state['yaw']),
        control['acc'], control['delta_cmd']
    ], dtype=torch.float32, device=device)

# --- MODEL AND ENCODER DEFINITIONS (Unchanged) ---
class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__(); self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        h = self.proj(x); h = self.transformer_encoder(h); h = h.mean(dim=1); h = self.outp(h); return self.final_norm(h)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__(); self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.lstm(x); return self.final_norm(h_t[:, -1, :])

class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__(); self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.gru(x); return self.final_norm(h_t[:, -1, :])

class MobileViTEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__(); self.conv_block = torch.nn.Sequential(torch.nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1), torch.nn.BatchNorm1d(d_model), torch.nn.SiLU(), torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1), torch.nn.BatchNorm1d(d_model), torch.nn.SiLU())
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers); self.final_norm = torch.nn.LayerNorm(d_model); self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        x_conv = x.permute(0, 2, 1); local_features = self.conv_block(x_conv); local_features = local_features.permute(0, 2, 1)
        global_features = self.transformer_encoder(local_features); fused_features = global_features + local_features
        h = fused_features.mean(dim=1); return self.final_norm(self.outp(h))

class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = 1; variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0), batch_shape=torch.Size([num_tasks]))
        variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist, learn_inducing_locations=True)
        mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strat, num_tasks=num_tasks); super().__init__(mts)
        bs = torch.Size([num_tasks]); self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs), batch_shape=bs)
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class ResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__(); self.device = device; self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        if encoder_type == 'transformer': self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm': self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru': self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'mobilevit': self.encoder = MobileViTEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else: raise ValueError(f"Unknown encoder type: {encoder_type}")
        Z = torch.randn(ckpt_args['ind'], ckpt_args['dmod'], device=device); self.gp  = SVGPLayer(Z).to(device); self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device)
    def forward(self, x):
        z = self.encoder(x); dist = self.gp(z); pred_norm = self.lik(dist).mean; return pred_norm * self.y_std + self.y_mean

class DeterministicResidualModel(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, ckpt_args, y_mean, y_std, device):
        super().__init__(); self.device = device; self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        if encoder_type == 'transformer': self.encoder = TransformerEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'lstm': self.encoder = LSTMEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'gru': self.encoder = GRUEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        elif encoder_type == 'mobilevit': self.encoder = MobileViTEncoder(input_dim=6, d_model=ckpt_args['dmod'], **encoder_args).to(device)
        else: raise ValueError(f"Unknown encoder type: {encoder_type}")
        self.regressor = torch.nn.Linear(ckpt_args['dmod'], 1).to(device)
    def forward(self, x):
        z = self.encoder(x); pred_norm = self.regressor(z); return pred_norm * self.y_std + self.y_mean

# --- SIMULATION & EVALUATION (Unchanged) ---
def run_base_simulation(meas, ctrl, dt):
    print("Running base simulation (kinematic model only)..."); t_start = time.time()
    N = len(meas); sim_states = []; state = meas[0].copy()
    for k in tqdm(range(N), desc="Base Sim"):
        sim_states.append(state.copy()); state = simulate_step(state, ctrl[k], dt) if k < N - 1 else state
    sim_time = time.time() - t_start
    results_dict = {'pos_x': np.array([s['pos_x'] for s in sim_states]), 'pos_y': np.array([s['pos_y'] for s in sim_states]), 'yaw': np.array([s['yaw'] for s in sim_states]), 'speed': np.array([s['speed'] for s in sim_states])}
    return results_dict, sim_time

def run_corrected_simulation(model, meas, ctrl, H, dt, device):
    print("Running corrected simulation (kinematic + residual model)..."); t_start = time.time()
    N = len(meas); model.eval(); sim_states = []; state = meas[0].copy()
    history_features = torch.zeros((H, 6), device=device)
    for k in tqdm(range(N), desc="Corrected Sim"):
        sim_states.append(state.copy());
        if k >= N - 1: continue
        current_features = feat_tensor(state, ctrl[k], device); history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)
        rz_residual = 0.0
        if k >= H:
            with torch.no_grad(): rz_residual = model(history_features.unsqueeze(0)).item()
        next_state_base = simulate_step(state, ctrl[k], dt); corrected_rz = next_state_base['r_z'] + rz_residual
        next_yaw_corr = wrap_to_pi(state['yaw'] + corrected_rz * dt); state = next_state_base.copy()
        state['yaw'] = next_yaw_corr; state['r_z'] = corrected_rz
    sim_time = time.time() - t_start
    results_dict = {'pos_x': np.array([s['pos_x'] for s in sim_states]), 'pos_y': np.array([s['pos_y'] for s in sim_states]), 'yaw': np.array([s['yaw'] for s in sim_states]), 'speed': np.array([s['speed'] for s in sim_states])}
    return results_dict, sim_time

# --- Metrics calculation (Unchanged) ---
def calculate_metrics(real_data, sim_data):
    """Calculates a dictionary of metrics, including error over time."""
    yaw_err_ts = np.array([wrap_to_pi(ry - sy) for ry, sy in zip(real_data['yaw'], sim_data['yaw'])])
    yaw_rmse = np.sqrt(np.mean(yaw_err_ts**2))
    speed_rmse = np.sqrt(np.mean((real_data['speed'] - sim_data['speed'])**2))
    pos_err_ts = np.sqrt((real_data['pos_x'] - sim_data['pos_x'])**2 + (real_data['pos_y'] - sim_data['pos_y'])**2)
    metrics = {
        'pos_err_final_m': pos_err_ts[-1], 'pos_err_mean_m': np.mean(pos_err_ts),
        'pos_err_max_m': np.max(pos_err_ts), 'speed_rmse_mps': speed_rmse,
        'yaw_rmse_rad': yaw_rmse, 'pos_err_vs_time': pos_err_ts
    }
    return metrics

def save_results_to_csv(results_dict, filename="evaluation_summary.csv"):
    results_to_save = results_dict.copy()
    results_to_save.pop('base_pos_err_vs_time', None)
    results_to_save.pop('corr_pos_err_vs_time', None)
    file_exists = Path(filename).exists()
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_to_save.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_to_save)
    print(f"Results appended to {filename}")

# --- NEW FUNCTION TO SAVE ERROR TIME SERIES ---
def save_error_timeseries_to_csv(times, base_errors, corr_errors, model_info_str):
    """Saves the error vs. time data to a dedicated CSV file."""
    df_error = pd.DataFrame({
        'time_s': times,
        'base_model_error_m': base_errors,
        'corrected_model_error_m': corr_errors
    })
    save_name = f"error_timeseries_{model_info_str.lower().replace(' ', '_')}.csv"
    df_error.to_csv(save_name, index=False)
    print(f"Saved error time series data to {save_name}")

# --- Plotting function (Unchanged) ---
def plot_results(times, real, sim_base, sim_corr, base_metrics, corr_metrics, model_info_str):
    print("Generating plots...")
    fig = plt.figure(figsize=(18, 16)); gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, :]); ax1.plot(real['pos_x'], real['pos_y'], 'k-', lw=2, label='Ground Truth')
    ax1.plot(sim_base['pos_x'], sim_base['pos_y'], 'g--', label=f"Base Sim (Final Err: {base_metrics['pos_err_final_m']:.2f} m)")
    ax1.plot(sim_corr['pos_x'], sim_corr['pos_y'], 'r-', lw=2.0, label=f"Corrected Sim (Final Err: {corr_metrics['pos_err_final_m']:.2f} m)")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_title("Trajectory Comparison"); ax1.legend(); ax1.grid(True); ax1.axis('equal')
    ax_pos_err = fig.add_subplot(gs[1, :]);
    ax_pos_err.plot(times, base_metrics['pos_err_vs_time'], 'g--', label=f"Base Sim (Mean Err: {base_metrics['pos_err_mean_m']:.2f} m)")
    ax_pos_err.plot(times, corr_metrics['pos_err_vs_time'], 'r-', label=f"Corrected Sim (Mean Err: {corr_metrics['pos_err_mean_m']:.2f} m)")
    ax_pos_err.set_xlabel("Time (s)"); ax_pos_err.set_ylabel("Positional Error (m)"); ax_pos_err.set_title("Positional Error over Time"); ax_pos_err.legend(); ax_pos_err.grid(True)
    ax2 = fig.add_subplot(gs[2, 0]); ax2.plot(times, real['speed'], 'k-', label='Real'); ax2.plot(times, sim_base['speed'], 'g--', label=f"Base (RMSE: {base_metrics['speed_rmse_mps']:.3f})")
    ax2.plot(times, sim_corr['speed'], 'r-', label=f"Corrected (RMSE: {corr_metrics['speed_rmse_mps']:.3f})"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Speed (m/s)"); ax2.set_title("Speed Comparison"); ax2.legend(); ax2.grid(True)
    ax3 = fig.add_subplot(gs[2, 1]); ax3.plot(times, [wrap_to_pi(y) for y in real['yaw']], 'k-', label='Real'); ax3.plot(times, [wrap_to_pi(y) for y in sim_base['yaw']], 'g--', label=f"Base (RMSE: {base_metrics['yaw_rmse_rad']:.3f})")
    ax3.plot(times, [wrap_to_pi(y) for y in sim_corr['yaw']], 'r-', label=f"Corrected (RMSE: {corr_metrics['yaw_rmse_rad']:.3f})"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Yaw (rad)"); ax3.set_title("Yaw Comparison"); ax3.legend(); ax3.grid(True)
    fig.suptitle(f'Long-Term Simulation Evaluation (Model: {model_info_str})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_name = f"evaluation_long_term_{model_info_str.lower().replace(' ', '_')}.png"
    plt.savefig(save_name); print(f"Saved plot to {save_name}"); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a trained residual dynamics model.")
    p.add_argument("--csv", required=True, help="Path to the test data CSV file.")
    p.add_argument("--model", required=True, help="Path to the saved model .pt file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading model from {args.model} onto {args.device}...")
    ckpt = torch.load(args.model, map_location=args.device)
    model_args = ckpt['args']
    encoder_type = ckpt.get('encoder_type', 'transformer'); print(f"Detected Encoder Type: {encoder_type.upper()}")
    encoder_args = {'nhead': model_args.get('tf_nhead', 2), 'd_hid': model_args.get('tf_d_hid', 128), 'nlayers': model_args.get('tf_nlayers', 2), 'dropout': model_args.get('tf_dropout', 0.1)} if encoder_type == 'transformer' else {'nlayers': model_args.get('rnn_nlayers', 2), 'dropout': model_args.get('rnn_dropout', 0.1)}
    if 'gp' in ckpt:
        print("Detected Probabilistic (SVGP) model."); model_variant = 'SVGP'
        model = ResidualModel(encoder_type=encoder_type, encoder_args=encoder_args, ckpt_args=model_args, y_mean=ckpt['y_mean'], y_std=ckpt['y_std'], device=args.device)
        model.encoder.load_state_dict(ckpt['encoder']); model.gp.load_state_dict(ckpt['gp']); model.lik.load_state_dict(ckpt['lik'])
    elif 'regressor' in ckpt:
        print("Detected Deterministic (Linear) model."); model_variant = 'Linear'
        model = DeterministicResidualModel(encoder_type=encoder_type, encoder_args=encoder_args, ckpt_args=model_args, y_mean=ckpt['y_mean'], y_std=ckpt['y_std'], device=args.device)
        model.encoder.load_state_dict(ckpt['encoder']); model.regressor.load_state_dict(ckpt['regressor'])
    else: raise ValueError("Could not determine model variant from checkpoint.")
    print("Model loaded successfully.")
    
    print(f"Loading test data from {args.csv}...")
    df = pd.read_csv(Path(args.csv)); times = df["time"].values
    real = {'pos_x': df["pos_x"].values, 'pos_y': df["pos_y"].values, 'speed': df["speed"].values, 'yaw': df["yaw"].values}
    acc_cmd = df["acceleration"].values; steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    N = len(times); dt = float(np.mean(np.diff(times))); r_z_meas = np.zeros(N); r_z_meas[1:] = np.diff(real['yaw']) / dt
    meas, ctrl = [], []
    for k in range(N):
        meas.append({'pos_x': real['pos_x'][k], 'pos_y': real['pos_y'][k], 'yaw': real['yaw'][k], 'speed': real['speed'][k], 'r_z': r_z_meas[k], 'a_x': acc_cmd[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})

    sim_base, time_base = run_base_simulation(meas, ctrl, dt)
    sim_corr, time_corr = run_corrected_simulation(model, meas, ctrl, model_args['hist'], dt, args.device)
    model_info_str = f"{encoder_type.upper()} {model_variant}"
    
    base_metrics = calculate_metrics(real, sim_base)
    corr_metrics = calculate_metrics(real, sim_corr)
    
    results_to_save = {
        'model_name': model_info_str, 'sim_duration_s': round(times[-1] - times[0], 2),
        'base_sim_time_s': round(time_base, 4), 'corrected_sim_time_s': round(time_corr, 4),
        'base_pos_err_final_m': round(base_metrics['pos_err_final_m'], 4),
        'base_pos_err_mean_m': round(base_metrics['pos_err_mean_m'], 4),
        'base_pos_err_max_m': round(base_metrics['pos_err_max_m'], 4),
        'base_speed_rmse_mps': round(base_metrics['speed_rmse_mps'], 4),
        'base_yaw_rmse_rad': round(base_metrics['yaw_rmse_rad'], 4),
        'corr_pos_err_final_m': round(corr_metrics['pos_err_final_m'], 4),
        'corr_pos_err_mean_m': round(corr_metrics['pos_err_mean_m'], 4),
        'corr_pos_err_max_m': round(corr_metrics['pos_err_max_m'], 4),
        'corr_speed_rmse_mps': round(corr_metrics['speed_rmse_mps'], 4),
        'corr_yaw_rmse_rad': round(corr_metrics['yaw_rmse_rad'], 4),
    }
    save_results_to_csv(results_to_save)
    
    # --- NEW: SAVE THE ERROR TIME SERIES DATA ---
    save_error_timeseries_to_csv(times, base_metrics['pos_err_vs_time'], corr_metrics['pos_err_vs_time'], model_info_str)
    
    print(f"\n--- Metrics Summary for {model_info_str} ---")
    print(f"               | {'Base Sim':<15} | {'Corrected Sim':<15}")
    print(f"---------------|-----------------|-----------------")
    print(f"Sim Time (s)   | {time_base:<15.3f} | {time_corr:<15.3f}")
    print(f"Final Pos Err (m)| {base_metrics['pos_err_final_m']:<15.3f} | {corr_metrics['pos_err_final_m']:<15.3f}")
    print(f"Mean Pos Err (m) | {base_metrics['pos_err_mean_m']:<15.3f} | {corr_metrics['pos_err_mean_m']:<15.3f}")
    print(f"Max Pos Err (m)  | {base_metrics['pos_err_max_m']:<15.3f} | {corr_metrics['pos_err_max_m']:<15.3f}")
    print(f"Speed RMSE     | {base_metrics['speed_rmse_mps']:<15.3f} | {corr_metrics['speed_rmse_mps']:<15.3f}")
    print(f"Yaw RMSE (rad) | {base_metrics['yaw_rmse_rad']:<15.3f} | {corr_metrics['yaw_rmse_rad']:<15.3f}\n")
    
    plot_results(times, real, sim_base, sim_corr, base_metrics, corr_metrics, model_info_str)