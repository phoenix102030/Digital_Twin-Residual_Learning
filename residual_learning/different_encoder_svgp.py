#!/usr/bin/env python3
"""
train_unified.py - MODULAR & OPTIMIZED FOR REAL-WORLD PERFORMANCE

- UPDATED: Saves the best model based on **final positional error** from a full,
           closed-loop simulation on a held-out validation trajectory.
- This is a more robust metric for simulation tasks than one-step MSE.

Train a residual model for YAW RATE ONLY with a choice of:
1. Model Type: SVGP (probabilistic) or Linear (deterministic)
2. Encoder: Transformer, LSTM, GRU, or MobileViT
... and genuine multi-step rollout loss.
"""
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ----------------------- BICYCLE PARAMETERS -----------------------
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]

# ----------------------- HELPERS --------------------------
def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def simulate_step(state: dict, control: dict, dt: float) -> dict:
    delta = state['delta_prev'] + (dt / TAU_DELTA) * (control['delta_cmd'] - state['delta_prev'])
    x     = state['pos_x'] + state['speed'] * math.cos(state['yaw']) * dt
    y     = state['pos_y'] + state['speed'] * math.sin(state['yaw']) * dt
    psi   = wrap_to_pi(state['yaw'] + (state['speed'] / L) * math.tan(delta) * dt)
    v     = state['speed'] + (control['acc'] - CD * state['speed']**2) * dt
    r_z   = (psi - state['yaw']) / dt
    a_x   = (v - state['speed']) / dt
    return {'pos_x': x, 'pos_y': y, 'yaw': psi, 'speed': v, 'delta_prev': delta, 'r_z': r_z, 'a_x': a_x}

def simulate_step_torch(states: dict, controls: dict, dt: float) -> dict:
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
    # Feature tensor for single-step simulation (non-batch)
    return torch.tensor([
        state['a_x'], state['speed'],
        math.sin(state['yaw']), math.cos(state['yaw']),
        control['acc'], control['delta_cmd']
    ], dtype=torch.float32, device=device)

def feat_tensor_torch(states: dict, controls: dict) -> torch.Tensor:
    # Feature tensor for batched rollout loss
    return torch.stack([
        states['a_x'], states['speed'],
        torch.sin(states['yaw']), torch.cos(states['yaw']),
        controls['acc'], controls['delta_cmd']
    ], dim=1)

# ----------------------- NEW: SIMULATION EVALUATION FUNCTION -----------------------
def evaluate_simulation_performance(model, val_meas, val_ctrl, H, dt, device):
    """
    Runs a full closed-loop simulation on a validation trajectory and returns the final positional error.
    """
    model.eval()
    N = len(val_meas)
    sim_states = []
    state = val_meas[0].copy()
    history_features = torch.zeros((H, 6), device=device)

    with torch.no_grad():
        for k in range(N):
            sim_states.append(state)
            if k >= N - 1: continue

            current_features = feat_tensor(state, val_ctrl[k], device)
            history_features = torch.cat([history_features[1:], current_features.unsqueeze(0)], dim=0)
            
            rz_residual = 0.0
            if k >= H:
                rz_residual = model(history_features.unsqueeze(0)).item()
            
            next_state_base = simulate_step(state, val_ctrl[k], dt)
            corrected_rz = next_state_base['r_z'] + rz_residual
            next_yaw_corr = wrap_to_pi(state['yaw'] + corrected_rz * dt)
            
            state = next_state_base.copy()
            state['yaw'] = next_yaw_corr
            state['r_z'] = corrected_rz

    real_pos_x = np.array([s['pos_x'] for s in val_meas])
    real_pos_y = np.array([s['pos_y'] for s in val_meas])
    sim_pos_x = np.array([s['pos_x'] for s in sim_states])
    sim_pos_y = np.array([s['pos_y'] for s in sim_states])
    
    # Calculate final positional error
    pos_error = np.sqrt((real_pos_x[-1] - sim_pos_x[-1])**2 + (real_pos_y[-1] - sim_pos_y[-1])**2)
    return pos_error

# ----------------------- PLOTTING HELPER --------------------------
def plot_curves(train_history, val_history, val_epochs, model_type, encoder_type):
    """Plots training loss and validation positional error on a twin-axis plot."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training loss
    epochs_train = range(1, len(train_history) + 1)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average Training MSE Loss', color=color)
    ax1.plot(epochs_train, train_history, 'b-o', label='Training MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Create a second y-axis for the validation positional error
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Positional Error [m]', color=color)
    ax2.plot(val_epochs, val_history, 'r-s', label='Validation Positional Error')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(f'Training & Validation Performance for {model_type.upper()} with {encoder_type.upper()} Encoder')
    fig.tight_layout()
    plt.savefig(f'training_curve_{model_type}_{encoder_type}.png')
    print(f"Saved training curve to training_curve_{model_type}_{encoder_type}.png")
    plt.show()

# ----------------------- DATASET HELPERS --------------------------
class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X; self.Y = Y
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], idx

# ----------------------- ENCODER DEFINITIONS -----------------------
class MobileViTEncoder(torch.nn.Module):
    """
    A 1D adaptation of the MobileViT concept for sequential data.
    It uses 1D convolutions to learn local features efficiently and a
    Transformer to learn global relationships.
    """
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        
        # Convolutional block to learn local features
        self.conv_block = torch.nn.Sequential(
            # Input shape: (Batch, Channels=input_dim, Length=history)
            torch.nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.SiLU(),
            torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.SiLU()
        )

        # Standard Transformer Encoder for global features
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.outp = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        # Input x shape: (Batch, History, Features)
        x_conv = x.permute(0, 2, 1) # -> (Batch, Features, History)
        local_features = self.conv_block(x_conv)
        local_features = local_features.permute(0, 2, 1) # -> (Batch, History, Features)

        global_features = self.transformer_encoder(local_features)
        
        # Simple fusion: element-wise addition
        fused_features = global_features + local_features
        
        # Aggregate features over the time dimension
        h = fused_features.mean(dim=1)
        
        return self.final_norm(self.outp(h))

class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nhead=2, d_hid=128, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, d_model)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.outp = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        h = self.proj(x); h = self.transformer_encoder(h); h = h.mean(dim=1); h = self.outp(h); return self.final_norm(h)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.lstm(x); return self.final_norm(h_t[:, -1, :])

class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, d_model=32, nlayers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, d_model, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.final_norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        h_t, _ = self.gru(x); return self.final_norm(h_t[:, -1, :])

# ----------------------- SVGP LAYER (Unchanged) -----------------------
class SVGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = 1; variational_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0), batch_shape=torch.Size([num_tasks]))
        variational_strat = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist, learn_inducing_locations=True)
        mts = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strat, num_tasks=num_tasks); super().__init__(mts)
        bs = torch.Size([num_tasks]); self.mean_module  = gpytorch.means.ConstantMean(batch_shape=bs)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=bs) + gpytorch.kernels.RBFKernel(batch_shape=bs), batch_shape=bs)
    def forward(self, x): return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

# ----------------------- MODEL DEFINITIONS -----------------------
def build_encoder(encoder_type, d_model, encoder_args):
    if encoder_type == 'transformer': return TransformerEncoder(input_dim=6, d_model=d_model, **encoder_args)
    if encoder_type == 'lstm': return LSTMEncoder(input_dim=6, d_model=d_model, **encoder_args)
    if encoder_type == 'gru': return GRUEncoder(input_dim=6, d_model=d_model, **encoder_args)
    if encoder_type == 'mobilevit': return MobileViTEncoder(input_dim=6, d_model=d_model, **encoder_args)
    raise ValueError(f"Unknown encoder type: {encoder_type}")

class ResidualModel_SVGP(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, y_mean, y_std, d_model, n_inducing, lr, weight_decay, device):
        super().__init__(); self.device = device; self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        self.encoder = build_encoder(encoder_type, d_model, encoder_args).to(device); Z = torch.randn(n_inducing, d_model, device=device); self.gp  = SVGPLayer(Z).to(device)
        self.lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1).to(device); self.mll = gpytorch.mlls.VariationalELBO(self.lik, self.gp, num_data=1)
        self.opt = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay}, {'params': self.gp.parameters(), 'lr': lr, 'weight_decay': weight_decay}, {'params': self.lik.parameters(), 'lr': lr}])
    def forward(self, x):
        z = self.encoder(x); dist = self.gp(z); pred_norm = self.lik(dist).mean; return pred_norm * self.y_std + self.y_mean

class ResidualModel_Linear(torch.nn.Module):
    def __init__(self, encoder_type, encoder_args, y_mean, y_std, d_model, lr, weight_decay, device):
        super().__init__(); self.device = device; self.y_mean = torch.tensor(y_mean, dtype=torch.float32, device=device); self.y_std  = torch.tensor(y_std,  dtype=torch.float32, device=device)
        self.encoder = build_encoder(encoder_type, d_model, encoder_args).to(device); self.regressor = torch.nn.Linear(d_model, 1).to(device)
        self.loss_fn = torch.nn.MSELoss(); self.opt = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},{'params': self.regressor.parameters(), 'lr': lr, 'weight_decay': weight_decay}])
    def forward(self, x):
        z = self.encoder(x); pred_norm = self.regressor(z); return pred_norm * self.y_std + self.y_mean
    def predict_normalized(self, x):
        z = self.encoder(x); return self.regressor(z)

# ----------------------- BUILD DATA -----------------------
def build_data(meas, ctrl, H, dt):
    N = len(meas); M = N - (H + 1); X = np.zeros((M, H, 6), np.float32); Y = np.zeros((M, 1), np.float32)
    print("Building one-step residual training data...");
    for i in tqdm(range(M)):
        for j in range(H):
            k = i + j; X[i, j] = [meas[k]['a_x'], meas[k]['speed'], math.sin(meas[k]['yaw']), math.cos(meas[k]['yaw']), ctrl[k]['acc'], ctrl[k]['delta_cmd']]
        state0 = meas[i+H-1].copy(); control0 = ctrl[i+H-1].copy(); next_state = simulate_step(state0, control0, dt)
        Y[i, 0] = meas[i+H]['r_z'] - next_state['r_z']
    return torch.from_numpy(X), torch.from_numpy(Y)

# ----------------------- ROLLOUT STEP for training loss (Unchanged) -----------------------
def rollout_step(model, hists, i0s, H, N, K_roll, meas, ctrl, dt, device):
    roll_r_loss = 0.0; B = hists.size(0); initial_indices = i0s + H - 1
    batch_states = {'pos_x': torch.tensor([meas[i]['pos_x'] for i in initial_indices], device=device, dtype=torch.float32),'pos_y': torch.tensor([meas[i]['pos_y'] for i in initial_indices], device=device, dtype=torch.float32),'yaw':   torch.tensor([meas[i]['yaw'] for i in initial_indices], device=device, dtype=torch.float32),'speed': torch.tensor([meas[i]['speed'] for i in initial_indices], device=device, dtype=torch.float32),'delta_prev': torch.tensor([meas[i]['delta_prev'] for i in initial_indices], device=device, dtype=torch.float32)}
    for k_step in range(K_roll):
        current_indices = i0s + H - 1 + k_step; true_next_indices = current_indices + 1
        if np.max(true_next_indices) >= N: break
        rs = model(hists)
        batch_controls = {'acc': torch.tensor([ctrl[i]['acc'] for i in current_indices], device=device, dtype=torch.float32),'delta_cmd': torch.tensor([ctrl[i]['delta_cmd'] for i in current_indices], device=device, dtype=torch.float32)}
        base_next_states = simulate_step_torch(batch_states, batch_controls, dt); corrected_next_states = base_next_states.copy(); corrected_next_states['r_z'] += rs.squeeze(-1)
        true_next_rz = torch.tensor([meas[i]['r_z'] for i in true_next_indices], device=device, dtype=torch.float32)
        roll_r_loss += torch.nn.functional.mse_loss(corrected_next_states['r_z'], true_next_rz, reduction='sum')
        new_feats = feat_tensor_torch(corrected_next_states, batch_controls); hists = torch.cat([hists[:, 1:, :], new_feats.unsqueeze(1)], dim=1); batch_states = corrected_next_states
    return roll_r_loss / (B * K_roll) if K_roll > 0 else 0.0

# ----------------------- UNIFIED TRAINING LOOP -----------------------
def train_model(model, train_loader, val_meas, val_ctrl, full_meas, full_ctrl, dt, device, epochs, args):
    scheduler = CosineAnnealingWarmRestarts(model.opt, T_0=20, T_mult=2)
    best_pos_error = float('inf')
    save_path = Path(args.save)
    best_model_path = save_path.with_name(f"{save_path.stem}_{args.model_type}_{args.encoder}_best_pos_err{save_path.suffix}")
    
    train_mse_history = []
    val_pos_error_history = []
    val_epochs = []

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [{args.model_type.upper()} Train]")
        total_train_mse = 0.0

        for bx, by, idx in pbar:
            bx, by, i0s = bx.to(device), by.to(device), idx.numpy()
            model.opt.zero_grad()
            
            # --- Calculate Loss ---
            is_svgp = hasattr(model, 'mll')
            if is_svgp:
                dist_norm = model.gp(model.encoder(bx))
                one_step_loss = -model.mll(dist_norm, by)
                pred_unnorm = model.lik(dist_norm).mean * model.y_std + model.y_mean
            else: # Linear model
                pred_norm = model.predict_normalized(bx)
                one_step_loss = model.loss_fn(pred_norm, by)
                pred_unnorm = pred_norm * model.y_std + model.y_mean

            roll_r = rollout_step(model, bx.clone(), i0s, args.hist, len(full_meas), args.k_roll, full_meas, full_ctrl, dt, device)
            loss = one_step_loss + args.rz_weight * roll_r
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.opt.step()
            
            # --- Log training metrics ---
            target_unnorm = by * model.y_std + model.y_mean
            mse_unnorm_sum = torch.nn.functional.mse_loss(pred_unnorm, target_unnorm, reduction='sum').item()
            total_train_mse += mse_unnorm_sum
            roll_val = roll_r.item() if torch.is_tensor(roll_r) else float(roll_r)
            pbar.set_postfix(
                one_step_loss=f"{one_step_loss.item():.2f}",
                mse=f"{mse_unnorm_sum/bx.size(0):.4f}",
                roll_mse=f"{roll_val:.4f}"
            )

        avg_train_mse = total_train_mse / len(train_loader.dataset)
        train_mse_history.append(avg_train_mse)

        # --- Periodic Validation with Full Simulation ---
        if ep % args.eval_freq == 0 or ep == epochs:
            pos_error = evaluate_simulation_performance(model, val_meas, val_ctrl, args.hist, dt, device)
            print(f"\n--- Validation Sim @ Epoch {ep} | Final Positional Error: {pos_error:.3f} m ---\n")
            val_pos_error_history.append(pos_error)
            val_epochs.append(ep)
            
            if pos_error < best_pos_error:
                best_pos_error = pos_error
                print(f"🎉 New best model! Positional Error: {best_pos_error:.3f} m. Saving to {best_model_path}...")
                
                save_payload = {'model_type': args.model_type, 'encoder_type': args.encoder, 'encoder': model.encoder.state_dict(), 'args': vars(args), 'y_mean': model.y_mean.cpu().numpy(), 'y_std': model.y_std.cpu().numpy()}
                if is_svgp:
                    save_payload.update({'gp': model.gp.state_dict(), 'lik': model.lik.state_dict()})
                else:
                    save_payload['regressor'] = model.regressor.state_dict()
                torch.save(save_payload, best_model_path)
        
        scheduler.step()
    
    return train_mse_history, val_pos_error_history, val_epochs

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a residual dynamics model, optimizing for positional error.")
    p.add_argument("--csv", required=True, help="Path to the training data CSV.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to hold out for validation simulation.")
    p.add_argument("--eval_freq", type=int, default=10, help="Run full validation simulation every N epochs.")
    p.add_argument("--hist", type=int, default=10, help="History window size.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--save", default="residual_model.pt")
    p.add_argument("--model_type", type=str, required=True, choices=["svgp", "linear"])
    p.add_argument("--encoder", type=str, required=True, choices=["transformer", "lstm", "gru", "mobilevit"])
    p.add_argument("--dmod", type=int, default=32)
    p.add_argument("--ind", type=int, default=150)
    p.add_argument("--k_roll", type=int, default=5)
    p.add_argument("--rz_weight", type=float, default=1.0)
    p.add_argument("--tf_nhead", type=int, default=4)
    p.add_argument("--tf_d_hid", type=int, default=128)
    p.add_argument("--tf_nlayers", type=int, default=2)
    p.add_argument("--tf_dropout", type=float, default=0.1)
    p.add_argument("--rnn_nlayers", type=int, default=2)
    p.add_argument("--rnn_dropout", type=float, default=0.1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model: {args.model_type.upper()}, Encoder: {args.encoder.upper()}")

    # --- Load and Split Data ---
    df = pd.read_csv(Path(args.csv)); assert not df.isnull().values.any(), "NaN values found in input CSV!"
    times = df["time"].values.astype(np.float32); pos_x_meas = df["pos_x"].values; pos_y_meas = df["pos_y"].values
    acc_meas  = df["acceleration"].values; speed_meas= df["speed"].values; yaw_meas  = df["yaw"].values
    acc_cmd   = acc_meas.copy(); steer_cmd = np.deg2rad(df["steer_deg"].values) * K_DELTA
    if len(times) <= args.hist + 1: raise ValueError("Not enough data for history window.")
    
    N = len(times); dt = float(np.mean(np.diff(times)))
    r_z_meas = np.zeros(N, dtype=np.float32); r_z_meas[1:] = np.diff(yaw_meas) / dt
    
    full_meas, full_ctrl = [], []
    for k in range(N):
        full_meas.append({'pos_x': pos_x_meas[k], 'pos_y': pos_y_meas[k], 'yaw': yaw_meas[k], 'speed': speed_meas[k], 'r_z': r_z_meas[k], 'a_x': acc_meas[k], 'delta_prev': steer_cmd[k-1] if k > 0 else steer_cmd[0]})
        full_ctrl.append({'acc': acc_cmd[k], 'delta_cmd': steer_cmd[k]})

    # Split trajectory data for training and validation simulation
    n_total = len(full_meas)
    n_val_traj = int(n_total * args.val_split)
    n_train_traj = n_total - n_val_traj
    
    train_meas, train_ctrl = full_meas[:n_train_traj], full_ctrl[:n_train_traj]
    val_meas, val_ctrl = full_meas[n_train_traj:], full_ctrl[n_train_traj:]
    print(f"Data split: {n_train_traj} steps for training, {n_val_traj} steps for validation simulation.")

    # Build windowed dataset for training ONLY
    X, Y = build_data(train_meas, train_ctrl, args.hist, dt)
    assert not torch.isnan(X).any(); assert not torch.isnan(Y).any()
    
    y_mean = Y.mean(dim=0).numpy(); y_std  = Y.std(dim=0).numpy() + 1e-8
    Y_norm = (Y - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)
    
    train_ds = WindowDataset(X, Y_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    print("\n" + "="*50); print(f"Values needed for validation.py:\n--y_mean {y_mean[0]:.8f}\n--y_std  {y_std[0]:.8f}"); print("="*50 + "\n")

    # --- Initialize Model ---
    if args.encoder in ['transformer', 'mobilevit']:
        encoder_args = {'nhead': args.tf_nhead, 'd_hid': args.tf_d_hid, 'nlayers': args.tf_nlayers, 'dropout': args.tf_dropout}
    else: # lstm, gru
        encoder_args = {'nlayers': args.rnn_nlayers, 'dropout': args.rnn_dropout}
        
    model = None
    if args.model_type == 'svgp':
        model = ResidualModel_SVGP(args.encoder, encoder_args, y_mean, y_std, args.dmod, args.ind, args.lr, args.wd, device).to(device)
        model.mll.num_data = len(train_ds)
    elif args.model_type == 'linear':
        model = ResidualModel_Linear(args.encoder, encoder_args, y_mean, y_std, args.dmod, args.lr, args.wd, device).to(device)

    # --- Train Model ---
    train_history, val_history, val_epochs = train_model(
        model, train_loader, val_meas, val_ctrl, train_meas, train_ctrl, dt, device, args.epochs, args
    )

    print("\nTraining complete. The best model (based on positional error) has been saved.")

    # Plot the training and validation curves
    if train_history and val_history:
        plot_curves(train_history, val_history, val_epochs, args.model_type, args.encoder)
