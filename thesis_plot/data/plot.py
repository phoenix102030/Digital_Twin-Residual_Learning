import pandas as pd
import matplotlib.pyplot as plt

# Load the trajectory datasets
traj_left_turn_df = pd.read_csv('traj_left_turn.csv')
traj_right_turn_df = pd.read_csv('traj_right_turn.csv')
traj_u_turn_df = pd.read_csv('traj_u_turn.csv')
traj_zig_zag_df = pd.read_csv('traj_zig_zag.csv')

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Trajectory Comparison of Different Architectures (Modified Style)', fontsize=16)

# --- Plot for Left Turn Trajectory ---
axes[0, 0].plot(traj_left_turn_df['gt_x'].to_numpy(), traj_left_turn_df['gt_y'].to_numpy(), label='Ground Truth', color='black', linewidth=2.5)
axes[0, 0].plot(traj_left_turn_df['baseline_pred_x'].to_numpy(), traj_left_turn_df['baseline_pred_y'].to_numpy(), label='Baseline', linestyle=':')
axes[0, 0].plot(traj_left_turn_df['transformer_pred_x'].to_numpy(), traj_left_turn_df['transformer_pred_y'].to_numpy(), label='Transformer')
axes[0, 0].plot(traj_left_turn_df['mobilevit_pred_x'].to_numpy(), traj_left_turn_df['mobilevit_pred_y'].to_numpy(), label='MobileViT')
axes[0, 0].plot(traj_left_turn_df['lstm_pred_x'].to_numpy(), traj_left_turn_df['lstm_pred_y'].to_numpy(), label='LSTM')
axes[0, 0].plot(traj_left_turn_df['gru_pred_x'].to_numpy(), traj_left_turn_df['gru_pred_y'].to_numpy(), label='GRU')
axes[0, 0].set_title('Left Turn Trajectory')
axes[0, 0].set_xlabel('X Coordinate')
axes[0, 0].set_ylabel('Y Coordinate')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_aspect('equal', adjustable='box')

# --- Plot for Right Turn Trajectory ---
axes[0, 1].plot(traj_right_turn_df['gt_x'].to_numpy(), traj_right_turn_df['gt_y'].to_numpy(), label='Ground Truth', color='black', linewidth=2.5)
axes[0, 1].plot(traj_right_turn_df['baseline_pred_x'].to_numpy(), traj_right_turn_df['baseline_pred_y'].to_numpy(), label='Baseline', linestyle=':')
axes[0, 1].plot(traj_right_turn_df['transformer_pred_x'].to_numpy(), traj_right_turn_df['transformer_pred_y'].to_numpy(), label='Transformer')
axes[0, 1].plot(traj_right_turn_df['mobilevit_pred_x'].to_numpy(), traj_right_turn_df['mobilevit_pred_y'].to_numpy(), label='MobileViT')
axes[0, 1].plot(traj_right_turn_df['lstm_pred_x'].to_numpy(), traj_right_turn_df['lstm_pred_y'].to_numpy(), label='LSTM')
axes[0, 1].plot(traj_right_turn_df['gru_pred_x'].to_numpy(), traj_right_turn_df['gru_pred_y'].to_numpy(), label='GRU')
axes[0, 1].set_title('Right Turn Trajectory')
axes[0, 1].set_xlabel('X Coordinate')
axes[0, 1].set_ylabel('Y Coordinate')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_aspect('equal', adjustable='box')

# --- Plot for U-Turn Trajectory ---
axes[1, 0].plot(traj_u_turn_df['gt_x'].to_numpy(), traj_u_turn_df['gt_y'].to_numpy(), label='Ground Truth', color='black', linewidth=2.5)
axes[1, 0].plot(traj_u_turn_df['baseline_pred_x'].to_numpy(), traj_u_turn_df['baseline_pred_y'].to_numpy(), label='Baseline', linestyle=':')
axes[1, 0].plot(traj_u_turn_df['transformer_pred_x'].to_numpy(), traj_u_turn_df['transformer_pred_y'].to_numpy(), label='Transformer')
axes[1, 0].plot(traj_u_turn_df['mobilevit_pred_x'].to_numpy(), traj_u_turn_df['mobilevit_pred_y'].to_numpy(), label='MobileViT')
axes[1, 0].plot(traj_u_turn_df['lstm_pred_x'].to_numpy(), traj_u_turn_df['lstm_pred_y'].to_numpy(), label='LSTM')
axes[1, 0].plot(traj_u_turn_df['gru_pred_x'].to_numpy(), traj_u_turn_df['gru_pred_y'].to_numpy(), label='GRU')
axes[1, 0].set_title('U-Turn Trajectory')
axes[1, 0].set_xlabel('X Coordinate')
axes[1, 0].set_ylabel('Y Coordinate')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_aspect('equal', adjustable='box')

# --- Plot for Zig-Zag Trajectory ---
axes[1, 1].plot(traj_zig_zag_df['gt_x'].to_numpy(), traj_zig_zag_df['gt_y'].to_numpy(), label='Ground Truth', color='black', linewidth=2.5)
axes[1, 1].plot(traj_zig_zag_df['baseline_pred_x'].to_numpy(), traj_zig_zag_df['baseline_pred_y'].to_numpy(), label='Baseline', linestyle=':')
axes[1, 1].plot(traj_zig_zag_df['transformer_pred_x'].to_numpy(), traj_zig_zag_df['transformer_pred_y'].to_numpy(), label='Transformer')
axes[1, 1].plot(traj_zig_zag_df['mobilevit_pred_x'].to_numpy(), traj_zig_zag_df['mobilevit_pred_y'].to_numpy(), label='MobileViT')
axes[1, 1].plot(traj_zig_zag_df['lstm_pred_x'].to_numpy(), traj_zig_zag_df['lstm_pred_y'].to_numpy(), label='LSTM')
axes[1, 1].plot(traj_zig_zag_df['gru_pred_x'].to_numpy(), traj_zig_zag_df['gru_pred_y'].to_numpy(), label='GRU')
axes[1, 1].set_title('Zig-Zag Trajectory')
axes[1, 1].set_xlabel('X Coordinate')
axes[1, 1].set_ylabel('Y Coordinate')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_aspect('equal', adjustable='box')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('architecture_trajectory_plots_modified.png')