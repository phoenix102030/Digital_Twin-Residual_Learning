import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
left_turn_df = pd.read_csv('position_error_left_turn.csv')
right_turn_df = pd.read_csv('position_error_right_turn.csv')
u_turn_df = pd.read_csv('position_error_u_turn.csv')
zig_zag_df = pd.read_csv('position_error_zig_zag.csv')

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Position Error for Different Maneuvers', fontsize=16)

# Plot for Left Turn
axes[0, 0].plot(left_turn_df)
axes[0, 0].set_title('(a) Left Turn')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Position Error')
axes[0, 0].legend(left_turn_df.columns)
axes[0, 0].grid(True)

# Plot for Right Turn
axes[0, 1].plot(right_turn_df)
axes[0, 1].set_title('(b) Right Turn')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Position Error')
axes[0, 1].legend(right_turn_df.columns)
axes[0, 1].grid(True)

# Plot for U-Turn
axes[1, 0].plot(u_turn_df)
axes[1, 0].set_title('(c) U-Turn')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Position Error')
axes[1, 0].legend(u_turn_df.columns)
axes[1, 0].grid(True)

# Plot for Zig-Zag
axes[1, 1].plot(zig_zag_df)
axes[1, 1].set_title('(d) Zig-Zag')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Position Error')
axes[1, 1].legend(zig_zag_df.columns)
axes[1, 1].grid(True)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('position_error_plots.png')

print("Generated plot saved as position_error_plots.png")