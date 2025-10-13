"""
Runs the already-tuned *kinematic bicycle* model with
      • wheel-base   **L = 1.0 m**
      • steer gain   **Kδ = 1.38**
      • steer lag    **τδ = 0.028 s**
      • drag coeff   **Cd = 3.0 e-5 s m⁻¹**
Prints RMSE metrics.
Produces **one figure with three subplots** and saves it as a file:
   XY trajectory, speed-vs-time, and yaw-vs-time.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- PARAMETERS -----------------------
L         = 1.0       # wheel-base [m]
K_DELTA   = 1.38      # steering gain (column → road-wheel)
TAU_DELTA = 0.028     # steering first-order lag [s]
CD        = 3.0e-5    # quadratic drag [s/m]


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle or array of angles to the range [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def load_csv(path: Path) -> Tuple[np.ndarray, ...]:
    """Return t, pos_x, pos_y, yaw, speed, acc_cmd, steer_cmd (rad)."""
    df = pd.read_csv(path)
    steer_rad = np.deg2rad(df["steer_deg"].values) * K_DELTA
    return (
        df["time"].values,
        df["pos_x"].values,
        df["pos_y"].values,
        df["yaw"].values,
        df["speed"].values,
        df["acceleration"].values,
        steer_rad,
    )


def simulate(
    t: np.ndarray,
    pos_x0: float,
    pos_y0: float,
    yaw0: float,
    speed0: float,
    acc_cmd: np.ndarray,
    steer_cmd: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    
    dt = float(np.mean(np.diff(t)))
    N = t.size

    sx = np.empty(N)
    sy = np.empty(N)
    syaw = np.empty(N)
    sv = np.empty(N)

    x, y, psi, v = float(pos_x0), float(pos_y0), float(yaw0), float(speed0)
    delta = 0.0

    for k in range(N):
        sx[k], sy[k], syaw[k], sv[k] = x, y, psi, v

        # steering actuator: τ δ̇ + δ = δ_cmd
        delta += (dt / TAU_DELTA) * (steer_cmd[k] - delta)

        # kinematic bicycle step
        x   += v * math.cos(psi) * dt
        y   += v * math.sin(psi) * dt
        psi += (v / L) * math.tan(delta) * dt
        v   += (acc_cmd[k] - CD * v * v) * dt

        # Wrap the yaw angle to the [-pi, pi] range
        psi = wrap_to_pi(psi)

    return sx, sy, syaw, sv


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(csv_path: str):
    (
        t,
        px,
        py,
        yaw,
        speed,
        acc_cmd,
        steer_cmd,
    ) = load_csv(Path(csv_path))

    yaw = wrap_to_pi(yaw)

    sx, sy, syaw, sv = simulate(
        t,
        px[0],
        py[0],
        yaw[0],
        speed[0],
        acc_cmd,
        steer_cmd,
    )

    print("RMSE position [m] :", round(rmse(np.hypot(px - sx, py - sy), np.zeros_like(px)), 3))
    print("RMSE yaw [rad]    :", round(rmse(yaw, syaw), 3))
    print("RMSE speed [m/s]  :", round(rmse(speed, sv), 3))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Real vs. Simulated Vehicle Dynamics', fontsize=16)

    # Subplot 1: Trajectory (top, spanning both columns)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(px, py, label="Real", lw=2)
    ax1.plot(sx, sy, "--", label="Simulated", lw=2)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajectory Comparison")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Speed (bottom-left)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.plot(t, speed, label="Real speed", lw=2)
    ax2.plot(t, sv, "--", label="Sim speed", lw=2)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("speed [m/s]")
    ax2.set_title("Speed Comparison")
    ax2.grid(True)
    ax2.legend()

    # Subplot 3: Yaw (bottom-right)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.plot(t, yaw, label="Real yaw", lw=2)
    ax3.plot(t, syaw, "--", label="Sim yaw", lw=2)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("yaw [rad]")
    ax3.set_title("Yaw Comparison")
    ax3.set_ylim(-math.pi - 0.5, math.pi + 0.5)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for the suptitle
    output_filename = "simulation_comparison.png"
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    # csv = sys.argv[1] if len(sys.argv) > 1 else "rosbag_aligned_data.csv"
    csv = sys.argv[1] if len(sys.argv) > 1 else "/home/sitong/Digital_Twin-Residual_Learning/residual_learning/test_data/u.csv"
    main(csv)