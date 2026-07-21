# This script was used to manually identify synchronization indices
# by visually inspecting force signals and detecting the tap/jump event peak.
# Identified indices are stored in metadata/trials.csv (idx_sync_* columns).


import sys
import os

cdir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(cdir)))

from shared.DataLoadsol import DataLoadsol
from shared.DataForceplates import DataForceplates
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Paths
cdir = os.getcwd()
path_loadsol = os.path.abspath(os.path.join(cdir, "../", "data", "raw", "loadsol"))
path_forceplates = os.path.abspath(
    os.path.join(cdir, "../", "data", "raw", "forceplates")
)
path_trials = os.path.abspath(os.path.join(cdir, "../", "data", "metadata"))

# Load trial list
trials = pd.read_csv(
    os.path.abspath(os.path.join(path_trials, "trials.csv")),
    sep=",",
    header=0,
    na_values="-",
    dtype=str,
)

# Load insoles codes
with open(
    os.path.abspath(os.path.join(cdir, "shared", "insoles_correspondance.yaml")), "r"
) as f:
    insoles_correspondance = yaml.safe_load(f)

# Forceplates numbers used during each session [left,right]
fp_number = {"session_1": [1, 2], "session_2": [2, 5]}

# Create empty dictionnaries containing objets
ls_dict = {}
fp_dict = {}

for _, trial in trials.iterrows():

    # Loadsol data
    if trial.loadsol_available == "no":
        print(f"WARNING: no loadsol file for {trial.subject_id}_{trial.trial_id}")
        continue
    else:
        # Create loadsol object
        ls_dict[f"{trial.subject_id}_{trial.trial_id}"] = DataLoadsol(
            dir_path=path_loadsol,
            file_name=f"{trial.subject_id}_{trial.trial_id}_ls",
            frequency=200,
            insoles=insoles_correspondance[trial.subject_id]["insole_code"],
        )

        # Read raw data
        ls_dict[f"{trial.subject_id}_{trial.trial_id}"].read_csv(state="curated")

    
    # Forceplates data
    if trial.fp_available == "no":
        print(f"WARNING: no forceplates file for {trial.subject_id}_{trial.trial_id}")
        continue
    else:
        # Create forceplates object
        fp_dict[f"{trial.subject_id}_{trial.trial_id}"] = DataForceplates(
            dir_path=path_forceplates,
            file_name=f"{trial.subject_id}_{trial.trial_id}_fp",
            frequency=1000,
            fp_number=fp_number[f"session_{trial.session}"],
        )

        # Read raw data
        fp_dict[f"{trial.subject_id}_{trial.trial_id}"].read_csv(state="curated")

# Plot raw loadsol and forceplates data
def plot_loadsol_data(loadsol_data: pd.DataFrame, trial_name: str):
    """
    Plot Loadsol insole data (forces, accelerations, angular velocities)
    for left and right foot.

    Args:
        loadsol_data: DataFrame containing Loadsol data.
        trial_name: Trial name for the figure title (optional).
    """
    # Data to plot per row: (columns_suffix, labels, ylabel)
    rows = [
        (
            ["f_heel", "f_medial", "f_lateral", "f_total"],
            ["Heel", "Medial", "Lateral", "Total"],
            "Force (N)"
        ),
        (
            ["acc_x", "acc_y", "acc_z"],
            ["Acc x", "Acc y", "Acc z"],
            "Acceleration (g)"
        ),
        (
            ["gyro_x", "gyro_y", "gyro_z"],
            ["Gyro x", "Gyro y", "Gyro z"],
            "Angular velocity (rad/s)"
        ),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Loadsol data — {trial_name}")

    for row_idx, (columns, labels, ylabel) in enumerate(rows):
        for col_idx, side in enumerate(["l", "r"]):
            ax = axs[row_idx, col_idx]
            for col, label in zip(columns, labels):
                ax.plot(loadsol_data[f"{col}_{side}"], label=label)
            ax.set_xlabel("Frame")
            ax.set_ylabel(ylabel)
            ax.legend()
            if row_idx == 0:
                ax.set_title("Left foot" if side == "l" else "Right foot")

    fig.tight_layout()


def plot_fp_data(fp_data: pd.DataFrame, trial_name: str):
    """
    Plot force plate data (forces) for left and right foot.

    Args:
        fp_data: DataFrame containing force plate data.
        trial_name: Trial name for the figure title.
    """
    # Data to plot per row: (columns_suffix, labels, ylabel)
    rows = [
        (
            ["fx", "fy", "fz"],
            ["Fx (L)", "Fy (AP)", "Fz (V)"],
            "Force (N)"
        ),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 ligne, 2 colonnes
    fig.suptitle(f"Forceplates data — {trial_name}")

    for _, (columns, labels, ylabel) in enumerate(rows):
        for col_idx, side in enumerate(["l", "r"]):
            ax = axs[col_idx]
            for col, label in zip(columns, labels):
                ax.plot(fp_data[f"{col}_{side}"], label=label)
            ax.set_xlabel("Frame")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.set_title("Left foot" if side == "l" else "Right foot")

    fig.tight_layout()

# Exclude ls_S08_T04: almost half of the trial is missing for left insole
trial_to_exclude= "S08_T04"

for _, trial in trials.iterrows():
    trial_name = f"{trial.subject_id}_{trial.trial_id}"

    # Plot forceplates si disponible
    if trial_name in fp_dict:
        plot_fp_data(fp_data=fp_dict[trial_name].raw_data, trial_name=trial_name)

    # Plot loadsol si disponible
    if trial_name in ls_dict and trial_name != trial_to_exclude:
        plot_loadsol_data(loadsol_data=ls_dict[trial_name].raw_data, trial_name=trial_name)

    plt.show()
