# This script was used to manually identify synchronization indices
# by visually inspecting force signals and detecting the tap event peak.
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



# # for trial in list_of_trials:
#     # Create DataLoadsol and DataForceplates objects
#     if trial not in ls_fp_to_exclude:
#         ls_dict[trial] = DataLoadsol(
#             dir_path=path_insoles, file_name=trial, frequency=200
#         )
#         fp_dict[trial] = DataForceplates(
#             dir_path=path_forceplates, file_name=trial, file_type="csv", frequency=1000
#         )

#         # Pre-treat DataLoadsol objects
#         ls_dict[trial].fill_missing_data()

#         # Pre-treat DataForceplates objects
#         fp_dict[trial].pre_process_data()


# for trial in ls_dict.keys():
#     fig1, axs = plt.subplots(3, 2)

#     # Plot insoles data
#     fig1.suptitle(f"Pre-treated insoles data ({trial})")
#     axs[0, 0].plot(ls_dict[trial].filled_data["f_heel_l"], label="f heel")
#     axs[0, 0].plot(ls_dict[trial].filled_data["f_medial_l"], label="f medial")
#     axs[0, 0].plot(ls_dict[trial].filled_data["f_lateral_l"], label="f lateral")
#     axs[0, 0].plot(ls_dict[trial].filled_data["f_total_l"], label="f total")
#     axs[0, 0].set_xlabel("Frame")
#     axs[0, 0].set_ylabel("Force (N)")
#     axs[0, 0].legend()
#     axs[0, 0].set_title("Left")

#     axs[1, 0].plot(ls_dict[trial].filled_data["acc_x_l"], label="acc x")
#     axs[1, 0].plot(ls_dict[trial].filled_data["acc_y_l"], label="acc y")
#     axs[1, 0].plot(ls_dict[trial].filled_data["acc_z_l"], label="acc z")
#     axs[1, 0].set_xlabel("Frame")
#     axs[1, 0].set_ylabel("Acceleration (g)")
#     axs[1, 0].legend()

#     axs[2, 0].plot(ls_dict[trial].filled_data["gyro_x_l"], label="gyro x")
#     axs[2, 0].plot(ls_dict[trial].filled_data["gyro_y_l"], label="gyro y")
#     axs[2, 0].plot(ls_dict[trial].filled_data["gyro_z_l"], label="gyro z")
#     axs[2, 0].set_xlabel("Frame")
#     axs[2, 0].set_ylabel("Angular velocity (rad/s)")
#     axs[2, 0].legend()

#     axs[0, 1].plot(ls_dict[trial].filled_data["f_heel_r"], label="f heel")
#     axs[0, 1].plot(ls_dict[trial].filled_data["f_medial_r"], label="f medial")
#     axs[0, 1].plot(ls_dict[trial].filled_data["f_lateral_r"], label="f lateral")
#     axs[0, 1].plot(ls_dict[trial].filled_data["f_total_r"], "-o", label="f total")
#     axs[0, 1].set_xlabel("Frame")
#     axs[0, 1].set_ylabel("Force (N)")
#     axs[0, 1].legend()
#     axs[0, 1].set_title("Right")

#     axs[1, 1].plot(ls_dict[trial].filled_data["acc_x_r"], "-o", label="acc x")
#     axs[1, 1].plot(ls_dict[trial].filled_data["acc_y_r"], "-o", label="acc y")
#     axs[1, 1].plot(ls_dict[trial].filled_data["acc_z_r"], "-o", label="acc z")
#     axs[1, 1].set_xlabel("Frame")
#     axs[1, 1].set_ylabel("Acceleration (g)")
#     axs[1, 1].legend()

#     axs[2, 1].plot(ls_dict[trial].filled_data["gyro_x_r"], "-o", label="gyro x")
#     axs[2, 1].plot(ls_dict[trial].filled_data["gyro_y_r"], "-o", label="gyro y")
#     axs[2, 1].plot(ls_dict[trial].filled_data["gyro_z_r"], "-o", label="gyro z")
#     axs[2, 1].set_xlabel("Frame")
#     axs[2, 1].set_ylabel("Angular velocity (rad/s)")
#     axs[2, 1].legend()

#     fig1.tight_layout()

#     # Plot insoles data
#     fig2, axs = plt.subplots(3, 2)
#     fig2.suptitle(f"Pre-treated forceplates data ({trial})")
#     axs[0, 0].plot(fp_dict[trial].pre_processed_data["fx1"], label="fx")
#     axs[0, 0].set_xlabel("Frame")
#     axs[0, 0].set_ylabel("Force (N)")
#     axs[0, 0].set_title("Left")
#     axs[0, 0].legend()

#     axs[1, 0].plot(fp_dict[trial].pre_processed_data["fy1"], label="fy")
#     axs[1, 0].set_xlabel("Frame")
#     axs[1, 0].set_ylabel("Force (N)")
#     axs[1, 0].legend()

#     axs[2, 0].plot(fp_dict[trial].pre_processed_data["fz1"], label="fz")
#     axs[2, 0].set_xlabel("Frame")
#     axs[2, 0].set_ylabel("Force (N)")
#     axs[2, 0].legend()

#     axs[0, 1].plot(fp_dict[trial].pre_processed_data["fx2"], "-o", label="fx")
#     axs[0, 1].set_xlabel("Frame")
#     axs[0, 1].set_ylabel("Force (N)")
#     axs[0, 1].set_title("Right")
#     axs[0, 1].legend()

#     axs[1, 1].plot(fp_dict[trial].pre_processed_data["fy2"], label="fy")
#     axs[1, 1].set_xlabel("Frame")
#     axs[1, 1].set_ylabel("Force (N)")
#     axs[1, 1].legend()

#     axs[2, 1].plot(fp_dict[trial].pre_processed_data["fz2"], "-o", label="fz")
#     axs[2, 1].set_xlabel("Frame")
#     axs[2, 1].set_ylabel("Force (N)")
#     axs[2, 1].legend()
#     fig2.tight_layout()
#     plt.show()


# print("ok")
