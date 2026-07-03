import sys
import os

cdir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(cdir)))

from shared.DataForceplates import DataForceplates
import pandas as pd
import csv

# Choose session
SESSION = "session_1"

# Paths
path_forceplates = {
    "session_1": "D://DATA//MANIPS_SESSION_1//forceplates//",
    "session_2": "D://DATA//MANIPS_SESSION_2//forceplates//",
}
path_trials = os.path.abspath(os.path.join(cdir, "../", "data", "metadata"))
path_output = os.path.abspath(os.path.join(cdir, "..", "data", "raw", "forceplates"))
log_path = os.path.abspath(os.path.join(path_trials, f"fp_frames_log_{SESSION}.csv"))

# Load trial list
trials = pd.read_csv(
    os.path.abspath(os.path.join(path_trials, "trials.csv")),
    sep=",",
    header=0,
    na_values="-",
    dtype=str,
)

# Load files names correspondance
names = pd.read_csv(
    os.path.abspath(
        os.path.join(cdir, "data_curation", "utils", "files_names_correspondance.csv")
    ),
    sep=",",
    header=0,
    na_values="-",
    dtype=str,
)

# Forceplates numbers used during each session [left,right]
fp_number = {"session_1": [1, 2], "session_2": [2, 5]}

# Filter trials for current session only
trials_session = trials[trials["session"] == SESSION[-1]]

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "trial_id", "n_frames_fp"])

# Main loop
for _, trial in trials_session.iterrows():

    # Original file name corresponding to the actual trial
    row = names[
        (names["subject_id"] == trial.subject_id)
        & (names["trial_id"] == trial.trial_id)
    ]

    if trial.fp_available == "no":
        print(f"WARNING: no forceplates file for {trial.subject_id}_{trial.trial_id}")
        continue
    else:
        original_filename = row["original_filename"].values[0]

        # Create forceplates object
        forceplates = DataForceplates(
            dir_path=path_forceplates[SESSION],
            file_name=f"{original_filename}",
            frequency=1000,
            fp_number=fp_number[SESSION],
        )

        # Read csv
        forceplates.read_csv()

        # GRF
        forceplates.pre_process_data()

        # Unify axes
        match SESSION:
            case "session_1":
                unified_forceplates = {
                    "time": forceplates.pre_processed_data["time"],
                    f"fx{fp_number[SESSION][0]}": - forceplates.pre_processed_data[f"fy{fp_number[SESSION][0]}"],
                    f"fy{fp_number[SESSION][0]}": forceplates.pre_processed_data[f"fx{fp_number[SESSION][0]}"],
                    f"fz{fp_number[SESSION][0]}": forceplates.pre_processed_data[f"fz{fp_number[SESSION][0]}"],
                    f"fx{fp_number[SESSION][1]}": - forceplates.pre_processed_data[f"fy{fp_number[SESSION][1]}"],
                    f"fy{fp_number[SESSION][1]}": forceplates.pre_processed_data[f"fx{fp_number[SESSION][1]}"],
                    f"fz{fp_number[SESSION][1]}": forceplates.pre_processed_data[f"fz{fp_number[SESSION][1]}"],
                }
                unified_forceplates = pd.DataFrame(unified_forceplates)

        # Rename columns
        match SESSION:
            case "session_1":
                unified_forceplates.rename(
            columns={
                f"fx{fp_number[SESSION][0]}": "fx_l",
                f"fy{fp_number[SESSION][0]}": "fy_l",
                f"fz{fp_number[SESSION][0]}": "fz_l",
                f"fx{fp_number[SESSION][1]}": "fx_r",
                f"fy{fp_number[SESSION][1]}": "fy_r",
                f"fz{fp_number[SESSION][1]}": "fz_r",
            },
            inplace=True,
        )
            case "session_2":
                forceplates.pre_processed_data.rename(
                    columns={
                        f"fx{fp_number[SESSION][0]}": "fx_l",
                        f"fy{fp_number[SESSION][0]}": "fy_l",
                        f"fz{fp_number[SESSION][0]}": "fz_l",
                        f"fx{fp_number[SESSION][1]}": "fx_r",
                        f"fy{fp_number[SESSION][1]}": "fy_r",
                        f"fz{fp_number[SESSION][1]}": "fz_r",
                    },
                    inplace=True,
                )

        # Export csv rearranged
        match SESSION:
            case "session_1":
                forceplates.export_pre_treated_data(data=unified_forceplates,path=path_output,name=f"{trial.subject_id}_{trial.trial_id}_fp")
            case "session_2":
                forceplates.export_pre_treated_data(data=forceplates.pre_processed_data,path=path_output,name=f"{trial.subject_id}_{trial.trial_id}_fp")

        # Export the number of frames for each trial
        n_frames_fp = len(forceplates.raw_data)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial.subject_id, trial.trial_id, n_frames_fp])
