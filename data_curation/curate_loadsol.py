import sys
import os

cdir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(cdir)))

from shared.DataLoadsol import DataLoadsol
import pandas as pd
import yaml
import csv

# Choose session
SESSION = "session_1"

# Paths
path_loadsol = {"session_1": "D://DATA//MANIPS_SESSION_1//insoles//",
                "session_2": "D://DATA//MANIPS_SESSION_2//insoles//"}
path_trials = os.path.abspath(os.path.join(cdir, "../", "data", "metadata"))
path_output = os.path.abspath(os.path.join(cdir,"..","data","raw","loadsol"))
log_path = os.path.abspath(os.path.join(path_trials, f"loadsol_frames_log_{SESSION}.csv"))

# Load trial list
trials = pd.read_csv(
    os.path.abspath(os.path.join(path_trials, "trials.csv")),
    sep=",", header=0, na_values="-", dtype=str
)

# Load files names correspondance
names = pd.read_csv(
    os.path.abspath(os.path.join(cdir,"data_curation","utils", "files_names_correspondance.csv")),
    sep=",", header=0, na_values="-", dtype=str
)

# Load insoles codes
with open(os.path.abspath(os.path.join(cdir,"shared","insoles_correspondance.yaml")),"r") as f:
    insoles_correspondance = yaml.safe_load(f)

# Filter trials for current session only
trials_session = trials[trials["session"] == SESSION[-1]]

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "trial_id", "n_frames_loadsol"])

# Main loop
for _, trial in trials_session.iterrows():

    # Original file name corresponding to the actual trial
    row = names[
    (names["subject_id"] == trial.subject_id) & 
    (names["trial_id"] == trial.trial_id)
]

    if trial.loadsol_available=="no":
        print(f"WARNING: no loadsol file for {trial.subject_id}_{trial.trial_id}")
        continue
    else:
        original_filename = row["original_filename"].values[0]

        # Create loadsol object
        loadsol = DataLoadsol(dir_path=path_loadsol[SESSION],file_name=f"{original_filename}",frequency=200,insoles=insoles_correspondance[trial.subject_id]["insole_code"])

        # Convert txt into csv file
        loadsol.convert_txt_to_csv()

        # Read csv file
        loadsol.read_csv()

        # Export csv rearranged
        loadsol.export_pre_treated_data(data=loadsol.raw_data,path=path_output,name=f"{trial.subject_id}_{trial.trial_id}_ls")

        # Export the number of frames for each trial
        n_frames_ls = len(loadsol.raw_data)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial.subject_id, trial.trial_id, n_frames_ls])





