from DataPreTreatment import DataPreTreatment
import os
import yaml
import pandas as pd

# Paths
cdir = os.getcwd()

path_ls_curated = os.path.abspath(os.path.join(cdir, "..", "data", "raw", "loadsol"))
path_fp_curated = os.path.abspath(
    os.path.join(cdir, "..", "data", "raw", "forceplates")
)

path_trials = os.path.abspath(os.path.join(cdir, "../", "data", "metadata"))

path_indexes_synchro_cut = os.path.abspath(
    os.path.join(cdir, "pre_processing", "utils")
)

# Open file containing indexes for synchro and cutting
with open(
    os.path.abspath(os.path.join(path_indexes_synchro_cut, "indexes_synchro.yaml")), "r"
) as f:
    indexes_synchro = yaml.safe_load(f)

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

trial_test = "S14_T01"
SESSION = "session_2"

# Create DataPreTreatment object
data_ls_fp = DataPreTreatment(
    ls_path=path_ls_curated,
    ls_filename=trial_test + "_ls",
    ls_frequency=200,
    insoles=insoles_correspondance["S14"]["insole_code"],
    ls_state="curated",
    fp_path=path_fp_curated,
    fp_filename=trial_test + "_fp",
    fp_frequency=1000,
    forceplates=fp_number[SESSION],
    fp_state="curated",
)

# data_ls_fp.data_ls.fill_missing_data()
# data_ls_fp.data_fp.pre_process_data()

data_ls_fp.synchro_LS_FP(
    trial="S14T01",
    ls_state="curated",
    fp_state="curated",
    path_indexes=os.path.abspath(
        os.path.join(path_indexes_synchro_cut, "indexes_synchro.yaml")
    ),
)

data_ls_fp.plot_synchro_data(time=True)

print()
