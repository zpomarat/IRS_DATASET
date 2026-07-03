import sys
import os

cdir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(cdir)))

from shared.DataLoadsol import DataLoadsol
from shared.DataForceplates import DataForceplates

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate, resample, butter, filtfilt
import pandas as pd
from copy import deepcopy
from scipy.interpolate import interp1d
from math import ceil
import yaml


class DataPreTreatment:
    def __init__(
        self,
        ls_path: str,
        ls_filename: str,
        ls_frequency: int,
        insoles: list,
        ls_state: str,
        fp_path: str,
        fp_filename: str,
        fp_frequency: int,
        forceplates: list,
        fp_state: str
    ):
        """Creates a DataPreTreatment object containing pre-processed data from the insoles and the forceplates.

        Args:
            ls_path (str): path of the directory containing Loadsol data
            ls_filename (str): name of the file containing Loadsol data
            ls_frequency (int): frequency of the Loadsol signal
            fp_path (str): path of the directory containing Forceplates data
            fp_filename (str): name of the file containing Forceplates data
            fp_frequency (int): frequency of the Forceplates data
        """

        self.trial = ls_filename

        if ls_path is not None:
            self.data_ls = DataLoadsol(
                dir_path=ls_path,
                file_name=ls_filename,
                frequency=ls_frequency,
                insoles=insoles,
            )
            match ls_state:
                case "curated":
                    self.data_ls.read_csv("curated")
                case "filled":
                    self.data_ls.read_csv("raw")
                    self.data_ls.fill_missing_data()
                    self.data_ls.extract_timestamp()

        if fp_path is not None:
            self.data_fp = DataForceplates(
                dir_path=fp_path,
                file_name=fp_filename,
                frequency=fp_frequency,
                fp_number=forceplates,
            )

            match fp_state:
                case "curated":
                    self.data_fp.read_csv("curated")
                case "pre_processed":
                    self.data_fp.pre_process_data()
                    self.data_fp.extract_timestamp()

    def plot_pre_processed_data(self, time: bool):
        """Plot the signals of interest of both insoles and forceplates data to find the indexes of interest.

        Args:
            time (bool): True if x axis corresponds to the time, False if x-axis corresponds to the frame
        """

        fig, axs = plt.subplots(2, 1)
        fig.suptitle(f"Pre-processed insoles and forceplates data ({self.trial})")

        match time:
            case False:
                axs[0].set_xlabel("Frame")
                axs[0].set_ylabel("Force [N]")
                axs[1].set_xlabel("Frame")
                axs[1].set_ylabel("Force [N]")

                axs[0].plot(
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_f_total_r"],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_f_total_l"],
                    linewidth=1,
                    label="f_tot_l",
                )

                axs[1].plot(
                    self.data_fp.pre_processed_data["fz5"], linewidth=1, label="fz_r"
                )
                axs[1].plot(
                    self.data_fp.pre_processed_data["fz2"], linewidth=1, label="fz_l"
                )

            case True:
                axs[0].set_xlabel("Time [s]")
                axs[0].set_ylabel("Force [N]")
                axs[1].set_xlabel("Time [s]")
                axs[1].set_ylabel("Force [N]")

                axs[0].plot(
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_f_total_r"],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.filled_data[self.data_ls.insoles[0] + "_f_total_l"],
                    linewidth=1,
                    label="f_tot_l",
                )

                axs[1].plot(
                    self.data_fp.pre_processed_data["time"],
                    self.data_fp.pre_processed_data["fz5"],
                    linewidth=1,
                    label="fz_r",
                )
                axs[1].plot(
                    self.data_fp.pre_processed_data["time"],
                    self.data_fp.pre_processed_data["fz2"],
                    linewidth=1,
                    label="fz_l",
                )

        axs[0].legend()
        axs[1].legend()
        plt.show()

    def synchro_LS_FP(self, ls_state: str, fp_state, idx_synchro_ls:int, idx_synchro_fp:int):
        """Synchronizes the Loadsol signals on the Forceplates signals based on indexes of interest previously identified.

        Args:
            trial (str): name of the trial to synchro (the name must correspond to that indicated in the yaml file containing the indexes of interest)
        """

        # Define index of interest
        ls_idx = int(idx_synchro_ls)
        fp_idx = int(idx_synchro_fp)

        coarse_shift = int(fp_idx / 5 - ls_idx)

        # Define a range of interest (+/-0.5s) around the point of interest and create a new reduced signal on this range
        if ls_state == "filled" and fp_state == "pre_processed":
            ls_reduced = deepcopy(
                self.data_ls.filled_data["f_total_r"][
                    ls_idx - 100 : ls_idx + 100
                ]
            ).values

            fp_reduced = deepcopy(
                self.data_fp.pre_processed_data["fz_r"][fp_idx - 500 : fp_idx + 500]
            ).values
        elif ls_state == "curated" and fp_state == "curated":
            ls_reduced = deepcopy(
                self.data_ls.raw_data["f_total_r"][
                    ls_idx - 100 : ls_idx + 100
                ]
            ).values

            fp_reduced = deepcopy(
                self.data_fp.raw_data["fz_r"][fp_idx - 500 : fp_idx + 500]
            ).values

        # Normaliser les deux signaux entre 0 et 1
        ls_reduced = (ls_reduced - np.min(ls_reduced)) / (np.max(ls_reduced) - np.min(ls_reduced))
        fp_reduced = (fp_reduced - np.min(fp_reduced)) / (np.max(fp_reduced) - np.min(fp_reduced))

        # Resample the LS reduced signal to the frequency of the FP signal
        n_samples = int(len(ls_reduced) * (1000 / 200))
        ls_resampled = resample(ls_reduced, n_samples)

        # Compute the optimal shift by cross-correlation
        corr = correlate(ls_resampled, fp_reduced, mode="full")
        lags = np.arange(-len(fp_reduced) + 1, len(fp_reduced))

        optimal_shift_fp = int(lags[np.argmax(corr)])
        optimal_shift_ls = int(np.round(optimal_shift_fp * 200 / 1000))

        self.total_shift = coarse_shift + optimal_shift_ls

        # Create a new synchronized signal
        if ls_state == "filled" and fp_state == "pre_processed":
            self.data_ls.data_synchro = deepcopy(self.data_ls.filled_data)
            self.data_fp.data_synchro = deepcopy(self.data_fp.pre_processed_data)
        elif ls_state == "curated" and fp_state == "curated":
            self.data_ls.data_synchro = deepcopy(self.data_ls.raw_data)
            self.data_fp.data_synchro = deepcopy(self.data_fp.raw_data)

        # Create a matrix with zeros of the length of nb
        data_to_add = np.zeros((self.total_shift, self.data_ls.data_synchro.shape[1]))

        # Convert it to a DataFrame
        data_to_add = pd.DataFrame(
            data_to_add, columns=self.data_ls.data_synchro.keys()
        )

        # Add data to the beginning of the signal
        self.data_ls.data_synchro = pd.concat(
            [data_to_add, self.data_ls.data_synchro], ignore_index=True
        )

        # Compute new time        
        self.data_ls.data_synchro["time"] = self.data_ls.data_synchro.index/self.data_ls.frequency

    def plot_synchro_data(self, time: bool):
        """Plot the signals of interest of both insoles and forceplates data after synchronization and downsampling."""

        fig, axs = plt.subplots(2, 1)
        fig.suptitle(f"Synchronized insoles and forceplates data ({self.trial})")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Force [N]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Force [N]")

        match time:
            case True:
                axs[0].plot(
                    self.data_ls.data_synchro["time"],
                    self.data_ls.data_synchro["f_total_r"],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_fp.data_synchro["time"],
                    self.data_fp.data_synchro["fz_r"],
                    linewidth=1,
                    label="fz_r",
                )

                axs[1].plot(
                    self.data_ls.data_synchro["time"],
                    self.data_ls.data_synchro["f_total_l"],
                    linewidth=1,
                    label="f_tot_l",
                )
                axs[1].plot(
                    self.data_fp.data_synchro["time"],
                    self.data_fp.data_synchro["fz_l"],
                    linewidth=1,
                    label="fz_l",
                )

            case False:
                axs[0].plot(
                    self.data_ls.data_synchro["f_total_r"],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_fp.downsampled_data["fz_r"],
                    linewidth=1,
                    label="fz_r",
                )

                axs[1].plot(
                    self.data_ls.data_synchro["f_total_l"],
                    linewidth=1,
                    label="f_tot_l",
                )
                axs[1].plot(
                    self.data_fp.downsampled_data["fz_l"],
                    linewidth=1,
                    label="fz_l",
                )

        axs[0].legend()
        axs[1].legend()
        plt.show()

    def downsample(self, signal: str, final_frequency: int):
        """Downsamples data to the final frequency.

        Args:
            signal (str): signal to be downsampled ("LS" or "FP")
            final_frequency (int): downsampling frequency
        """

        # Initialise downsampled data
        downsampled_data = pd.DataFrame()

        # Create new time vector based on the final frequency
        t_ds = np.arange(
            self.data_fp.data_synchro["time"].iloc[0],
            self.data_fp.data_synchro["time"].iloc[-1],
            1 / final_frequency,
        )

        for key in self.data_fp.data_synchro.keys():

            # Create interpolation function
            f = interp1d(
                self.data_fp.data_synchro["time"],
                self.data_fp.data_synchro.get(key),
            )

            # Downsample data
            downsampled_data[key] = f(t_ds)

        match signal:
            case "LS":
                self.data_ls.downsampled_data = downsampled_data
                self.data_ls.final_frequency = final_frequency
            case "FP":
                self.data_fp.downsampled_data = downsampled_data
                self.data_fp.final_frequency = final_frequency

    def plot_downsampled_data(self):
        """Plot the signals of interest of both insoles and forceplates data after synchronization and downsampling."""

        plt.title(f"Downsampled forceplates data ({self.trial})")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")

        plt.plot(
            self.data_fp.pre_processed_data["time"],
            self.data_fp.pre_processed_data["fz5"],
            "-o",
            linewidth=1,
            label="fz",
        )
        plt.plot(
            self.data_fp.downsampled_data["time"],
            self.data_fp.downsampled_data["fz5"],
            "-x",
            linewidth=1,
            label="fz downsampled",
        )

        plt.legend()
        plt.show()

    def cut_signal(self,idx_start:int,idx_end:int,downsample_fp:str):
        """Cut the signal to keep only the part of interest (between the jumps used for synchro)

        Args:
            trial (str): Name of the trial to process.
        """

        # Cut signal based on the LS frequency
        start = idx_start
        end = idx_end

        t_start = (self.total_shift + start) / self.data_ls.frequency
        t_end = (self.total_shift + end) / self.data_ls.frequency

        ls_cut = deepcopy(
            self.data_ls.data_synchro[
                (self.total_shift + start) : (self.total_shift + end)
            ]
        )
        if downsample_fp == "True":
            fp_cut = deepcopy(
                self.data_fp.downsampled_data[
                    (self.data_fp.downsampled_data["time"] >= t_start) &
                    (self.data_fp.downsampled_data["time"] <= t_end)
                ]
            )
        else:
            fp_cut = deepcopy(
                self.data_fp.data_synchro[
                    (self.data_fp.data_synchro["time"] >= t_start) &
                    (self.data_fp.data_synchro["time"] <= t_end)
                ]
            )

        # Reset index
        ls_cut.reset_index(drop=True, inplace=True)
        fp_cut.reset_index(drop=True, inplace=True)

        ls_cut["time"] = ls_cut["time"] - ls_cut["time"].iloc[0]
        fp_cut["time"] = fp_cut["time"] - fp_cut["time"].iloc[0]

        self.data_ls.cut_data = ls_cut
        self.data_fp.cut_data = fp_cut

        # Suppress time_l and r
        self.data_ls.cut_data.drop(columns=["time_r", "time_l"], inplace=True)

        # Put "time" column in first position
        cols = ["time"] + [col for col in self.data_ls.cut_data.columns if col != "time"]
        self.data_ls.cut_data = self.data_ls.cut_data[cols]

        # Cut longest signal to the size of the smallest one
        t_min_end = min(ls_cut["time"].iloc[-1], fp_cut["time"].iloc[-1])
        self.data_ls.cut_data = self.data_ls.cut_data[self.data_ls.cut_data["time"] <= t_min_end]
        self.data_fp.cut_data = self.data_fp.cut_data[self.data_fp.cut_data["time"] <= t_min_end]

    def cut_signal_thrust_only(self, idx_start:int,idx_end:int, downsample_fp:str):
        """Cut the signal to keep only the part of interest (corresponding to the thrust)

        Args:
            trial (str): Name of the trial to process.
        """

        # Cut signal based on the LS frequency
        start = idx_start
        end = idx_end

        t_start = start / self.data_ls.frequency
        t_end = end / self.data_ls.frequency

        ls_cut = deepcopy(self.data_ls.data_synchro[start:end])

        if downsample_fp == "True":
            fp_cut = deepcopy(
                self.data_fp.downsampled_data[
                    (self.data_fp.downsampled_data["time"] >= t_start) &
                    (self.data_fp.downsampled_data["time"] <= t_end)
                ]
            )
        else:
            fp_cut = deepcopy(
                self.data_fp.data_synchro[
                    (self.data_fp.data_synchro["time"] >= t_start) &
                    (self.data_fp.data_synchro["time"] <= t_end)
                ]
            )

        # Reset index
        ls_cut.reset_index(drop=True, inplace=True)
        fp_cut.reset_index(drop=True, inplace=True)

        ls_cut["time"] = ls_cut["time"] - ls_cut["time"].iloc[0]
        fp_cut["time"] = fp_cut["time"] - fp_cut["time"].iloc[0]

        self.data_ls.cut_data_thrust = ls_cut
        self.data_fp.cut_data_thrust = fp_cut

    def plot_cut_data(self, signal: str):
        """_summary_

        Args:
            trial (str): Name of the trial to process.
        """

        plt.figure()

        plt.title(f"Cut data ({self.trial})")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")

        match signal:
            case "whole":
                plt.plot(
                    self.data_ls.cut_data["time"],
                    self.data_ls.cut_data["f_total_r"],
                    label="f_tot_r",
                )
                plt.plot(
                    self.data_fp.cut_data["time"],
                    self.data_fp.cut_data["fz_r"],
                    label="fz_r",
                )

            case "thrust":
                plt.plot(
                    self.data_ls.cut_data_thrust["time"],
                    self.data_ls.cut_data_thrust["f_total_r"]
                    + self.data_ls.cut_data_thrust[
                        "f_total_l"
                    ],
                    label="ls",
                )
                plt.plot(
                    self.data_fp.cut_data_thrust["time"],
                    self.data_fp.cut_data_thrust["fz_r"]
                    + self.data_fp.cut_data_thrust["fz_l"],
                    label="fp",
                )

        plt.legend()
        plt.show()

    def filter(self, fs: int, signal: str, cut_type: str, order: int, fcut: int):
        """Apply a butterworth filter with a backward & forward pass.

        Args:
            signal (str): signal to be filtered ("LS" or "FP").
            cut_type (str): "whole" or "thrust".
            fs (int): sampling frequency of the signal (after downsampling).
            order (int): order of the filter. Note that with the backward & forward pass, this order will be multiplied by 2.
            fcut (int): Must be smaller than the half of the sampling frequency.
        """

        if fcut > fs:
            raise ValueError(
                "The cutting frequency should be lower than the downsampling frequency."
            )

        self.cut_type = cut_type

        # Initialise downsampled data and define keys associated
        match signal:
            case "LS":
                match cut_type:
                    case "whole":
                        filtered_data = deepcopy(self.data_ls.cut_data)
                        keys = [
                            key for key in filtered_data.keys() if "time" not in key
                        ]
                        self.data_fp.filtered_data = deepcopy(self.data_fp.cut_data)
                    case "thrust":
                        filtered_data = deepcopy(self.data_ls.cut_data_thrust)
                        keys = [
                            key for key in filtered_data.keys() if "time" not in key
                        ]
                        self.data_fp.filtered_data = deepcopy(
                            self.data_fp.cut_data_thrust
                        )

            case "FP":
                match cut_type:
                    case "whole":
                        filtered_data = deepcopy(self.data_fp.cut_data)
                        keys = [
                            key for key in filtered_data.keys() if "time" not in key
                        ]
                        self.data_ls.filtered_data = deepcopy(self.data_ls.cut_data)
                    case "thrust":
                        filtered_data = deepcopy(self.data_fp.cut_data_thrust)
                        keys = [
                            key for key in filtered_data.keys() if "time" not in key
                        ]
                        self.data_ls.filtered_data = deepcopy(
                            self.data_ls.cut_data_thrust
                        )

        Wn = fcut / (fs / 2)

        b, a = butter(order, Wn, analog=False)

        # Filter data
        for key in keys:
            filtered_data[key] = filtfilt(b, a, filtered_data[key])

        match signal:
            case "LS":
                self.data_ls.filtered_data = filtered_data
            case "FP":
                self.data_fp.filtered_data = filtered_data

    def plot_filtered_data(self):
        """Plot the signals of interest of both insoles and forceplates data after synchronization and downsampling."""

        fig, axs = plt.subplots(2, 1)
        fig.suptitle(f"Filtered insoles and forceplates data ({self.trial})")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Force [N]")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Force [N]")

        match self.cut_type:
            case "whole":
                axs[0].plot(
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_f_total_r"],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_acc_x_r"] * 1000,
                    linewidth=1,
                    label="acc_x_r",
                )
                axs[0].plot(
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data[self.data_ls.insoles[0] + "_gyro_x_r"] * 1000,
                    linewidth=1,
                    label="gyro_x_r",
                )
                axs[1].plot(
                    self.data_fp.cut_data["time"],
                    self.data_fp.cut_data["fz2"],
                    linewidth=1,
                    label="fz_l",
                )

            case "thrust":
                axs[0].plot(
                    self.data_ls.cut_data_thrust[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data_thrust[
                        self.data_ls.insoles[0] + "_f_total_r"
                    ],
                    linewidth=1,
                    label="f_tot_r",
                )
                axs[0].plot(
                    self.data_ls.cut_data_thrust[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data_thrust[self.data_ls.insoles[0] + "_acc_x_r"]
                    * 1000,
                    linewidth=1,
                    label="acc_x_r",
                )
                axs[0].plot(
                    self.data_ls.cut_data_thrust[self.data_ls.insoles[0] + "_time"],
                    self.data_ls.cut_data_thrust[self.data_ls.insoles[0] + "_gyro_x_r"]
                    * 1000,
                    linewidth=1,
                    label="gyro_x_r",
                )
                axs[1].plot(
                    self.data_fp.cut_data_thrust["time"],
                    self.data_fp.cut_data_thrust["fz2"],
                    linewidth=1,
                    label="fz_l",
                )

        axs[0].plot(
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_time"],
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_f_total_r"],
            linewidth=1,
            label="f_tot_r filtered",
        )

        axs[0].plot(
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_time"],
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_acc_x_r"] * 1000,
            linewidth=1,
            label="acc_x_r filtered",
        )
        axs[0].plot(
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_time"],
            self.data_ls.filtered_data[self.data_ls.insoles[0] + "_gyro_x_r"] * 1000,
            linewidth=1,
            label="gyro_x_r filtered",
        )
        axs[1].plot(
            self.data_fp.filtered_data["time"],
            self.data_fp.filtered_data["fz2"],
            linewidth=1,
            label="fz_l filtered",
        )

        axs[0].legend()
        axs[1].legend()
        plt.show()

    def plot_both_feet(self, trial: str):

        plt.figure()

        plt.title(f"Data on both feet ({self.trial})")
        plt.xlabel("Frame")
        plt.ylabel("Force [N]")

        plt.plot(
            self.data_ls.data_synchro[self.data_ls.insoles[0] + "_f_total_r"]
            + self.data_ls.data_synchro[self.data_ls.insoles[0] + "_f_total_l"],
            label="ls",
        )
        plt.plot(
            self.data_fp.downsampled_data["fz2"] + self.data_fp.downsampled_data["fz5"],
            label="fp",
        )

        plt.legend()
        plt.show()

    def merge_data(self):

        # Rename time column for insole data
        self.data_ls.filtered_data.rename(
            columns={self.data_ls.insoles[0] + "_time": "time"}, inplace=True
        )

        # Create a DataFrame which will contain all data (insoles + forceplates)
        data_merged = pd.merge(
            self.data_ls.filtered_data,
            self.data_fp.filtered_data,
            on="time",
            how="outer",
        )

        # Add column with the name of the trial
        data_merged["trial"] = self.trial

        # Rename columns
        data_merged.rename(
            columns={
                self.data_ls.insoles[0] + "_f_heel_r": "f_heel_r",
                self.data_ls.insoles[0] + "_f_medial_r": "f_medial_r",
                self.data_ls.insoles[0] + "_f_lateral_r": "f_lateral_r",
                self.data_ls.insoles[0] + "_f_total_r": "f_total_r",
                self.data_ls.insoles[0] + "_f_heel_l": "f_heel_l",
                self.data_ls.insoles[0] + "_f_medial_l": "f_medial_l",
                self.data_ls.insoles[0] + "_f_lateral_l": "f_lateral_l",
                self.data_ls.insoles[0] + "_f_total_l": "f_total_l",
                self.data_ls.insoles[0] + "_acc_x_r": "acc_x_r",
                self.data_ls.insoles[0] + "_acc_y_r": "acc_y_r",
                self.data_ls.insoles[0] + "_acc_z_r": "acc_z_r",
                self.data_ls.insoles[0] + "_acc_x_l": "acc_x_l",
                self.data_ls.insoles[0] + "_acc_y_l": "acc_y_l",
                self.data_ls.insoles[0] + "_acc_z_l": "acc_z_l",
                self.data_ls.insoles[0] + "_gyro_x_r": "gyro_x_r",
                self.data_ls.insoles[0] + "_gyro_y_r": "gyro_y_r",
                self.data_ls.insoles[0] + "_gyro_z_r": "gyro_z_r",
                self.data_ls.insoles[0] + "_gyro_x_l": "gyro_x_l",
                self.data_ls.insoles[0] + "_gyro_y_l": "gyro_y_l",
                self.data_ls.insoles[0] + "_gyro_z_l": "gyro_z_l",
                "fx2": "fx_l",
                "fy2": "fy_l",
                "fz2": "fz_l",
                "fx5": "fx_r",
                "fy5": "fy_r",
                "fz5": "fz_r",
            },
            inplace=True,
        )

        # Reorganize columns
        data_merged = data_merged[
            [
                "trial",
                "time",
                "f_heel_r",
                "f_medial_r",
                "f_lateral_r",
                "f_total_r",
                "acc_x_r",
                "acc_y_r",
                "acc_z_r",
                "gyro_x_r",
                "gyro_y_r",
                "gyro_z_r",
                "f_heel_l",
                "f_medial_l",
                "f_lateral_l",
                "f_total_l",
                "acc_x_l",
                "acc_y_l",
                "acc_z_l",
                "gyro_x_l",
                "gyro_y_l",
                "gyro_z_l",
                "fx_r",
                "fy_r",
                "fz_r",
                "fx_l",
                "fy_l",
                "fz_l"
            ]
        ]

        self.merged_data = data_merged

    def export_processed_data(self, data: pd.DataFrame, path: str, name: str):
        """Exports the DataFrame containing the curated data to a csv file.

        Args:
            path (str): path of the directory to save the csv file.
            name (str): name of the csv file to save.
        """

        data.to_csv(os.path.abspath(os.path.join(path, name + ".csv")))