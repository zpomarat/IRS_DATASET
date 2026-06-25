import csv
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from copy import deepcopy
from scipy import signal
import yaml


class DataLoadsol:
    def __init__(self, dir_path: str, file_name: str, frequency: int, insoles: list):
        """_summary_

        Args:
            dir_path (str): path of the directory containing Loadsol files
            file_name (str): name of the Loadsol file
            frequency (int): frequency of the Loadsol data
            insoles (list): list of the insoles names
            indexes_cut (str): path of the yaml file containing the indexes of the different phases fot cutting
        """
        self.path = dir_path
        self.file_name = file_name
        self.path_txt = os.path.abspath(os.path.join(dir_path, self.file_name + ".txt"))
        self.path_csv = os.path.abspath(os.path.join(dir_path, self.file_name + ".csv"))
        self.csv_file_created = False
        self.missing_data = False
        self.frequency = frequency
        self.raw_data = None
        self.timestamp = None
        self.cleaned_data = None
        self.filled_data = None
        self.downsampled_data = None
        self.final_frequency = None
        self.filtered_data = None
        self.data_synchro = None
        self.data_m_synchro = None
        self.insoles = insoles

    def convert_txt_to_csv(self):
        """Converts a txt file to a csv file and change the attribute "path" to the DataLoadsol class by the path of csv file.

        Args:
            output_directory: str
                Path of the directory contaning the converted file.
        """

        if os.path.exists(self.path_txt) == False:
            raise ValueError("The txt file does not exist.")

        # Delimiter used in the input file
        input_delimiter = "\t"

        # Delimiter used in the output file
        output_delimiter = ","

        # Open the input file in read mode
        with open(self.path_txt, "r", newline="", encoding="utf-8") as infile:

            # Read the line containing columns names
            lines = infile.readlines()
            line = lines[2]

            # Check if data are missing
            line_elements = line.split()

            # If exists, remove the element "Notes"
            if "Notes" in line_elements:
                line_elements.remove("Notes")

            match len(self.insoles):
                case 1:
                    if len(line_elements) != 20:
                        self.path_csv = None
                        self.missing_data = True
                        raise ValueError(
                            "Data missing, the file '"
                            + self.file_name
                            + "' is not processed."
                        )
                case 2:
                    if len(line_elements) != 40:
                        self.path_csv = None
                        self.missing_data = True
                        raise ValueError(
                            "Data missing, the file '"
                            + self.file_name
                            + "' is not processed."
                        )
                case 3:
                    if len(line_elements) != 60:
                        self.path_csv = None
                        self.missing_data = True
                        raise ValueError(
                            "Data missing, the file '"
                            + self.file_name
                            + "' is not processed."
                        )

            # Line containing the timestamp
            self.line_ts = lines[0]

            # Create a CSV reader object for the input file
            reader = csv.reader(lines, delimiter=input_delimiter)

            # Read the data from the input file
            data = list(reader)

            output_filepath = self.path_csv

        infile.close()

        if os.path.exists(self.path_csv) == False:
            # Open the output file in write mode
            with open(output_filepath, "w", newline="", encoding="utf-8") as outfile:

                # Create a CSV writer object for the output file
                writer = csv.writer(outfile, delimiter=output_delimiter)

                # Write the data to the CSV file
                writer.writerows(data)

            outfile.close()

    def extract_timestamp(self):
        """Extracts the timestamp from the .txt file.

        timestamp = {date: YEAR-MONTH-DAY,
                    time: HOUR-MINUTE-SECOND-MICROSECOND}
        """

        if self.csv_file_created == False:
            self.convert_txt_to_csv()
            self.csv_file_created = True

        # Split the line to extract the part containing the timestamp
        txt = self.line_ts.split(".")

        # Select the item of txt containing the timestamp and split it
        ts = txt[0].split("_")

        # Extract timestamp for the old version of loadsol files (before May 2025)
        if len(ts) > 3:
            # Define the timestamp
            self.timestamp = datetime.datetime(
                int(ts[1]),
                int(ts[2]),
                int(ts[3]),
                int(ts[4]),
                int(ts[5]),
                int(ts[6]),
                int(ts[7]) * 1000,
            )

        else:
            # Extract timestamp for the new ersion of Loadsol files (from May 2025)
            date = ts[1].split("-")
            time = ts[2].split("-")

            # Define the timestamp
            self.timestamp = datetime.datetime(
                int(date[0]),
                int(date[1]),
                int(date[2]),
                int(time[0]),
                int(time[1]),
                int(time[2]),
            )

    def read_csv(self,state="raw"):
        """Reads csv file containing raw insoles data. Creates a pd.DataFrame containing the raw time and the raw data of each insole. Valid for ST players only (due to the force plates configuration during experiments at CREPS).

        Returns:
            self.raw_data: pd.DataFrame
                For each pair of insoles, contains raw time and raw data of both insoles of a pair.

            keys: time_left, f_heel_l, f_medial_l, f_lateral_l, f_tot_l, acc_x_l, acc_y_l, acc_z_l, gyro_x_l, gyro_y_l, gyro_z_l,
                  time_right, f_heel_r, f_medial_r, f_lateral_r, f_tot_r, acc_x_r, acc_y_r, acc_z_r, gyro_x_r, gyro_y_r, gyro_z_r.
        """

        if os.path.isfile(self.path_csv) == False and self.csv_file_created == False:
            self.convert_txt_to_csv()
            self.csv_file_created = True

        if os.path.isfile(self.path_csv) == False:
            raise ValueError("The csv file does not exist.")

        # Read the csv file as a Dataframe
        if state == "raw":
            data_csv = pd.read_csv(
                self.path_csv, sep=",", header=2, na_values="-", dtype=str
            )

            # Suppress the columns with only nan values
            for column in data_csv.columns:
                if data_csv[column].isna().all():
                    data_csv.drop(labels=column, axis=1, inplace=True)

            # Suppres the line with units
            data_csv.drop(0, axis=0, inplace=True)

            # Remove data after the end end of the measurement
            if "Notes" in data_csv.keys():
                idx_end_measurement = data_csv[
                    data_csv["Notes"] == "End measurement"
                ].index[0]
                data_csv.drop(
                    index=range(idx_end_measurement, len(data_csv)), axis=0, inplace=True
                )

                # Suppress the column "Notes" if it exists
                data_csv.drop(labels="Notes", axis=1, inplace=True)

            # Reset index
            data_csv.reset_index(drop=True, inplace=True)

            # Remove useless time columns (all the columns containing "Unnamed" that are not placed before a column containing "lateral" or "heel" or "medial")
            columns_list = list(data_csv.columns)

            col_to_suppress = []

            keywords = {"medial", "heel", "lateral"}

            # Columns to suppress are those that do not have a following column or whose following column does not contain a keyword
            for idx, col in enumerate(columns_list):
                if "Unnamed" in col:
                    if idx + 1 == len(columns_list) or not any(
                        keyword in columns_list[idx + 1] for keyword in keywords
                    ):
                        col_to_suppress.append(idx)

            data_csv.drop(data_csv.columns[col_to_suppress], axis=1, inplace=True)

            # Open and read the yaml file containing the correspondance between insoles names and codes
            cdir = os.getcwd()

            with open(
                os.path.abspath(os.path.join(cdir, "data_curation", "utils", "insoles_codes.yaml")), "r"
            ) as file:
                insoles_codes = yaml.safe_load(file)

            # Rename columns names (force and IMU data)
            for insole in self.insoles:
                for key in data_csv.keys():
                    match key:
                        case code_right if (
                            insoles_codes[insole]["code_right"] in code_right
                        ):
                            match code_right:
                                case lat if "lateral" in lat:
                                    data_csv.rename(
                                        columns={code_right: "f_lateral_r"},
                                        inplace=True,
                                    )
                                case med if "medial" in med:
                                    data_csv.rename(
                                        columns={code_right: "f_medial_r"},
                                        inplace=True,
                                    )
                                case heel if "heel" in heel:
                                    data_csv.rename(
                                        columns={code_right: "f_heel_r"},
                                        inplace=True,
                                    )
                                case tot if (
                                    insoles_codes[insole]["code_right"] in tot
                                    and ":" not in tot
                                ):
                                    data_csv.rename(
                                        columns={code_right: "f_total_r"},
                                        inplace=True,
                                    )
                                case xacc if "xAcc" in xacc:
                                    data_csv.rename(
                                        columns={code_right: "acc_x_r"},
                                        inplace=True,
                                    )

                                case yacc if "yAcc" in yacc:
                                    data_csv.rename(
                                        columns={code_right: "acc_y_r"},
                                        inplace=True,
                                    )
                                case zacc if "zAcc" in zacc:
                                    data_csv.rename(
                                        columns={code_right: "acc_z_r"},
                                        inplace=True,
                                    )
                                case xgyro if "xGyro" in xgyro:
                                    data_csv.rename(
                                        columns={code_right: "gyro_x_r"},
                                        inplace=True,
                                    )

                                case ygyro if "yGyro" in ygyro:
                                    data_csv.rename(
                                        columns={code_right: "gyro_y_r"},
                                        inplace=True,
                                    )
                                case zgyro if "zGyro" in zgyro:
                                    data_csv.rename(
                                        columns={code_right: "gyro_z_r"},
                                        inplace=True,
                                    )
                        case insole_right if (
                            insole in insole_right and "-R" in insole_right
                        ):
                            match insole_right:
                                case lat if "lateral" in lat:
                                    data_csv.rename(
                                        columns={insole_right: "f_lateral_r"},
                                        inplace=True,
                                    )
                                case med if "medial" in med:
                                    data_csv.rename(
                                        columns={insole_right: "f_medial_r"},
                                        inplace=True,
                                    )
                                case heel if "heel" in heel:
                                    data_csv.rename(
                                        columns={insole_right: "f_heel_r"},
                                        inplace=True,
                                    )
                                case tot if (
                                    insole in tot and "-R" in tot and ":" not in tot
                                ):
                                    data_csv.rename(
                                        columns={insole_right: "f_total_r"},
                                        inplace=True,
                                    )
                                case xacc if "xAcc" in xacc:
                                    data_csv.rename(
                                        columns={insole_right: "acc_x_r"},
                                        inplace=True,
                                    )

                                case yacc if "yAcc" in yacc:
                                    data_csv.rename(
                                        columns={insole_right: "acc_y_r"},
                                        inplace=True,
                                    )
                                case zacc if "zAcc" in zacc:
                                    data_csv.rename(
                                        columns={insole_right: "acc_z_r"},
                                        inplace=True,
                                    )
                                case xgyro if "xGyro" in xgyro:
                                    data_csv.rename(
                                        columns={insole_right: "gyro_x_r"},
                                        inplace=True,
                                    )

                                case ygyro if "yGyro" in ygyro:
                                    data_csv.rename(
                                        columns={insole_right: "gyro_y_r"},
                                        inplace=True,
                                    )
                                case zgyro if "zGyro" in zgyro:
                                    data_csv.rename(
                                        columns={insole_right: "gyro_z_r"},
                                        inplace=True,
                                    )
                        case code_left if insoles_codes[insole]["code_left"] in code_left:
                            match code_left:
                                case lat if "lateral" in lat:
                                    data_csv.rename(
                                        columns={code_left: "f_lateral_l"},
                                        inplace=True,
                                    )
                                case med if "medial" in med:
                                    data_csv.rename(
                                        columns={code_left: "f_medial_l"},
                                        inplace=True,
                                    )
                                case heel if "heel" in heel:
                                    data_csv.rename(
                                        columns={code_left: "f_heel_l"},
                                        inplace=True,
                                    )
                                case tot if (
                                    insoles_codes[insole]["code_left"] in tot
                                    and ":" not in tot
                                ):
                                    data_csv.rename(
                                        columns={code_left: "f_total_l"},
                                        inplace=True,
                                    )
                                case xacc if "xAcc" in xacc:
                                    data_csv.rename(
                                        columns={code_left: "acc_x_l"},
                                        inplace=True,
                                    )

                                case yacc if "yAcc" in yacc:
                                    data_csv.rename(
                                        columns={code_left: "acc_y_l"},
                                        inplace=True,
                                    )
                                case zacc if "zAcc" in zacc:
                                    data_csv.rename(
                                        columns={code_left: "acc_z_l"},
                                        inplace=True,
                                    )
                                case xgyro if "xGyro" in xgyro:
                                    data_csv.rename(
                                        columns={code_left: "gyro_x_l"},
                                        inplace=True,
                                    )

                                case ygyro if "yGyro" in ygyro:
                                    data_csv.rename(
                                        columns={code_left: "gyro_y_l"},
                                        inplace=True,
                                    )
                                case zgyro if "zGyro" in zgyro:
                                    data_csv.rename(
                                        columns={code_left: "gyro_z_l"},
                                        inplace=True,
                                    )
                        case insole_left if insole in insole_left and "-L" in insole_left:
                            match insole_left:
                                case lat if "lateral" in lat:
                                    data_csv.rename(
                                        columns={insole_left: "f_lateral_l"},
                                        inplace=True,
                                    )
                                case med if "medial" in med:
                                    data_csv.rename(
                                        columns={insole_left: "f_medial_l"},
                                        inplace=True,
                                    )
                                case heel if "heel" in heel:
                                    data_csv.rename(
                                        columns={insole_left: "f_heel_l"},
                                        inplace=True,
                                    )
                                case tot if (
                                    insole in tot and "-L" in tot and ":" not in tot
                                ):
                                    data_csv.rename(
                                        columns={insole_left: "f_total_l"},
                                        inplace=True,
                                    )
                                case xacc if "xAcc" in xacc:
                                    data_csv.rename(
                                        columns={insole_left: "acc_x_l"},
                                        inplace=True,
                                    )

                                case yacc if "yAcc" in yacc:
                                    data_csv.rename(
                                        columns={insole_left: "acc_y_l"},
                                        inplace=True,
                                    )
                                case zacc if "zAcc" in zacc:
                                    data_csv.rename(
                                        columns={insole_left: "acc_z_l"},
                                        inplace=True,
                                    )
                                case xgyro if "xGyro" in xgyro:
                                    data_csv.rename(
                                        columns={insole_left: "gyro_x_l"},
                                        inplace=True,
                                    )

                                case ygyro if "yGyro" in ygyro:
                                    data_csv.rename(
                                        columns={insole_left: "gyro_y_l"},
                                        inplace=True,
                                    )
                                case zgyro if "zGyro" in zgyro:
                                    data_csv.rename(
                                        columns={insole_left: "gyro_z_l"},
                                        inplace=True,
                                    )

            # Rename columns names (time)
            for insole in self.insoles:
                for key in data_csv.keys():
                    index = data_csv.columns.get_loc(key)
                    match key:
                        case right if (
                            "Unnamed" in right
                            and "_r" in data_csv.keys()[index + 1]
                        ):
                            data_csv.rename(columns={key: "time_r"}, inplace=True)
                        case left if (
                            "Unnamed" in left
                            and "_l" in data_csv.keys()[index + 1]
                        ):
                            data_csv.rename(columns={key: "time_l"}, inplace=True)

        elif state == "curated":
            data_csv = pd.read_csv(
                self.path_csv, sep=",", header=0, na_values="-", dtype=str
            )

            # Columns to suppress are those that do not have a following column or whose following column does not contain a keyword
            col_to_suppress = []
            for idx, col in enumerate(data_csv.columns):
                if "Unnamed" in col:
                        col_to_suppress.append(idx)

            data_csv.drop(data_csv.columns[col_to_suppress], axis=1, inplace=True)

        # Convert values into floats
        data_csv = data_csv.astype(float)
        self.raw_data = data_csv

    def clean_data(self):
        """Replaces the incorrect values in force data by nan (incorrect value = negative value of force per zone).
        Finds and suppresses duplicate data.
        """

        if self.raw_data is None:
            self.read_csv()

        # Initialize cleaned data
        cleaned_data = deepcopy(self.raw_data)

        # Total number of frames before cleaning
        total_frames = len(cleaned_data)

        # Indexes of incorrect values
        indexes_incorrect_values_l = self.raw_data[
            self.raw_data["f_heel_l"] == -1
        ].index.tolist()

        indexes_incorrect_values_r = self.raw_data[
            self.raw_data["f_heel_r"] == -1
        ].index.tolist()

        # Proportion of incorrect values compared to the total number of frames
        incorrect_frames_l = len(indexes_incorrect_values_l)
        incorrect_frames_r = len(indexes_incorrect_values_r)

        print(f"Number of incorrect frames (left): {incorrect_frames_l}")
        print(f"Number of incorrect frames (right): {incorrect_frames_r}")

        proportion_incorrect_frames_l = np.round(
            (incorrect_frames_l / total_frames) * 100, 1
        )
        proportion_incorrect_frames_r = np.round(
            (incorrect_frames_r / total_frames) * 100, 1
        )

        print(
            f"Proportion of incorrect frames (left): {proportion_incorrect_frames_l} %"
        )
        print(
            f"Proportion of incorrect frames (right): {proportion_incorrect_frames_r} %"
        )

        # Columns names
        names_l = {
            "f_heel_l",
            "f_medial_l",
            "f_lateral_l",
            "f_total_l",
        }
        names_r = {
            "f_heel_r",
            "f_medial_r",
            "f_lateral_r",
            "f_total_r",
        }

        # Replace by nan incorrect values
        for name_l, name_r in zip(names_l, names_r):
            cleaned_data.loc[indexes_incorrect_values_l, name_l] = np.nan
            cleaned_data.loc[indexes_incorrect_values_r, name_r] = np.nan

        # Indexes of duplicate time data
        duplicate_l = cleaned_data["time_l"][
            cleaned_data["time_l"].duplicated() == True
        ].index

        duplicate_r = cleaned_data["time_r"][
            cleaned_data["time_r"].duplicated() == True
        ].index

        # Proportion of duplicate values compared to the total number of frames
        duplicate_frames_l = len(duplicate_l)
        duplicate_frames_r = len(duplicate_r)

        print(f"Number of incorrect frames (left): {duplicate_frames_l}")
        print(f"Number of incorrect frames (right): {duplicate_frames_r}")

        proportion_duplicate_frames_l = np.round(
            (duplicate_frames_l / total_frames) * 100, 1
        )
        proportion_duplicate_frames_r = np.round(
            (duplicate_frames_r / total_frames) * 100, 1
        )

        print(
            f"Proportion of incorrect frames (left): {proportion_duplicate_frames_l} %"
        )
        print(
            f"Proportion of incorrect frames (right): {proportion_duplicate_frames_r} %"
        )

        # Indexes of data to be suppressed (the duplicate data to be suppressed is the one immediately before the previously identified data)
        index_duplicate_l = duplicate_l - 1
        index_duplicate_r = duplicate_r - 1

        # Columns names
        names_left = {
            "time_l",
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
        }

        names_right = {
            "time_r",
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
        }

        # Supress duplicate time values and other values associated
        for name_left, name_right in zip(names_left, names_right):
            cleaned_data.loc[index_duplicate_l, name_left] = np.nan
            cleaned_data.loc[index_duplicate_r, name_right] = np.nan

        self.cleaned_data = cleaned_data

        # Identify the indexes of NaN values
        nan_indexes_l = cleaned_data["f_total_l"].isna()
        nan_indexes_r = cleaned_data["f_total_r"].isna()

        # Identify NaN ranges with an unique identificator
        nan_range_l = (nan_indexes_l != nan_indexes_l.shift()).cumsum()
        nan_range_r = (nan_indexes_r != nan_indexes_r.shift()).cumsum()

        # Keep only the groups of consecutive NaN
        nan_blocks_l = cleaned_data[nan_indexes_l].groupby(nan_range_l)
        nan_blocks_r = cleaned_data[nan_indexes_r].groupby(nan_range_r)

        # Compute the duration of each block
        durations_l = []
        for _, group in nan_blocks_l:
            # Start and end indexes of the block
            start_idx = group.index.min()
            end_idx = group.index.max()

            # Use known data before and after to estimate duration
            if start_idx > 0 and end_idx + 1 < len(cleaned_data):
                t_start = cleaned_data.loc[start_idx - 1, "time_l"]
                t_end = cleaned_data.loc[end_idx + 1, "time_l"]
                if pd.notna(t_start) and pd.notna(t_end):
                    durations_l.append(t_end - t_start)

        durations_r = []
        for _, group in nan_blocks_r:
            # Start and end indexes of the block
            start_idx = group.index.min()
            end_idx = group.index.max()

            # Use known data before and after to estimate duration
            if start_idx > 0 and end_idx + 1 < len(cleaned_data):
                t_start = cleaned_data.loc[start_idx - 1, "time_r"]
                t_end = cleaned_data.loc[end_idx + 1, "time_r"]
                if pd.notna(t_start) and pd.notna(t_end):
                    durations_r.append(t_end - t_start)

        mean_duration_l = np.round(np.mean(durations_l), 2) if durations_l else 0
        mean_duration_r = np.round(np.mean(durations_r), 2) if durations_r else 0

        print(
            f"Mean duration of removed data (left): {mean_duration_l} s"
        )
        print(
            f"Mean duration of removed data (right): {mean_duration_r} s"
        )

    def fill_missing_data(self):
        """Interpolates missing data."""

        if self.cleaned_data is None:
            self.clean_data()

        # Initialize filled data
        filled_data = deepcopy(self.cleaned_data)

        # Rename Time column and suppress time_r
        filled_data.drop(columns=["time_r"], inplace=True)
        filled_data.rename(
            columns={"time_l": "time"}, inplace=True
        )

        # Truncate lines if there are nan values at the end of the DataFrame
        while filled_data.iloc[-1].isna().any():
            filled_data.drop(filled_data.index[-1], axis=0, inplace=True)

        # Interpolate time if there is missing data (due to duplicate data suppressed)
        filled_data["time"] = filled_data["time"].interpolate(
            method="linear"
        )

        # Columns to interpolate
        col_to_interpol_l = [
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
        ]

        col_to_interpol_r = [
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
        ]

        for col in col_to_interpol_l:
            known = filled_data[col].notna()
            x_l = filled_data.loc[known, "time"]
            y_l = filled_data.loc[known, col]

            # Create interpolator (spline)
            interp = PchipInterpolator(x_l, y_l)

            # Interpolate
            filled_data[col] = interp(filled_data["time"])

            # Constraint to avoid force negative values
            if col in [
                "f_heel_l",
                "f_medial_l",
                "f_lateral_l",
                "f_total_l",
            ]:
                filled_data[col] = filled_data[col].clip(lower=0)

        for col in col_to_interpol_r:
            known = filled_data[col].notna()
            x_r = filled_data.loc[known, "time"]
            y_r = filled_data.loc[known, col]

            # Create interpolator (spline)
            interp = PchipInterpolator(x_r, y_r)

            # Interpolate
            filled_data[col] = interp(filled_data["time"])

            # Constraint to avoid force negative values
            if col in [
                "f_heel_r",
                "f_medial_r",
                "f_lateral_r",
                "f_total_r",
            ]:
                filled_data[col] = filled_data[col].clip(lower=0)

        self.filled_data = filled_data

    def cut_data(self, data: pd.DataFrame, path: str, file: str, row: str, trial: str):
        """Cuts signal around the scrummaging (between the instruction "Crouch" to the final whistle blast).

        Args:
            path (str): path of the directory containing the yaml file containing the indexes for cutting signals
            file (str): name of the yaml file containing the indexes for cutting signals
            row (list): list of the row of the player (FR: First Row, SR: Second Row, TR: Third Row)
            trial (str): name of the trial associated with the data file
        """

        # Initialize cut data
        data_cut = deepcopy(data)

        # Read yaml file containing the indexes for synchronization
        with open(os.path.abspath(os.path.join(path, file)), "r") as idxs:
            indexes = yaml.safe_load(idxs)

        # Add a column with events
        total_length = data_cut.shape[0]
        data_cut.insert(loc=0, column="events", value=[np.nan] * total_length)
        data_cut["events"] = data_cut["events"].astype("object")
        data_cut.loc[
            indexes[trial]["crouch"][row] : indexes[trial]["bind"][row], "events"
        ] = "crouch"
        data_cut.loc[
            indexes[trial]["bind"][row] : indexes[trial]["set"][row], "events"
        ] = "bind"
        data_cut.loc[
            indexes[trial]["set"][row] : indexes[trial]["end"][row], "events"
        ] = "set"

        # Cut signals from start to end
        data_cut = data_cut[indexes[trial]["crouch"][row] : indexes[trial]["end"][row]]

        # Reset index
        data_cut.reset_index(drop=True, inplace=True)

        # Redefine time
        new_time = data_cut["time"] - data_cut["time"][0]
        data_cut.insert(loc=0, column="time", value=new_time)

        for insole in self.insoles:
            data_cut.drop(insole + "time", axis=1, inplace=True)

        self.data_cut = data_cut

    def downsample(self, data: pd.DataFrame, final_frequency: int):
        """Downsamples data to the final frequency.

        Args:
            final_frequency (int): downsampled frequency
        """

        self.final_frequency = final_frequency

        # Downsampling ratio
        self.ratio = int(self.frequency / self.final_frequency)

        # Initialise downsampled data
        downsampled_data = pd.DataFrame()

        # Create new time vector based on the final frequency
        t_ds = np.arange(
            data["time"].iloc[0],
            data["time"].iloc[-1],
            1 / final_frequency,
        )

        for key in data.keys():
            # Create interpolation function
            f = interp1d(data["time"], data.get(key))

            # Downsample data
            downsampled_data[key] = f(t_ds)

        self.downsampled_data = downsampled_data

    def filter(self, data: pd.DataFrame, order: int, fcut: int):
        """Apply a butterworth filter with a backward & forward pass.

        Args:
            fs: sampling frequency of the signal.
            order (int): order of the filter. Note that with the backward & forward pass, this order will be multiplied by 2.
            fcut (int): Must be smaller than the half of the sampling frequency.
        """

        if self.final_frequency is None:
            self.final_frequency = self.frequency

        # Initialise downsampled data
        self.filtered_data = deepcopy(data)

        Wn = fcut / (self.final_frequency / 2)

        b, a = signal.butter(order, Wn, analog=False)

        # Filter data (except time)
        for column in self.filtered_data.columns:
            if "time" not in column:
                self.filtered_data[column] = signal.filtfilt(
                    b, a, self.filtered_data[column]
                )

        # Substitute too small values (< tolerance) by zero (except time)
        tol = 0.001
        for column in self.filtered_data.columns:
            if "time" not in column and "acc" not in column and "gyro" not in column:
                self.filtered_data[column] = np.where(
                    np.abs(self.filtered_data[column]) < tol,
                    0,
                    self.filtered_data[column],
                )

                # Suppress parasitic negative values by replacing them by zero
                self.filtered_data[column] = np.maximum(self.filtered_data[column], 0)

                # Round force data
                self.filtered_data[column] = round(self.filtered_data[column], 2)

            # Round acc and gyro data
            if "acc" in column or "gyro" in column:
                self.filtered_data[column] = round(self.filtered_data[column], 3)

        # Apply correction to still have the sum of the force on each area equal to the total value, even after filtering
        col_corrected = pd.DataFrame()
        for side in ["_l", "_r"]:
            cols = [
                "f_heel" + side,
                "f_medial" + side,
                "f_lateral" + side,
            ]

            total_col = "f_total" + side

            # Replace NaN values by zero (division by zero so NaN values) and compute the sum on the three areas
            sum_cols = self.filtered_data[cols].sum(axis=1).replace(0, np.nan)

            # Apply correction
            col_corrected[cols] = (
                self.filtered_data[cols]
                .multiply(self.filtered_data[total_col], axis=0)
                .div(sum_cols, axis=0)
            ).fillna(0.0)

            # Find the max value between the three areas fo each line
            max_value = col_corrected[cols].idxmax(axis=1)

            # Round all values on the columns
            for col in cols:
                col_corrected[col] = col_corrected[col].round(2)

            # Apply correction
            for i, row in col_corrected.iterrows():
                max_col = max_value[i]
                other_cols = [c for c in cols if c != max_col]
                col_corrected.at[i, max_col] = (
                    self.filtered_data.at[i, total_col] - row[other_cols].sum()
                )

            # Attribute new corrected values to the filtered data
            self.filtered_data[cols] = col_corrected[cols].round(2)


    def export_pre_treated_data(self, data: pd.DataFrame, path: str, name: str):
        """Exports the DataFrame containing the pre-treated data to a csv file.

        Args:
            path (str): path of the directory to save the csv file.
            name (str): name of the csv file to save.
        """

        data.to_csv(os.path.abspath(os.path.join(path, name + ".csv")))
