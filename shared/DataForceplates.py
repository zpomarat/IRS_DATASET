import os.path
import datetime
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import c3d


class DataForceplates:
    def __init__(self, dir_path: str, file_name: str, frequency: int, fp_number: list):
        """Creates a DataForceplates object.

        Args:
            dir_path (str): path of the directory containing the file from which the object is created
            file_name (str): name of the file from which the object is created
            frequency (int): frequency of the data contained in the file from which the object is created
            fp_number (list): list of the forceplates numbers to be treated
        """
        self.path = dir_path
        self.file_name = file_name
        self.path_xcp = os.path.abspath(os.path.join(self.path,self.file_name + ".xcp"))
        self.path_csv = os.path.abspath(os.path.join(self.path,self.file_name + ".csv"))
        self.path_c3d = os.path.abspath(os.path.join(self.path,self.file_name + ".c3d"))
        self.frequency = frequency
        self.final_frequency = None
        self.fp_number = fp_number
        self.raw_data = None
        self.timestamp = None
        self.pre_processed_data = None
        self.downsampled_data = None
        self.filtered_data = None
        self.data_synchro = None

    def extract_timestamp(self):
        """Extracts the timestamp from the .xcp file.

        timestamp = {date: YEAR-MONTH-DAY,
                    time: HOUR-MINUTE-SECOND-MICROSECOND}
        """

        if os.path.exists(self.path_xcp) == False:
            raise ValueError("The xcp file does not exist.")

        # Read the line line containing the timestamp in the text file
        with open(self.path_xcp) as file:
            lines = file.readlines()
        file.close()

        # Line containing the timestamp
        line = lines[7]

        # Split the line to extract the part containing the timestamp
        txt = line.split('START_TIME="')

        # Select the item of txt containing the timestamp and split it
        ts = txt[-1].split(" ")

        # Specify the format of the timestamp (Date: Year-Month-Day, Time: Hour:Minute:Second.Microsecond)
        format_date = "%Y-%m-%d"
        format_time = "%H:%M:%S.%f"

        # Extract timestamp
        parsed_date = None
        parsed_time = None

        for element in ts:
            try:
                if not parsed_date:
                    parsed_date = datetime.datetime.strptime(
                        element, format_date
                    ).date()
                elif not parsed_time:
                    parsed_time = datetime.datetime.strptime(
                        element, format_time
                    ).time()
            except ValueError:
                continue

        assert parsed_date is not None, "Date could not be parsed."
        assert parsed_time is not None, "Time could not be parsed."

        self.timestamp = datetime.datetime.combine(parsed_date, parsed_time)

    def read_c3d(self):
        """Reads a c3d file and extracts the raw analog data and time for the forceplates 1 and 2."""

        if os.path.exists(self.path_c3d) == False:
            raise ValueError("The c3d file does not exist.")

        # Read the c3d file
        file = open(self.path_c3d, "rb")
        reader = c3d.Reader(file)

        # Extract analog data (raw data + others parameters)
        data_analog = []

        for analog in reader.read_frames():
            data_analog.append(analog)
        file.close()

        # Create an array containing only the raw analog data
        raw_data = []

        for i in range(len(data_analog)):
            if i == 0:
                raw_data = data_analog[i][2].T
            else:
                raw_data = np.concatenate((raw_data, data_analog[i][2].T), axis=0)

        ## Extract time
        time = np.arange(0, len(raw_data) / self.frequency, 1 / self.frequency)

        ## Extract raw data
        raw_data_f1 = raw_data[:, 0:3]
        raw_data_f2 = raw_data[:, 6:9]

        # Create a DataFrame with raw data
        data = np.array(
            [
                time,
                raw_data_f1[:, 0],
                raw_data_f1[:, 1],
                raw_data_f1[:, 2],
                raw_data_f2[:, 0],
                raw_data_f2[:, 1],
                raw_data_f2[:, 2],
            ]
        )
        columns = ["time", "fx1", "fy1", "fz1", "fx2", "fy2", "fz2"]
        self.raw_data = pd.DataFrame(data, columns).T

    def read_csv(self,state="raw"):
        """Reads a csv file and extracts the raw data and time for the forceplates 2 and 5 (Petite Plateforme)."""

        if os.path.isfile(self.path_csv) == False:
            raise ValueError("The csv file does not exist.")

        if state == "raw":    

            # Read the csv file as a Dataframe
            data_csv = pd.read_csv(
                self.path_csv, sep=",", header=2, na_values="-", dtype=str
            )

            # Correspondance columns names and fp numbers
            forceplates_names = {"1": "Plateforme Sensix 1 - Force",
                                "2": "Plateforme Sensix 2 - Force",
                                "3": "Plateforme Sensix 3 - Force",
                                "4": "Grande Plateforme AMTI - Force",
                                "5": "Petite Plateforme AMTI - Force"}

            # Indexes of columns to keep
            columns_to_keep = []

            for number in self.fp_number:
                columns_to_keep.append(
                    data_csv.columns.get_loc(forceplates_names[str(number)])
                )
                columns_to_keep.append(
                    data_csv.columns.get_loc(forceplates_names[str(number)]) + 1
                )
                columns_to_keep.append(
                    data_csv.columns.get_loc(forceplates_names[str(number)]) + 2
                )

            # Suppress all others columns
            data_csv.drop(
                columns=[
                    data_csv.columns[idx]
                    for idx in range(len(data_csv.columns))
                    if idx not in columns_to_keep
                ],
                inplace=True,
            )

            # Rename columns names
            rename_dict = {}
            for number in self.fp_number:
                col_idx = data_csv.columns.get_loc(forceplates_names[str(number)])
                rename_dict[data_csv.columns[col_idx]]     = f"fx{number}"
                rename_dict[data_csv.columns[col_idx + 1]] = f"fy{number}"
                rename_dict[data_csv.columns[col_idx + 2]] = f"fz{number}"

            data_csv.rename(columns=rename_dict, inplace=True)

            # Suppres the lines with  force denomination and units
            data_csv.drop([0, 1], axis=0, inplace=True)

            # Reset index
            data_csv.reset_index(drop=True, inplace=True)

            # Add time
            time = np.arange(0, len(data_csv) / self.frequency, 1 / self.frequency)
            if len(time) > len(data_csv):
                time = time[0 : len(data_csv)]

            data_csv.insert(loc=0, column="time", value=time)
        
        elif state == "curated":
            # Read the csv file as a Dataframe
            data_csv = pd.read_csv(
                self.path_csv, sep=",", header=0, na_values="-", dtype=str
            )

            # Columns to suppress are those that do not have a following column or whose following column does not contain a keyword
            col_to_suppress = []
            for idx, col in enumerate(data_csv.columns):
                if "Unnamed" in col:
                        col_to_suppress.append(idx)

            data_csv.drop(data_csv.columns[col_to_suppress], axis=1, inplace=True)

        # Convert valued into floats
        data_csv = data_csv.astype(float)

        self.raw_data = data_csv

    def pre_process_data(self):
        """Inverses the forces to have the reaction of the ground on the foot and set values to zero."""

        if self.raw_data is None:
            if os.path.isfile(self.path_csv) == False:
                if os.path.isfile(self.path_c3d) == True:
                    self.read_c3d()
            if os.path.isfile(self.path_c3d) == False:
                if os.path.isfile(self.path_csv) == True:
                    self.read_csv()


        # Initialise pre processed data
        self.pre_processed_data = deepcopy(self.raw_data)

        # Change orientation
        keys = [key for key in self.pre_processed_data.keys() if "time" not in key]

        for key in keys:
            self.pre_processed_data[key] = -self.pre_processed_data[key]

    def downsample(self, data: pd.DataFrame, final_frequency: int):
        """Downsamples data to the final frequency.

        Args:
            final_frequency (int): downsampled frequency
        """

        if final_frequency > self.frequency:
            raise ValueError(
                "The final frequency should be lower than the frequency of the signal to do a downsampling."
            )

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
            if len(data.get(key)) != 0:
                f = interp1d(
                    data["time"], data.get(key)
                )

                # Downsample data
                downsampled_data[key] = f(t_ds)

        self.downsampled_data = downsampled_data

    def filter(self, data: pd.DataFrame, order: int, fcut: int):
        """Apply a butterworth filter with a backward & forward pass.

        Args:
            fs: sampling frequency of the signal (after downsampling).
            order (int): order of the filter. Note that with the backward & forward pass, this order will be multiplied by 2.
            fcut (int): Must be smaller than the half of the sampling frequency.
        """

        if self.downsampled_data is None:
            if fcut > self.frequency:
                raise ValueError(
                    "The cutting frequency should be lower than the frequency."
                )
        else:
            if fcut > self.final_frequency:
                raise ValueError(
                    "The cutting frequency should be lower than the frequency."
                )

        # Initialise downsampled data
        self.filtered_data = deepcopy(data)

        if self.final_frequency is None:
            Wn = fcut / (self.frequency / 2)
        else:
            Wn = fcut / (self.final_frequency / 2)

        b, a = signal.butter(order, Wn, analog=False)

        # Filter data
        keys = [key for key in data.keys() if "time" not in key]

        for key in keys:
            self.filtered_data[key] = signal.filtfilt(b, a, self.filtered_data[key])

    def export_pre_treated_data(self, data: pd.DataFrame, path: str, name: str):
        """Exports the DataFrame containing the curated data to a csv file.

        Args:
            path (str): path of the directory to save the csv file.
            name (str): name of the csv file to save.
        """

        data.to_csv(os.path.abspath(os.path.join(path, name + ".csv")))