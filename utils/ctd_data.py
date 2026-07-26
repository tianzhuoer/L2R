import os
import json
import scipy.io
import random
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from config import *


def _load_rl_filelist(folder_path, manifest_path):
    
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)['manifest']
        files = [k for k, v in manifest.items() if v == 'rl_train']
        if files:
            return files
        print("[CTDDataProcessor] 警告：manifest 中无 rl_train 文件，回退到全量扫描。")
    return [fn for fn in os.listdir(folder_path) if fn.endswith('.mat')]


class CTDDataProcessor:
    def __init__(self, folder_path="profileresults", manifest_path=None):
        self.folder_path = folder_path
        _manifest = manifest_path if manifest_path is not None else CTD_SPLIT_MANIFEST
        self._rl_files = _load_rl_filelist(folder_path, _manifest)

    def datenum2datetime(self, datenum):
        """
        Convert MATLAB datenum to Python datetime.

        Args:
            datenum: A scalar or array of MATLAB datenum values.

        Returns:
            A datetime object or an array of datetime objects.
        """
        return datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=datenum % 1) - datetime.timedelta(days=366)

    def load_random_mat_file(self):
        if not self._rl_files:
            print("No rl_train .mat files found.")
            return None, None
        random_filename = random.choice(self._rl_files)
        file_path = os.path.join(self.folder_path, random_filename)
        try:
            data = scipy.io.loadmat(file_path)
            return random_filename[:-4], data
        except Exception as e:
            print(f"Error loading {random_filename}: {e}")
            return None, None

    def random_sample(self, sample_days=SAMPLE_DAYS, max_tries=30):
        """
        Loads a random .mat file, selects a random start point, and extracts data for a specified number of days.

        Args:
            folder_path (str): Path to the folder containing .mat files.
            sample_days (int): Number of days to sample.
            max_tries (int): Maximum number of files to try before giving up.

        Returns:
            tuple: (filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid) or (None, None, None, None, None) on error.
        """
        for _try in range(max_tries):
            result = self._random_sample_once(sample_days)
            if result[0] is not None:
                return result
        print(f"[CTDDataProcessor] random_sample: 尝试 {max_tries} 次后未找到有效片段。")
        return None, None, None, None, None

    def _random_sample_once(self, sample_days=SAMPLE_DAYS):
        filename, mat_data = self.load_random_mat_file()
        # print(f"Loaded file: {filename}")
        if not filename:
            print("Failed to load a .mat file.")
            return None, None, None, None, None

        if 'new_T_interp' not in mat_data or 'new_S_interp' not in mat_data or 'new_time_grid' not in mat_data or 'depth_grid' not in mat_data:
            print("Required variables not found in .mat file.")
            return None, None, None, None, None

        new_T_interp = mat_data['new_T_interp']
        new_S_interp = mat_data['new_S_interp']
        new_time_grid = mat_data['new_time_grid']
        depth_grid = mat_data['depth_grid']

        # Convert to numpy arrays
        new_T_interp = np.asarray(new_T_interp)
        new_S_interp = np.asarray(new_S_interp)
        new_time_grid = np.asarray(new_time_grid).flatten()
        depth_grid = np.asarray(depth_grid).flatten()

        nan_count_per_column = np.isnan(new_T_interp).sum(axis=0)


        mask_too_many_nan = nan_count_per_column > 20
        new_T_interp = new_T_interp[:, ~mask_too_many_nan]
        new_S_interp = new_S_interp[:, ~mask_too_many_nan]
        new_time_grid = new_time_grid[~mask_too_many_nan]

        mask_Nan = np.isnan(new_T_interp).any(axis=1)
        new_T_interp = new_T_interp[~mask_Nan, :]
        new_S_interp = new_S_interp[~mask_Nan, :]

        depth_grid = depth_grid[~mask_Nan]

        if len(new_time_grid) == 0 or new_T_interp.shape[1] == 0:
            return None, None, None, None, None

        # Convert new_time_grid to datetime objects
        try:
            new_time_grid_datetime = [self.datenum2datetime(dn) for dn in new_time_grid]
        except TypeError as e:
            print(f"TypeError in file '{filename}': {e}")
            print(f"new_time_grid type: {type(new_time_grid)}, dtype: {getattr(new_time_grid, 'dtype', None)}")
            return None, None, None, None, None


        total_time_range = new_time_grid_datetime[-1] - new_time_grid_datetime[0]
        if total_time_range < datetime.timedelta(days=sample_days):
            return None, None, None, None, None

        max_start_index = len(new_time_grid_datetime) - 1 - 2 * sample_days
        if max_start_index < 0:
            return None, None, None, None, None
        start_index = random.randint(0, max_start_index)

        # Ensure that we can extract 4 days from the start index
        start_date = new_time_grid_datetime[start_index]
        end_date = start_date + datetime.timedelta(days=sample_days)

        # Find the index closest to the end date
        end_index = min((np.abs(np.asarray(new_time_grid_datetime) - end_date)).argmin(), len(new_time_grid_datetime) - 1)

        # Extract the data
        sampled_T_interp = new_T_interp[:, start_index:end_index+1]
        sampled_S_interp = new_S_interp[:, start_index:end_index+1]
        sampled_time_grid_datetime = new_time_grid_datetime[start_index:end_index+1]
        if np.isnan(sampled_T_interp).any():
            print(f"Warning: NaN found in sampled_T_interp: {sampled_T_interp}")



        return filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid

    def sample_in_year_range(self, year_start, year_end, sample_days=SAMPLE_DAYS, max_tries=100):
        
        _base_sample = CTDDataProcessor.random_sample
        for _ in range(max_tries):
            result = _base_sample(self, sample_days)
            if result[0] is None:
                continue
            start_dt = result[3][0]
            if year_start <= start_dt.year <= year_end:
                return result
        return (None, None, None, None, None)

    def plot_sampled_data(self, filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid):
        """Plots the sampled temperature and salinity data."""
        if filename:
            print(f"Sampled from file: {filename}")
            print(f"Sampled T_interp shape: {sampled_T_interp.shape}")
            print(f"Sampled S_interp shape: {sampled_S_interp.shape}")
            print(f"Sampled time_grid shape: {len(sampled_time_grid_datetime)}")

            # Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plotting new_T_interp
            im1 = ax1.pcolormesh(sampled_time_grid_datetime, depth_grid, sampled_T_interp)
            ax1.set_xlabel("Time (Year-Month)")
            ax1.set_ylabel("Depth")
            ax1.set_title("Sampled new_T_interp vs. Time and Depth")
            fig.colorbar(im1, ax=ax1, label="new_T_interp")
            ax1.invert_yaxis()
            # Format the x-axis to show Year and Month
            date_format = mdates.DateFormatter('%Y-%m')
            ax1.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()  # Rotate date labels for better readability

            # Plotting new_S_interp
            im2 = ax2.pcolormesh(sampled_time_grid_datetime, depth_grid, sampled_S_interp)
            ax2.set_xlabel("Time (Year-Month)")
            ax2.set_ylabel("Depth")
            ax2.set_title("Sampled new_S_interp vs. Time and Depth")
            fig.colorbar(im2, ax=ax2, label="new_S_interp")
            ax2.invert_yaxis()
            # Format the x-axis to show Year and Month
            ax2.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()  # Rotate date labels for better readability

            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.show()

        else:
            print("Sampling failed.")

# # Example Usage
# processor = CTDDataProcessor(folder_path="CTD")
# for i in range(300):
#     filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid = processor.random_sample()

#     if np.isnan(sampled_T_interp).any():
#         print(f"There are NaN values in sampled_T_interp for file: {filename}")

# # processor.plot_sampled_data(filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid)
