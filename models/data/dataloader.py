import os
import pandas as pd
import numpy as np
import torch
import h5py
import torch.utils.data as data
from obspy import UTCDateTime
from sklearn.model_selection import train_test_split

class STEADDataset(data.Dataset):
    def __init__(self, file, csv_file, window_size, transform=None):
        self.file = file
        self.window_size = window_size
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.edf = self.df[(self.df.trace_category == "earthquake_local") & (self.df.source_distance_km <= 30) & (self.df.source_magnitude > 3)]
        self.ndf = self.df[(self.df.trace_category == 'noise')]
        self.ev_list = self.edf['trace_name'].tolist()
        self.no_list = self.ndf['trace_name'].tolist()
        self.splits = self.split_data()
        self.f = h5py.File(self.file, "r")

    def __len__(self):
        return len(self.ev_list) + len(self.no_list)
    
    def len(self):
        return f"The total number of earthquake events is {len(self.ev_list)} and the total number of noise events is {len(self.no_list)}"
    
    def split_data(self):
        ev_train, ev_temp = train_test_split(self.ev_list, test_size=0.2, random_state=42)
        ev_test, ev_dev = train_test_split(ev_temp, test_size=0.5, random_state=42)
        no_test, no_dev = train_test_split(self.no_list, test_size=0.1, random_state=42)
        return {
        "eq_train": ev_train,
        "eq_test": ev_test,
        "eq_dev": ev_dev,
        "noise_test": no_test,
        "noise_dev": no_dev
        }
    
    def extract_wave_window(data, wave_index, window_size):
        half_window = window_size // 2 
        start_index = max(0, wave_index - half_window)
        end_wave = wave_index + half_window    
        end_index = min(end_wave, data.shape[1])
        return data[:, start_index:end_index]
    
    def get_eq_waveform(self, trace_name):
        dataset = self.f.get(f"data/{trace_name}")
        if dataset is None:
            raise ValueError(f"Trace {trace_name} not found in HDF5 file.")
        
        data = np.array(dataset)

        p_arrival_index = int(dataset.attrs.get("p_arrival_sample", -1))
        s_arrival_index = int(dataset.attrs.get("s_arrival_sample", -1))

        window_size = self.window_size
        s_p_diff = s_arrival_index - p_arrival_index

        if s_p_diff < window_size:
            return None
        
        p_data = self.extract_wave_window(data, p_arrival_index, window_size)
        s_data = self.extract_wave_window(data, s_arrival_index, window_size)

        data = np.stack([p_data, s_data], axis=0)
        # data[0] = p_data, data[1] = s_data
        data = torch.tensor(data, dtype=torch.float32)
        # p_wave = torch.tensor(p_wave, dtype=torch.float32) if p_wave is not None else None
        # s_wave = torch.tensor(s_wave, dtype=torch.float32) if s_wave is not None else None

        if self.transform:
            data = self.transform(data)

        return data
    
    def get_noise_data(self, trace_name):
        dataset = self.f.get(f"data/{trace_name}")
        if dataset is None:
            raise ValueError(f"Trace {trace_name} not found in HDF5 file.")
        
        data = np.array(dataset)

        total_samples = data.shape[1]
        if total_samples < self.window_size:
            return None
        
        num_windows = total_samples // self.window_size
        sliced_data = [data[:, i * self.window_size:(i + 1) * self.window_size] for i in range(num_windows)]

        data = torch.tensor(sliced_data, dtype=torch.float32)

        return data

    
    def get_train_data(self):
        return [self.get_eq_waveform(trace) for trace in self.splits["eq_train"]]
    
    def get_test_data(self):
        test_waveforms = [self.get_eq_waveform(trace) for trace in self.splits["eq_test"]]
        noise_waveforms = [self.get_waveform(trace) for trace in self.splits["noise_test"]]
        return test_waveforms + noise_waveforms
    
    def __getitem__(self, idx):
        if idx < len(self.ev_list):
            return self.get_waveform(self.ev_list[idx]), 1  # 1 for earthquake
        else:
            return self.get_waveform(self.no_list[idx - len(self.ev_list)]), 0  # 0 for noise

    
