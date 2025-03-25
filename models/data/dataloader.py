import os
import pandas as pd
import numpy as np
import torch
import h5py
import torch.utils.data as data
from sklearn.model_selection import train_test_split

class STEADDataset(data.Dataset):
    def __init__(self, file, csv_file, transform=None):
        self.file = file
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.edf = self.df[(self.df.trace_category == "earthquake_local") & (self.df.source_distance_km <= 30) & (self.df.source_magnitude > 3)]
        self.ndf = self.df[(self.df.trace_category == 'noise')]
        self.ev_list = self.edf['trace_name'].tolist()
        self.no_list = self.ndf['trace_name'].tolist()
        self.splits = self.split_data()
    
    def __len__(self):
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
    
    def get_waveform(self, trace_name):
        with h5py.File(self.file, "r") as f:
            dataset = f.get(f"data/{trace_name}")
            if dataset is None:
                raise ValueError(f"Trace {trace_name} not found in HDF5 file.")
            
            data = np.array(dataset)
            data = torch.tensor(data, dtype=torch.float32)

            if self.transform:
                data = self.transform(data)

        return data
    
    def get_train_data(self):
        return [self.get_waveform(trace) for trace in self.splits["eq_train"]]
    
    def get_test_data(self):
        test_waveforms = [self.get_waveform(trace) for trace in self.splits["eq_test"]]
        noise_waveforms = [self.get_waveform(trace) for trace in self.splits["noise_test"]]
        return test_waveforms + noise_waveforms
    
