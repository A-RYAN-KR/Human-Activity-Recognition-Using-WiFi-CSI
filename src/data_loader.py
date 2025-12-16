import os
import re
import numpy as np
import csiread
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def load_and_preprocess_file(file_path):
    """
    Load CSI, flatten antenna dims to preserve spatial diversity, apply LPF.
    """
    configs_to_try = [(1, 3), (2, 3), (3, 3)]
    csi_data = None
    for ntx, nrx in configs_to_try:
        try:
            temp_csi = csiread.Intel(file_path, ntxnum=ntx, nrxnum=nrx)
            temp_csi.read()
            if temp_csi.csi is not None and temp_csi.csi.size > 0:
                csi_data = temp_csi
                break
        except Exception:
            continue
    
    if csi_data is None:
        return None

    csi_complex = csi_data.csi # (N, Nrx, Ntx, Sub)
    csi_complex = np.squeeze(csi_complex)

    # Flatten spatial dimensions: (Time, Features)
    if csi_complex.ndim > 2:
        T = csi_complex.shape[0]
        csi_complex = csi_complex.reshape(T, -1)

    amplitude = np.abs(csi_complex)

    # Low Pass Filter (Butterworth order 6)
    b, a = butter(6, 0.1, btype='low', analog=False)
    filtered = filtfilt(b, a, amplitude, axis=0)
    return filtered

def make_fixed_length_sequence(ts_data, seq_len):
    """
    Ensure fixed sequence length via crop or pad.
    """
    T, F = ts_data.shape
    # Downsample if extremely long
    if T > seq_len * 4:
        factor = T // (seq_len * 4)
        ts_data = ts_data[::factor, :]
    
    T = ts_data.shape[0]
    if T >= seq_len:
        start = (T - seq_len) // 2
        return ts_data[start:start+seq_len, :]
    else:
        pad = np.zeros((seq_len - T, F), dtype=ts_data.dtype)
        return np.vstack([ts_data, pad])

def augment_data(sequence):
    """
    Add noise and scaling to increase robustness.
    """
    # Jitter
    noise = np.random.normal(0, 0.02, sequence.shape)
    jittered = sequence + noise
    # Scale
    scaling_factor = np.random.normal(loc=1.0, scale=0.1)
    scaled = jittered * scaling_factor
    return scaled

def load_dataset(data_root_dir, seq_len=384):
    """
    Iterates through folders and loads .dat files.
    """
    # Define sub-paths based on your log structure
    # Note: Users should clone the repo into 'data_root_dir'
    sub_dirs = [
        'distance_factor_activity_data/1_alldata',
        'distance_factor_activity_data/3_alldata',
        'distance_factor_activity_data/6_alldata',
        'height_factor_activity_data/60_rawdata',
        'height_factor_activity_data/90_rawdata',
        'height_factor_activity_data/120_rawdata',
        'data/volunteer_a_all_data' 
    ]

    X_raw_list = []
    y_list = []
    label_pattern = re.compile(r'csi_a(\d+)_')

    print("Loading RAW Data...")
    
    for sub in sub_dirs:
        directory = os.path.join(data_root_dir, sub)
        if not os.path.exists(directory):
            print(f"Warning: Directory not found, skipping: {directory}")
            continue
        
        print(f"Processing: {directory}")
        files = sorted([f for f in os.listdir(directory) if f.endswith('.dat')])
        
        for filename in tqdm(files, desc="Loading files"):
            match = label_pattern.search(filename)
            if not match:
                continue
            
            label = int(match.group(1)) - 1
            file_path = os.path.join(directory, filename)
            
            ts_data = load_and_preprocess_file(file_path)
            
            if ts_data is None or ts_data.shape[0] < 10:
                continue
                
            seq = make_fixed_length_sequence(ts_data, seq_len)
            X_raw_list.append(seq)
            y_list.append(label)

    return X_raw_list, y_list
