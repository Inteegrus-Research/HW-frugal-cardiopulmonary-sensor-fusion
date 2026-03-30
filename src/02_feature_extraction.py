"""
02_feature_extraction.py
------------------------
Objective: Extract AASM-compliant Oxygen Desaturation Index (ODI) and 
ECG-Derived Respiration (EDR)/HRV features from raw PSG waveforms.
Implements strict Signal Quality Index (SQI) thresholding.
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import medfilt
from pyedflib import highlevel
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Directories
EDF_FOLDER = '../data/raw_edfs/'
COHORT_CSV = '../data/cohort_list.csv'
OUTPUT_CSV = '../data/extracted_features.csv'

def extract_features():
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    df_cohort = pd.read_csv(COHORT_CSV)
    df_cohort['mesaid'] = df_cohort['mesaid'].astype(float).astype(int).astype(str).str.zfill(4)
    total_patients = len(df_cohort)

    print(f"Initializing feature extraction for N={total_patients} patients...")

    for _, row in tqdm(df_cohort.iterrows(), total=total_patients, desc="Processing"):
        patient_id = row['mesaid']
        edf_path = os.path.join(EDF_FOLDER, f"mesa-sleep-{patient_id}.edf")
        
        if not os.path.exists(edf_path):
            continue
            
        try:
            edf_header = highlevel.read_edf_header(edf_path)
            ch_names = edf_header['channels']
            
            ecg_idx = next((i for i, ch in enumerate(ch_names) if 'EKG' in ch.upper() or 'ECG' in ch.upper()), None)
            spo2_idx = next((i for i, ch in enumerate(ch_names) if 'SPO2' in ch.upper()), None)
            
            if ecg_idx is None or spo2_idx is None:
                continue
                
            signals, signal_headers, _ = highlevel.read_edf(edf_path, ch_nrs=[ecg_idx, spo2_idx])
            ecg_signal, spo2_signal = signals[0], signals[1]
            
            fs_ecg = int(signal_headers[0].get('sample_rate', 256))
            fs_spo2 = int(signal_headers[1].get('sample_rate', 1))

            # --- A. SpO2 Processing (AASM Standard) ---
            spo2_quality_ok = 1
            spo2_clean = medfilt(spo2_signal, kernel_size=5)
            
            ct90_samples = np.sum(spo2_clean < 90)
            ct90_minutes = ct90_samples / fs_spo2 / 60.0 
            
            # Dynamic Baseline (120s trailing median)
            window_size = fs_spo2 * 120
            baseline = pd.Series(spo2_clean).rolling(window=window_size, min_periods=1).median().shift(1).fillna(method='bfill')
            
            drop_mask = (baseline - spo2_clean) >= 3
            min_samples_10s = 10 * fs_spo2
            events, current_run = 0, 0
            
            for val in drop_mask:
                if val:
                    current_run += 1
                else:
                    if current_run >= min_samples_10s:
                        events += 1
                    current_run = 0 
                    
            total_recording_hours = len(spo2_clean) / fs_spo2 / 3600.0
            odi = events / total_recording_hours if total_recording_hours > 0 else 0

            # --- B. Cardiac Processing & SQI ---
            ecg_quality_ok = 1
            rmssd, lf_hf_ratio, edr_var = np.nan, np.nan, np.nan
            
            # Central 4-hour extraction window
            start_sample = int(fs_ecg * 3600 * 2) 
            end_sample = int(fs_ecg * 3600 * 6)   
            
            if len(ecg_signal) < end_sample:
                start_sample = 0
                end_sample = len(ecg_signal)
                
            ecg_segment = ecg_signal[start_sample:end_sample]
            min_required_samples = 5 * 60 * fs_ecg
            
            # Signal variance and physiological limits check
            if len(ecg_segment) < min_required_samples or np.var(ecg_segment) < 0.001 or np.max(np.abs(ecg_segment)) > 10.0:
                ecg_quality_ok = 0
            else:
                try:
                    ecg_cleaned = nk.ecg_clean(ecg_segment, sampling_rate=fs_ecg)
                    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs_ecg)
                    
                    if len(rpeaks['ECG_R_Peaks']) > 10:
                        mean_rr = np.mean(np.diff(rpeaks['ECG_R_Peaks'])) / fs_ecg
                        mean_hr = 60 / mean_rr if mean_rr > 0 else 0
                        if mean_hr < 30 or mean_hr > 200:
                            ecg_quality_ok = 0
                    else:
                        ecg_quality_ok = 0
                    
                    if ecg_quality_ok == 1:
                        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs_ecg)
                        rmssd = hrv_time['HRV_RMSSD'].values[0]
                        
                        if len(rpeaks['ECG_R_Peaks']) > 300: 
                            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs_ecg)
                            lf_hf_ratio = hrv_freq['HRV_LFHF'].values[0]
                        
                        edr_signal = nk.ecg_rsp(ecg_cleaned, sampling_rate=fs_ecg)
                        edr_var = np.var(edr_signal)
                        
                except Exception:
                    ecg_quality_ok = 0

            # --- C. Compilation ---
            target_label = 1 if row['ahi_a0h4'] >= 15 else 0

            feature_dict = {
                'mesaid': patient_id,
                'Age': row['sleepage5c'],
                'BMI': row['bmi5c'],
                'ODI': round(odi, 2),
                'CT90': round(ct90_minutes, 2),
                'RMSSD': round(rmssd, 2) if ecg_quality_ok else np.nan,
                'LF_HF': round(lf_hf_ratio, 3) if ecg_quality_ok else np.nan,
                'EDR_var': round(edr_var, 4) if ecg_quality_ok else np.nan,
                'SpO2_Quality': spo2_quality_ok,
                'ECG_Quality': ecg_quality_ok,
                'Target_Apnea': target_label
            }
            
            df_single = pd.DataFrame([feature_dict])
            df_single.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
            
        except Exception:
            continue

    print(f"Feature extraction complete. Data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()
