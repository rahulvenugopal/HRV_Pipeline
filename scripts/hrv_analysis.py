# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:50:51 2026

HRV Analysis Pipeline using NeuroKit2

Processes ECG recordings to extract HRV params

Resources:
    1. NeuroKit2 Paper  : https://link.springer.com/article/10.3758%2Fs13428-020-01516-y
    2. GitHub Repo      : https://github.com/neuropsychology/NeuroKit
    3. Interval Analysis: https://neurokit2.readthedocs.io/en/latest/examples/intervalrelated.html
    4. HRV Docs         : https://neurokit2.readthedocs.io/en/latest/examples/hrv.html

Author  : Rahul Venugopal
"""
# Import libraries
import logging
import traceback
from pathlib import Path              # Safer, OS-agnostic path handling
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk

# Matplotlib guard: large ECG traces can exceed the default Agg renderer's
# path chunk limit, causing an OverflowError.  Setting this BEFORE any plot
# call prevents silent crashes on long recordings.
matplotlib.rcParams['agg.path.chunksize'] = 10000

# Use non-interactive Agg backend so plt.show() never blocks the script
# (important when running in batch / headless mode).
matplotlib.use('Agg')

# Setting up the folders
data_dir    = Path(filedialog.askdirectory(title='Select directory containing data files'))
results_dir = Path(filedialog.askdirectory(title='Select directory to save results'))

filelist = sorted(data_dir.glob('**/*.fif'))   # recurse into sub-folders

# Setting up the logger text file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),                          # still prints to terminal
        logging.FileHandler(results_dir / 'run_log.txt') # also saves to file
    ]
)

log = logging.getLogger(__name__)

if not filelist:
    log.error("No EEG files found under: %s", data_dir)
    raise SystemExit("No data files found — aborting.")

log.info("Found %d EEG file(s) to process.", len(filelist))

# Initilise a list to collect subject HRV DataFrames and concat once at the end 
masterlist = [] 

for file_no, eeg_path in enumerate(filelist, start=1):

    fname = eeg_path.stem
    log.info("── [%d/%d] Processing: %s", file_no,
             len(filelist),
             fname)

    subject_dir = results_dir / fname

    subject_dir.mkdir(parents=True, exist_ok=True)

    try:

        # Load raw data
        raw = mne.io.read_raw_fif(str(eeg_path), 
                                          preload=True,
                                          verbose=False)
        srate = raw.info['sfreq']
        log.info("   Sampling rate: %.1f Hz | Channels: %s",
                 srate,
                 raw.ch_names)

        # Determine ECG channel name (dataset-specific convention)
        if 'ECG1' in raw.ch_names:
            ecg_ch = 'ECG1'
        elif 'ECG' in raw.ch_names:
            ecg_ch = 'ECG'
        else:
            # Surface-level safety: log and skip rather than crash
            available = ', '.join(raw.ch_names)
            log.warning("   No ECG channel found (available: %s) — skipping.", available)
            continue

        # Extract the ECG channel as a plain 1D NumPy array
        ecg_raw_signal: np.ndarray = np.squeeze(raw.pick([ecg_ch]).get_data())
        
        log.info("   ECG signal length: %d samples (%.1f s)",
                 len(ecg_raw_signal), len(ecg_raw_signal) / srate)

        # Polarity correction
        # Some amplifiers (people by mistake) invert the ECG.
        # nk.ecg_invert() detects this and flips the signal if needed 
        # so that R-peaks are always positive.
        ecg_signal, is_inverted = nk.ecg_invert(
            ecg_raw_signal,
            sampling_rate=srate,
            show=False,
        )
        log.info("   Signal inverted: %s", is_inverted)
        
        # Primary ECG processing (cleaning + peak detection)
        signals, ecg_info = nk.ecg_process(ecg_signal, 
                                           sampling_rate=srate)

        # Signal Quality Index (SQI)
        # nk.ecg_quality() returns a per-sample score between 0 (unusable) and
        # 1 (perfect).  We average across the whole recording to get one number.
        # This must run AFTER ecg_process (we need ECG_Clean + R-peak locations)
        # but BEFORE we invest time in HRV computation on a bad signal.
        quality_scores = nk.ecg_quality(
            signals['ECG_Clean'],
            rpeaks=ecg_info['ECG_R_Peaks'],
            sampling_rate=srate,
        )
        mean_sqi = float(np.mean(quality_scores))
        log.info("Signal quality (SQI): %.3f  (0=unusable, 1=perfect)", mean_sqi)

        # Hard threshold: if the signal is too noisy, HRV metrics will be
        # unreliable.  We log a warning and skip rather than silently produce
        # bad numbers.  Threshold of 0.5 is a reasonable starting point —
        # adjust based on your recording conditions.
        SQI_THRESHOLD = 0.5
        if mean_sqi < SQI_THRESHOLD:
            log.warning("SQI %.3f < %.1f — signal too noisy, skipping %s.",
                        mean_sqi, SQI_THRESHOLD, fname)
            continue   # jump to next subject; this one is not added to masterlist
        
        # Under the hood, Neurokit uses a kickass algorithm
        # Read the paper -https://pubmed.ncbi.nlm.nih.gov/31314618/ by
        # Lipponen & Tarvainen (2019)
        
        # Instead of using hard physiological cutoffs It calculates a rolling
        # median of the surrounding RR intervals
        _, peaks_info = nk.ecg_peaks(
            signals['ECG_Clean'],
            correct_artifacts=True,   # removes ectopics and missed beats
            method='nabian2018',
            sampling_rate=srate,
        )

        # Overwrite the peak locations in the shared info dict so every
        # downstream call (hrv, ecg_plot, ecg_rate) uses identical peaks.
        ecg_info['ECG_R_Peaks']        = peaks_info['ECG_R_Peaks']
        ecg_info['ECG_fixpeaks_rr']    = peaks_info.get('ECG_fixpeaks_rr',
                                                         peaks_info['ECG_R_Peaks'])

        # Also update the binary peak column in the signals DataFrame so the
        # quality-check plot marks the correct beats.
        signals['ECG_R_Peaks'] = 0
        signals.loc[ecg_info['ECG_R_Peaks'], 'ECG_R_Peaks'] = 1

        # RR Interval Physiological Filtering + Interpolation
        # Convert R-peak sample indices → RR intervals in milliseconds.
        # np.diff() gives the gap between consecutive peaks in samples;
        # dividing by srate converts to seconds, ×1000 gives milliseconds.
        rr_ms = np.diff(ecg_info['ECG_R_Peaks']) / srate * 1000

        # Export RR interval time series as well
        tachogram = peaks_info['ECG_fixpeaks_rr']
        
        df_tacho = pd.DataFrame(tachogram, 
                                columns=['RR_Intervals'])
    
        # Export as csv
        df_tacho.to_csv(fname + '.csv', index=False)

        # Physiological bounds:
        #   300 ms → 200 BPM  (upper heart rate limit, e.g. during tachycardia)
        #  1500 ms →  40 BPM  (lower limit; below this is likely a missed beat)
        # Anything outside this window is noise, an ectopic, or a detection error.
        RR_MIN_MS = 300
        RR_MAX_MS = 1500

        bad_mask  = (rr_ms < RR_MIN_MS) | (rr_ms > RR_MAX_MS)
        n_bad     = int(np.sum(bad_mask))
        pct_bad   = 100.0 * n_bad / len(rr_ms) if len(rr_ms) > 0 else 0.0

        log.info("RR intervals — total: %d  |  implausible: %d (%.1f%%)",
                 len(rr_ms), n_bad, pct_bad)

        # Flag the recording quality in the output CSV so you can filter later.
        # >10% bad intervals is a common exclusion criterion in HRV literature.
        BAD_RR_THRESHOLD_PCT = 10.0
        if pct_bad > BAD_RR_THRESHOLD_PCT:
            data_quality_flag = 'POOR'
            log.warning("%.1f%% implausible RR intervals — "
                        "HRV metrics may be unreliable.", pct_bad)
        else:
            data_quality_flag = 'OK'            

        # Breathing rate estimated from ECG (ECG-derived respiration)
        ecg_rate_signal = nk.ecg_rate(
            peaks_info,
            sampling_rate=srate,
            desired_length=len(ecg_signal),
        )

        edr_signal = nk.ecg_rsp(ecg_rate_signal, sampling_rate=srate)

        rsp_signals, _ = nk.rsp_process(edr_signal, sampling_rate=srate)

        mean_rsp_rate = float(np.mean(rsp_signals['RSP_Rate']))
        sd_rsp_rate   = float(np.std(rsp_signals['RSP_Rate'],  ddof=1))
        # ddof=1 → sample std deviation (unbiased), more appropriate than
        # population std for a finite recording window.

        log.info("EDR — mean resp rate: %.2f  SD: %.2f (breaths/min)",
                 mean_rsp_rate, sd_rsp_rate)

        # Compute HRV (time-domain + frequency-domain + nonlinear)
        # nk.hrv() expects the info dict that contains the R-peak indices.
        # Passing show=True tells it to create an HRV summary figure.
        hrv_df = nk.hrv(ecg_info, sampling_rate=srate, show=True)
        
        # Save the figure
        hrv_fig = plt.gcf()
        hrv_fig.set_size_inches(18.5, 10.5)
        hrv_fig.savefig(subject_dir / f'{fname}_hrv_summary.png',
                        dpi=300)
        plt.close('all')

        # Attach metadata columns
        hrv_df['filename']          = fname
        hrv_df['is_inverted']       = is_inverted       # was signal flipped?
        hrv_df['mean_resp_rate']    = mean_rsp_rate     # EDR-derived resp rate
        hrv_df['sd_resp_rate']      = sd_rsp_rate
        hrv_df['n_rpeaks']          = len(ecg_info['ECG_R_Peaks'])  # beats kept
        # SQI columns
        hrv_df['mean_sqi']          = round(mean_sqi, 4)  # 0-1; below 0.5 = excluded
        # RR quality columns
        hrv_df['n_rr_total']        = len(rr_ms)
        hrv_df['n_rr_bad']          = n_bad               # count outside 300-1500 ms
        hrv_df['pct_rr_bad']        = round(pct_bad, 2)   # % implausible
        hrv_df['data_quality_flag'] = data_quality_flag   # 'OK' or 'POOR'

        # Quality-check plot  (cleaned ECG + R-peak annotations)
        # This plot lets you visually verify that peak detection was accurate.
        # Rows in the final CSV flagged as suspicious should be cross-checked
        # against this image.
        
        nk.ecg_plot(signals, info=ecg_info)
        qual_fig = plt.gcf()
        qual_fig.set_size_inches(18.5, 10.5)
        qual_fig.savefig(subject_dir / f'{fname}_quality_check.png', dpi=300)
        plt.close('all')

        # Accumulate into master list
        masterlist.append(hrv_df)
        log.info("Done — %d R-peaks retained.", len(ecg_info['ECG_R_Peaks']))

    except Exception:
        # Log the full traceback so you know exactly what failed and where,
        # but continue processing the remaining files.
        log.error("FAILED for %s:\n%s", fname, traceback.format_exc())
        continue   # ← skip to the next file;

# Aggregate results and save
if not masterlist:
    log.error("No subjects were processed successfully.  No CSV written.")
    raise SystemExit("Empty results — check logs above.")

master_df = pd.concat(masterlist, axis=0, ignore_index=True)

# Move identifier columns to the front for easier spreadsheet reading.
# Quality columns come first so they're immediately visible when you open the CSV.
id_cols     = ['filename', 'data_quality_flag', 'mean_sqi',
               'n_rpeaks', 'n_rr_total', 'n_rr_bad', 'pct_rr_bad',
               'is_inverted', 'mean_resp_rate', 'sd_resp_rate']
other_cols  = [c for c in master_df.columns if c not in id_cols]
master_df   = master_df[id_cols + other_cols]

output_csv = results_dir / 'hrv_parameters_mastersheet.csv'
master_df.to_csv(output_csv, index=False)

log.info("Pipeline complete.")
log.info("Subjects processed : %d / %d", len(masterlist), len(filelist))
log.info("Output CSV         : %s", output_csv)
