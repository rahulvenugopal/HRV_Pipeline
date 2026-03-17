# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:55:51 2021
Explore HRV analysis on data acquired from brain products system
1. Check data quality
2. Do appropriate pre-processing for HRV analyss
3. Deploy Neurokit2 analysis
4. Visualisations

TODO
1. Epoch level analysis
2.Cut out ECG data based on markers (Eyes open, closed etc)
3. Understand and interpret the new parameters

Resources:
    1. Read the paper - https://link.springer.com/article/10.3758%2Fs13428-020-01516-y
    2. Github repo - https://github.com/neuropsychology/NeuroKit
    3. Interval-related Analysis - https://neurokit2.readthedocs.io/en/latest/examples/intervalrelated.html
    4. HRV - https://neurokit2.readthedocs.io/en/latest/examples/hrv.html

@author: Rahul Venugopal
"""
#%% Load libraries
import os
import numpy as np
import mne
from tkinter import filedialog
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

import neurokit2 as nk
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size']= 13

#%% Load data

data_dir = filedialog.askdirectory(title='Please select a directory with data files')

results_dir = filedialog.askdirectory(title='Please select a directory with data files')

os.chdir(data_dir)

filelist = glob('*.vhdr')

# Initialise a list to collect HRV measures
masterlist = []

for file in filelist:  
    
    os.chdir(data_dir) # Changing directory to data folder

    data = mne.io.read_raw_brainvision(file,preload=True)
    data.info
    srate = data.info['sfreq']
    
    # select ECG channels
    ecg_channels = ['ECG']
    
    data.pick_channels(ecg_channels)
    
    # visualise ECG data
    # data.plot()

    # HRV analysis pipleine
    ecg_data = np.squeeze(data.get_data())

    # Clean the ECG trace and returns a big dataframe with many params
    signals, info1 = nk.ecg_process(ecg_data,
                                    sampling_rate = srate)

     # Find peaks
    _, info2 = nk.ecg_peaks(signals["ECG_Clean"],
                            correct_artifacts=True,
                            sampling_rate = srate)
    
    # Let us move to results directory
    os.chdir(results_dir)
    
    # create a folder with subject's name
    os.makedirs(file[0:-5])

    # move to the subject folder
    os.chdir(file[0:-5])

    # Get time domain and frequency domain parameters
    hrv_variables = nk.hrv(info2, show=True)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig((file[0:-5] + '.png'),facecolor="white", bbox_inches="tight", dpi = 600)
    plt.close()

    # Visualise HRV and extract features
    nk.ecg_plot(signals, info = info2)
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig((file[0:-5] + '_qual_check.png'), facecolor="white", bbox_inches="tight", dpi = 600)
    plt.close()
    
    hrv_variables['Group'] = file[0:-5]
    
    # add the dataframe to the list
    masterlist.append(hrv_variables)

# Creating a dataframe from a list of dataframes
df = pd.concat(masterlist, axis=0)
df = df.reset_index(drop=True)

df.to_csv('hrv_parameters_mastersheet.csv',
                  index=None)