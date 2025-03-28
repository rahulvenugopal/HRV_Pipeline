# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:26:23 2025
- Understand what is happening under Neurokit2's hood

@author: Rahul Venugopal
"""

#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neurokit2 import signal_smooth
import scipy

# Get a sample 5 minutes data and pick 10 seconds
data = pd.read_csv('ecg_data_sorted.csv')["ECG"][0:10000]

# Flip ECG data
data = -1 * data

# See the raw data
plt.plot(data)
plt.title('A clean ECG segment')
plt.tight_layout()

fig = plt.gcf()
fig.savefig('Raw data.png', dpi = 600)
plt.close()

#%% Find the gradient and absolute of the same
grad = np.gradient(data)
absgrad = np.abs(grad)

# Smooth the absgrad signal with tthe following set parameters
sampling_rate=1000
smoothwindow=0.1
avgwindow=0.75
gradthreshweight=1.5
minlenweight=0.4
mindelay=0.3

# We have two steps - To reduce noise in the gradient signal and smooth further
# 10 percent of sampling rate datapoints are taken
smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
# 75 percent of srates
avg_kernel = int(np.rint(avgwindow * sampling_rate))

smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)

# A dynamic threshold is computed by multiplying avggrad by gradthreshweight
# This threshold helps distinguish QRS complexes from non-QRS regions
gradthreshold = gradthreshweight * avggrad

# 30 percent of sampling rate (One third of a second) as minimum delay between peaks
mindelay = int(np.rint(sampling_rate * mindelay))

# Identify Start and End of QRS Complexes
qrs = smoothgrad > gradthreshold # A binary mask
beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]

# Edge case check
end_qrs = end_qrs[end_qrs > beg_qrs[0]] # Ensures end_qrs comes after beg_qrs

# Get the count of QRS complexes
num_qrs = min(beg_qrs.size, end_qrs.size)
# Calculate QRS duration and take a mean and ignore short QRS complexes
# The minimum acceptable duration of QRS complexes are set at mean of QRS complexes
min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight

# Peep into the QRS windows and look for an R peak
peaks = [0]

for i in range(num_qrs):
    beg = beg_qrs[i]
    end = end_qrs[i]
    len_qrs = end - beg

    if len_qrs < min_len:
        continue

    # Find local maxima and their prominence within QRS.
    data_qrs = data[beg:end]
    # Find peaks index and return left base, right base and prominences
    locmax, props = scipy.signal.find_peaks(data_qrs, prominence=(None, None))

    if locmax.size > 0: # If a peak is detected
        # Identify most prominent local maximum.
        peak = beg + locmax[np.argmax(props["prominences"])]
        # Enforce minimum delay between peaks.
        if peak - peaks[-1] > mindelay:
            peaks.append(peak)

# Pop the first placeholder zero
peaks.pop(0)

peaks = np.asarray(peaks).astype(int)  # Convert to int

#%% Passing this to Neurokit algorithm

'''
ecg_process is a wrapper function which has the below functions
1. ecg_clean: cleaning
2. ecg_delineate: QRS complex delineation
3. ecg_peaks: peak detection
4. signal_rate: heart rate calculation
5. ecg_phase: cardiac phase determination
6. ecg_quality: signal quality assessment


'''
#%% Plotters
plt.plot(data_qrs)
plt.title('First QRS Complex')
plt.tight_layout()

fig = plt.gcf()
fig.savefig('9 QRS Complex.png', dpi = 600)
plt.close()


__, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(data)
ax1.title.set_text("Raw Trace plus R peaks detected")
ax2.plot(smoothgrad)
ax2.title.set_text("Onset and Offset of detected QRS complex")
ax2.plot(gradthreshold)
ax2.plot(qrs)

ax1.scatter(peaks, data[peaks], c="r")

plt.tight_layout()
fig = plt.gcf()
fig.savefig('10 Overlay identified R peak.png', dpi = 600)
plt.close()


for i in range(len(peaks)):
    ax1.text(peaks[i], np.max(data), f"{peaks[i]}", ha='center', fontsize=10, color='red')