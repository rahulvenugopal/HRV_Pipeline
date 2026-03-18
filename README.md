# HRV analysis
- Batch Processing: Recursively churns through all files in a selected directory without crashing.
- Auto-detects the ECG channel polarity and automatically corrects inverted signals.
- Calculates a Signal Quality Index (SQI) and skips recordings that are too noisy to trust.
- Uses the Lipponen algorithm for ectopics/missed beats, plus a custom cubic-spline interpolation for massive outliers to keep our time-series continuous and pure.
- Extracts breathing rates (EDR) directly from the heart rate signal.
- Generates quality-check plots and HRV summary images for every subject, compiled alongside a single, clean master CSV with all our metrics.

# Neurokit2 based analysis and viz
Raw ECG trace looked like this with baseline correction
![Raw trace](https://github.com/rahulvenugopal/HRV_adventures/blob/main/ECGLeadPlacementVerification/After_baseline_correction.jpg)
---

Time domain, Frequency domain and Non-linear methods
![Parameters](https://github.com/rahulvenugopal/HRV_adventures/blob/main/ECGLeadPlacementVerification/All_parameters_HRV.png)
---

General visualisations
![Overall viz](https://github.com/rahulvenugopal/HRV_adventures/blob/main/ECGLeadPlacementVerification/Summary_HRV.png)
---

# Documentation of features from functions [page](https://github.com/neuropsychology/NeuroKit/tree/master/neurokit2/hrv)
- Frequency domain
`ulf=(0, 0.0033)`
`vlf=(0.0033, 0.04)`
`lf=(0.04, 0.15)`
`hf=(0.15, 0.4)`
`vhf=(0.4, 0.5)`
- For instance, 1, 2 and 5 minutes of high quality signal are the recommended minima for HF, LF and LF/HF, respectively
---

# **Characteristics of the Poincaré Plot Geometry**:
- **SD1**: SD1 is a measure of the spread of RR intervals on the Poincaré plot perpendicular to the line of identity.
It is an index of short-term RR interval fluctuations, i.e., beat-to-beat variability.
It is equivalent (although on another scale) to RMSSD, and therefore it is redundant to report correlations with both (Ciccone, 2017).
- **SD2**: SD2 is a measure of the spread of RR intervals on the Poincaré plot along the line of identity.
It is an index of long-term RR interval fluctuations.
- **SD1SD2**: the ratio between short and long term fluctuations of the RR intervals (SD1 divided by SD2).
- **S**: Area of ellipse described by SD1 and SD2 (``pi * SD1 * SD2``). It is proportional to *SD1SD2*.
- **CSI**: The Cardiac Sympathetic Index (Toichi, 1997), calculated by dividing the longitudinal variability of the Poincaré plot (``4*SD2``) by its transverse variability (``4*SD1``).
- **CVI**: The Cardiac Vagal Index (Toichi, 1997), equal to the logarithm of the product of longitudinal (``4*SD2``) and transverse variability (``4*SD1``).
- **CSI_Modified**: The modified CSI (Jeppesen, 2014) obtained by dividing the square of the longitudinal variability by its transverse variability.
- **Indices of Heart Rate Asymmetry (HRA), i.e., asymmetry of the Poincaré plot** (Yan, 2017)
- **GI**: Guzik's Index, defined as the distance of points above line of identity (LI) to LI divided by the distance of
all points in Poincaré plot to LI except those that are located on LI.
- **SI**: Slope Index, defined as the phase angle of points above LI divided by the phase angle of all points in Poincaré plot except
those that are located on LI.
- **AI**: Area Index, defined as the cumulative area of the sectors corresponding to the points that are located above LI divided by the
cumulative area of sectors corresponding to all points in the Poincaré plot except those that are located on LI.
- **PI**: Porta's Index, defined as the number of points below LI divided by the total number of points in Poincaré plot except those that are located on LI.
- **SD1d** and **SD1a**: short-term variance of contributions of decelerations (prolongations of RR intervals) and
accelerations (shortenings of RR intervals), respectively (Piskorski,  2011).
- **C1d** and **C1a**: the contributions of heart rate decelerations and accelerations to short-term HRV, respectively (Piskorski,  2011).
- **SD2d** and **SD2a**: long-term variance of contributions of decelerations (prolongations of RR intervals) and
accelerations (shortenings of RR intervals), respectively (Piskorski,  2011).
- **C2d** and **C2a**: the contributions of heart rate decelerations and accelerations to long-term HRV, respectively (Piskorski,  2011).
- **SDNNd** and **SDNNa**: total variance of contributions of decelerations (prolongations of RR intervals) and
accelerations (shortenings of RR intervals), respectively (Piskorski,  2011).
- **Cd** and **Ca**: the total contributions of heart rate decelerations and accelerations to HRV.
- **Indices of Heart Rate Fragmentation** (Costa, 2017)
- **PIP**: Percentage of inflection points of the RR intervals series.
- **IALS**: Inverse of the average length of the acceleration/deceleration segments.
- **PSS**: Percentage of short segments.
- **PAS**: IPercentage of NN intervals in alternation segments.

# **Indices of Complexity:**
- **ApEn**: The approximate entropy measure of HRV, calculated by `entropy_approximate()`.
- **SampEn**: The sample entropy measure of HRV, calculated by `entropy_sample()`.
- **ShanEn**: The Shannon entropy measure of HRV, calculated by `entropy_shannon()`.
- **FuzzyEn**: The fuzzy entropy measure of HRV, calculated by `entropy_fuzzy()`.
- **MSE**: The multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.
- **CMSE**: The composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.
- **RCMSE**: The refined composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.
- **CD**: The Correlation Dimension of the HR signal, calculated by `fractal_correlation()`.
- **HFD**: The Higuchi's Fractal Dimension of the HR signal, calculated by `fractal_higuchi()`. kmax is set to "default".
- **KFD**: The Katz's Fractal Dimension of the HR signal, calculated by `fractal_katz()`.
- **LZC**: The Lempel-Ziv complexity of the HR signal, calculated by `fractal_lempelziv()`.
- **DFA_alpha1**: The monofractal detrended fluctuation analysis of the HR signal corresponding to short-term correlations, calculated by `fractal_dfa()`.
- **DFA_alpha2**: The monofractal detrended fluctuation analysis of the HR signal corresponding to long-term correlations, calculated by `fractal_dfa()`.
- **DFA_alpha1_ExpRange**: The multifractal detrended fluctuation analysis of the HR signal 
corresponding to short-term correlations, calculated by `fractal_dfa()`. ExpRange is the range of singularity exponents,
correspoinding to the width of the singularity spectrum.
- **DFA_alpha2_ExpRange**: The multifractal detrended fluctuation analysis of the HR signal corresponding to long-term correlations,
calculated by `fractal_dfa()`. ExpRange is the range of singularity exponents, correspoinding to the width of the singularity spectrum.
- **DFA_alpha1_ExpMean**: Multifractal DFA. ExpMean is the mean of singularity exponents.
- **DFA_alpha2_ExpMean**: Multifractal DFA. ExpMean is the mean of singularity exponents.
- **DFA_alpha1_DimRange**: The multifractal detrended fluctuation analysis of the HR signal corresponding to short-term correlations,
calculated by `fractal_dfa()`. DimRange is the range of singularity dimensions, correspoinding to the height of the singularity spectrum.
- **DFA_alpha2_DimRange**: The multifractal detrended fluctuation analysis of the HR signal corresponding to long-term correlations,
calculated by `fractal_dfa()`. DimRange is the range of singularity dimensions, correspoinding to the height of the singularity spectrum.
- **DFA_alpha1_DimMean**: Multifractal DFA. Dimmean is the mean of singularity dimensions.
- **DFA_alpha2_DimMean**: Multifractal DFA. Dimmean is the mean of singularity dimensions.
---

**References:**
- Yan, C., Li, P., Ji, L., Yao, L., Karmakar, C., & Liu, C. (2017). Area asymmetry of heart
    rate variability signal. Biomedical engineering online, 16(1), 112.
- Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.
    (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
    56(4), 674-678.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.
- Costa, M. D., Davis, R. B., & Goldberger, A. L. (2017). Heart rate fragmentation: a new
    approach to the analysis of cardiac interbeat interval dynamics. Front. Physiol. 8, 255 (2017).
- Jeppesen, J., Beniczky, S., Johansen, P., Sidenius, P., & Fuglsang-Frederiksen, A. (2014).
    Using Lorenz plot and Cardiac Sympathetic Index of heart rate variability for detecting seizures
    for patients with epilepsy. In 2014 36th Annual International Conference of the IEEE Engineering
    in Medicine and Biology Society (pp. 4563-4566). IEEE.
- Piskorski, J., & Guzik, P. (2011). Asymmetric properties of long-term and total heart rate
    variability. Medical & biological engineering & computing, 49(11), 1289-1297.
- Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.
- Brennan, M. et al. (2001). Do Existing Measures of Poincaré Plot Geometry Reflect Nonlinear
    Features of Heart Rate Variability?. IEEE Transactions on Biomedical Engineering, 48(11), 1342-1347.
- Toichi, M., Sugiura, T., Murai, T., & Sengoku, A. (1997). A new method of assessing cardiac
    autonomic function and its comparison with spectral analysis and coefficient of variation of R–R
    interval. Journal of the autonomic nervous system, 62(1-2), 79-84.
- Acharya, R. U., Lim, C. M., & Joseph, P. (2002). Heart rate variability analysis using
    correlation dimension and detrended fluctuation analysis. Itbm-Rbm, 23(6), 333-339.
---

# **Time domain HRV metrics:**
- **MeanNN**: The mean of the RR intervals.
- **SDNN**: The standard deviation of the RR intervals.
- **SDANN1**, **SDANN2**, **SDANN5**: The standard deviation of average RR intervals extracted from n-minute segments of
        time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed
        (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
- **SDNNI1**, **SDNNI2**, **SDNNI5**: The mean of the standard deviations of RR intervals extracted from n-minute
        segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to
        be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
- **RMSSD**: The square root of the mean of the sum of successive differences between
        adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
        therefore it is redundant to report correlations with both (Ciccone, 2017).
- **SDSD**: The standard deviation of the successive differences between RR intervals.
- **CVNN**: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR
        intervals (MeanNN).
- **CVSD**: The root mean square of the sum of successive differences (RMSSD) divided by the
        mean of the RR intervals (MeanNN).
- **MedianNN**: The median of the absolute values of the successive differences between RR intervals.
- **MadNN**: The median absolute deviation of the RR intervals.
- **HCVNN**: The median absolute deviation of the RR intervals (MadNN) divided by the median
        of the absolute differences of their successive differences (MedianNN).
- **IQRNN**: The interquartile range (IQR) of the RR intervals.
- **pNN50**: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
- **pNN20**: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
- **TINN**: A geometrical parameter of the HRV, or more specifically, the baseline width of
        the RR intervals distribution obtained by triangular interpolation, where the error of least
        squares determines the triangle. It is an approximation of the RR interval distribution.
- **HTI**: The HRV triangular index, measuring the total number of RR intervals divded by the
        height of the RR intervals histogram.	

**References:**
- Ciccone, A. B., Siedlik, J. A., Wecht, J. M., Deckert, J. A., Nguyen, N. D., & Weir, J. P.
    (2017). Reminder: RMSSD and SD1 are identical heart rate variability metrics. Muscle & nerve,
    56(4), 674-678.
- Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
    electrophysiology review, 6(3), 239-244.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms.
    Frontiers in public health, 5, 258.

# Create a cookbook for HRV analysis + Video tutorial **(future idea/hobby project/ intern project)**