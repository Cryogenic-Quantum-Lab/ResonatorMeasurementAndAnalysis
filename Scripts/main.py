# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:44:52 2025

@author: Paul Song and Khubilai Bayarsaikhan

Runs the full pipeline: take a segmented sweep of each resonator across all
power levels (measurement.py), then fit and plot the results (analysis.py).
"""
import sys
import datetime

import config
import measurement
import analysis

# Double checking parameters
if not config.check_parameter_lengths():
    sys.exit()

# Startup
sys.path.append(measurement.DEPENDENCY_DIR)  # Make sure all dependencies are in 'Dependencies' folder!
measurement.startup()

# Calculating time estimate
duration = sum(config.average) * 4.565 / 3600
now = datetime.datetime.now()
later = now + datetime.timedelta(hours=duration)

print("Estimated time of completion: " + str(duration) + " hours")
print("Current time: " + str(now))
print("Predicted time: " + str(later))

# Start sweeps and fitting
data_directory = measurement.remote_sweep(
    config.res_freq, config.phase_offset, config.tau_split_count,
    config.assumed_Qtot, config.bandwidth, config.edelay,
    config.magn_slope, config.magn_offset, config.directory,
    config.power_sequence, config.attn, config.average
)
analysis.user_fit(config.directory, measurement.DEPENDENCY_DIR, data_directory)
