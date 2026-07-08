# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:44:52 2025

@author: Paul Song and Khubilai Bayarsaikhan

Measurement side of the resonator sweep: connects to the VNA and the remote
attenuator, sweeps each resonator across all power levels, and saves the raw
data to CSV files.

Can be run on its own (see the commented block at the bottom of this file)
or imported and called from main.py alongside analysis.py.
"""

###### Connecting to remote attenuator: make sure to fill these requirements:
#   1: Python.Net (pip install pythonnet)
#   2: Mini-Circuits' DLL API file (mcl_RUDAT_NET45.dll)
#      https://www.minicircuits.com/softwaredownload/mcl_RUDAT64_DLL.zip
#      Note: - Windows may block the DLL file after download as a precaution
#            - Right-click on the file, select properties, click "Unblock" (if shown)

import pyvisa as visa
import numpy as np
import os
import pandas as pd
import time

startTime = time.perf_counter()
# Change this variable to the address of your instrument
VISA_ADDRESS = 'TCPIP0::Cryostat-Computer::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'

# Fixed, one-time setup: folder containing mcl_RUDAT_NET45.dll and the
# fit_resonator package. Shared across all experiments — not something to
# change per experiment (that's what config.py is for). If you ever move
# this folder, also update the matching constant in analysis.py.
DEPENDENCY_DIR = r"C:\Users\cryoq\Documents\ResonatorMeasurementAndAnalysis\Dependencies"

# Create a connection (session) to the instrument
resourceManager = visa.ResourceManager()
session = resourceManager.open_resource(VISA_ADDRESS)

# session timeout is how long (in ms) the instrument would run one a code before it kicks you out
timeout = 43200000  # timeout after 12 hours
session.timeout = timeout
startTime = time.perf_counter()


# This will be the startup function to setup the VNA, was made separately for testing purposes
def startup():
    # Command to preset the instrument and deletes the default trace, measurement, and window
    session.write("SYST:FPR")

    # timeout settings
    session.write(f"SYST:COMM:VISA:RDEV:TIM {timeout}")

    # Select the measurement
    # Choose to calculate at specific S## here
    session.write("CALC1:MEAS1:DEF 'S21'")
    session.write("CALC1:MEAS2:DEF 'S21'")

    # Choose to calculate in different forms here
    session.write("CALC1:MEAS1:FORM MLOG")
    session.write("CALC1:MEAS2:FORM PHASe")
    # NOTE, if you want to change the data type, change the phrase after FORM

    # Turn on 2 windows, you can copy and paste this with numbers incremented to make more screens
    session.write("DISP:WIND1:STAT ON")
    session.write("DISP:WIND2:STAT ON")

    # Displays measurement 1 in window 1, measurement 2 in window 2
    # The FEED command assigns the next available trace number to the measurement
    session.write("DISP:MEAS1:FEED 1")
    session.write("DISP:MEAS2:FEED 2")

    # sets sweep type to segment
    session.write("SENS1:SWE:TYPE SEGM")

    # Segments can now overlap
    session.write("SENS1:SEGM:ARB ON")

    # Display the segment table
    # session.write("DISP:WIND2:TABL SEGM")

    print("Startup finished")


# This is a very important function. It essentially creates the segment table, used as a final function to setup the table.
# You feed it a np array for the segments in the form of, {CSPAN,num_segments,1,1,freq,freq,.....} and repeat the last four for each segment
# The first 1 is to turn on the segment, second number is for number of points per segment, and since we chose SSTOP, freq freq are both just the same resonance freq
# 1 segment state, 2 points, 3 start or center, 4 stop or span, IFBW, Dwell time, Power
def set_segment(scan_type, num_segments, segments):
    if scan_type not in ("CSPAN", "SSTOP"):
        raise ValueError("Please choose scan_type between CSPAN (center span freq) or SSTOP (start stop freq)")
    else:
        session.write(f"SENS:SEGM:LIST {scan_type},{num_segments},{segments}")
        # Display the segment table
        session.write("DISP:WIND2:TABL SEGM")
        # Take a sweep of the data once, to change it, change SING. Refer to commands list
        # This will then put the VNA program reading on HOLD on the bottom right next to the Bandwidth
        session.write("SENS1:SWE:MODE SING")

        # Keep the controller and the VNA "synched"
        session.query("*OPC?")


# This is just a function to set the parameters
def paraSet(bandwidth, averages, powerLevel, edelay, phase_offset, magn_slope, magn_offset):
    # bandwidth
    session.write(f"SENS1:BWID {bandwidth}KHZ")

    # Enable averaging
    session.write("SENS1:AVER ON")

    # Set averaging to sweep instead of point by point SWEEP or POINt
    session.write("SENS1:AVER:MODE POINt")

    # average
    session.write(f"SENS1:AVER:COUN {averages}")

    # sets the power level
    session.write(f"SOUR1:POW {powerLevel}")

    # sets electrical delay correction
    edelay = edelay * 1E-9
    session.write(f"CALC1:MEAS2:CORR:EDEL:TIME {edelay}")
    # sets the phase offset
    session.write(f"CALC1:MEAS2:OFFS:PHAS {phase_offset}")
    # sets the slope of the magnitude correction
    session.write(f"CALC1:OFFS1:MAGN:SLOP {magn_slope}")
    # sets the magnitude offset
    session.write(f"CALC1:OFFS1:MAGN {magn_offset}")


# This is just a small helper function to make creating the segments easier
def build_segment(points, freq):
    return f"1,{points},{freq},{freq}"


#### THIS IS THE MAIN FUNCTION THAT COLLECTS DATA!
def remote_sweep(res_freq, phase_offset, tau_split_count, assumed_Qtot, bandwidth, edelay, magn_slope, magn_offset, directory, power_sequence, attn, average):
    # This section is to connect to the remote attenuator
    import clr  # pythonnet
    clr.AddReference('mcl_RUDAT_NET45')  # Reference the DLL

    from mcl_RUDAT_NET45 import USB_RUDAT
    att = USB_RUDAT()  # Create an instance of the USB control class

    Status = att.Connect()  # Connect

    if Status[0] <= 0:
        print("Could not connect to attenuator.")
        exit

    Status = att.Send_SCPI(f":SETATT={attn}", "")  # Set attenuation to first attenuation value

    # Create data directory
    data_path = os.path.join(directory, "Data")
    os.makedirs(data_path, exist_ok=True)

    attn_start = attn  # remember the starting attenuation so we can reset it for each resonator

    res_counter = 1
    for f in res_freq:
        # Reset attenuation (and the physical attenuator) back to the starting
        # value at the beginning of every resonator's power sweep, since the
        # attn switch below only applies within a single resonator's pass
        # through power_sequence.
        attn = attn_start
        Status = att.Send_SCPI(f":SETATT={attn}", "")

        ave_counter = 0
        for powerLevel in power_sequence:
            powerLevelname = powerLevel - 68 - attn

            paraSet(bandwidth, average[ave_counter], powerLevel, edelay, phase_offset[res_counter - 1], magn_slope[res_counter - 1], magn_offset)
            adjusted_Qtot = assumed_Qtot[res_counter - 1]  # This is the Qtot we will use to calculate the segment frequencies, can be changed to be different from the assumed Qtot if we want
            startTimeLocal = time.perf_counter()
            file_name = (f"Resonator_{res_counter}_")
            # initialize freq list
            freq_list = np.empty((0,))
            # make a list of omega
            omega = np.linspace(-(24 * np.pi) / 25, (24 * np.pi) / 25, tau_split_count)
            # create a new frequency for each omega and add to list
            for t in omega:
                freq = f * (1 + (1 / (2 * adjusted_Qtot)) * np.tan(t / 2))
                freq_list = np.append(freq_list, freq)
            # sort from lowest to highest
            freq_list = np.sort(freq_list)

            # start adding into segment table
            segments = []
            for x in freq_list:
                segments.append(build_segment(1, x))
            seg_data = ",".join(segments)
            # adds segments made from frequency list to segment list
            set_segment("SSTOP", tau_split_count, seg_data)

            # start transferring data to csv and output etc
            # Default Data Format is ASCII, we set data taking format here, this is the fastest setting and if you want to change it, refer to command tree
            session.write("FORM:DATA ASCII,0")
            # Ask for the data from the sweep, pick one of the locations to read
            # If you want more measurements, I recommend to add more myMeas# and follow the coding convention
            myMeas1 = session.query_ascii_values("CALC1:MEAS1:DATA:FDATA?", container=np.array)
            myMeas2 = session.query_ascii_values("CALC1:MEAS2:DATA:FDATA?", container=np.array)

            # Here we attempt to put all the data into an array
            finalMeas = []

            # Adds columns of data to print to csv file
            # adds frequency
            finalMeas.append(freq_list)

            # Based on the type of form chosen in getting data, we add them as columns to final Meas
            finalMeas.append(myMeas1)
            finalMeas.append(myMeas2)
            finalMeas = np.transpose(finalMeas)

            # Create the csv file path
            csv_file_path = os.path.join(data_path, file_name + str(powerLevelname) + "dBm_" + str(attn) + "VARAT.csv")

            # Ensure the directory exists, if not, create it
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Convert the ndarray to a DataFrame
            df = pd.DataFrame(finalMeas, columns=['frequency', 'dBm', 'phase'])

            # Save the DataFrame to a CSV file
            df.to_csv(csv_file_path, index=False)
            # prints file path confirmation
            print('Measurement successful. CSV file saved at:')
            print(csv_file_path + "\n")

            # Prints the elapsed time for the measurement. Useful when we plan our measurements
            endTime = time.perf_counter()
            elapsed = endTime - startTime
            print(f'Total time so far to run this measurement: {elapsed:.3f} secounds\n')
            endTimeLocal = time.perf_counter()
            elapsedLocal = endTimeLocal - startTimeLocal
            print(f"Time taken to measure {file_name}{powerLevelname}dBm: {elapsedLocal:.3f} secounds\n")
            print(f"average is {average[ave_counter]}")
            print(f"power is {powerLevel}")
            print(f"attenuation is -{attn}")
            print('-------------------------------------------')

            ave_counter = ave_counter + 1

            if powerLevel == -40:  # End of first batch
                attn = 60  # THIS IS THE SECOND ATTENUATION VALUE!!!
                Status = att.Send_SCPI(f":SETATT={attn}", "")

        res_counter += 1

    att.Disconnect()  # disconnects w/attenuator, not sure if this is important
    return data_path


# =============================================================================
# Uncomment the block below to run measurement.py on its own (no analysis).
# =============================================================================
# import sys
# import datetime
# import config
#
# if config.check_parameter_lengths():
#     sys.path.append(DEPENDENCY_DIR)  # Make sure all dependencies are in 'Dependencies' folder!
#     startup()
#
#     # Calculating time estimate
#     duration = sum(config.average) * 4.565 / 3600
#     now = datetime.datetime.now()
#     later = now + datetime.timedelta(hours=duration)
#     print("Estimated time of completion: " + str(duration) + " hours")
#     print("Current time: " + str(now))
#     print("Predicted time: " + str(later))
#
#     data_directory = remote_sweep(
#         config.res_freq, config.phase_offset, config.tau_split_count,
#         config.assumed_Qtot, config.bandwidth, config.edelay,
#         config.magn_slope, config.magn_offset, config.directory,
#         config.power_sequence, config.attn, config.average
#     )
#     print(f"Data saved to: {data_directory}")
