# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:44:52 2025

@author: Khubilai Bayarsaikhan

This script was made to take a segmented sweep of code, and analyze it
"""

###### Connecting to remote attenuator: make sure to fill these requirements:
#   1: Python.Net (pip install pythonnet)
#   2: Mini-Circuits' DLL API file (mcl_RUDAT_NET45.dll)
#      https://www.minicircuits.com/softwaredownload/mcl_RUDAT64_DLL.zip
#      Note: - Windows may block the DLL file after download as a precaution
#            - Right-click on the file, select properties, click "Unblock" (if shown)

import clr # pythonnet
clr.AddReference('mcl_RUDAT_NET45')      # Reference the DLL

from mcl_RUDAT_NET45 import USB_RUDAT
att = USB_RUDAT()   # Create an instance of the USB control class

Status = att.Connect()       # Connect

if Status > 0:
    continue

else:
    print ("Could not connect to attenuator.")



# Importing all the necessary libraries
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import time


startTime = time.perf_counter()
# Change this variable to the address of your instrument
VISA_ADDRESS = 'TCPIP0::QuCAT_Lab::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'

# Create a connection (session) to the instrument
resourceManager = visa.ResourceManager()
session = resourceManager.open_resource(VISA_ADDRESS)

#session timeout is how long (in ms) the instrument would run one a code before it kicks you out
timeout = 43200000 #timeout after 12 hours
session.timeout = timeout
startTime = time.perf_counter()

#This will be the startup function to setup the VNA, was made separately for testing purposes
def startup():
    # Command to preset the instrument and deletes the default trace, measurement, and window
    session.write("SYST:FPR")
    
    #timeout settings
    session.write(f"SYST:COMM:VISA:RDEV:TIM {timeout}")
    
    # Select the measurement
    #Choose to calculate at specific S## here
    session.write("CALC1:MEAS1:DEF 'S21'")
    session.write("CALC1:MEAS2:DEF 'S21'")
    
    #Choose to calculate in different forms here
    session.write("CALC1:MEAS1:FORM MLOG")
    session.write("CALC1:MEAS2:FORM PHASe")
    #NOTE, if you want to change the data type, change the phrase after FORM
    
    # Turn on 2 windows, you can copy and paste this with numbers incremented to make more screens
    session.write("DISP:WIND1:STAT ON")
    session.write("DISP:WIND2:STAT ON")
    
    # Displays measurement 1 in window 1, measurement 2 in window 2
    # The FEED command assigns the next available trace number to the measurement
    session.write("DISP:MEAS1:FEED 1")
    session.write("DISP:MEAS2:FEED 2")
    
    #sets sweep type to segment
    session.write("SENS1:SWE:TYPE SEGM")
    
    #Segments can now overlap
    session.write("SENS1:SEGM:ARB ON")
    
    #Display the segment table
   # session.write("DISP:WIND2:TABL SEGM")
    
    print("Startup finished")
    

#here you can set parameters for what options you want to turn on in terms of each segment, but not useful for our needs
'''
def options_settings(BWID,TIME,POW):
    if not all(x in (0, 1) for x in (BWID,TIME,POW)):
        raise ValueError("please input 0 (OFF) or 1 (ON) for option settings(Bandwidth,Time,Power)")
    else:
        session.write(f"SENS:SEGM:BWID:CONT {BWID}")
        session.write(f"SENS:SEGM:SWE:TIME:CONT {TIME}")
        session.write(f"SENS:SEGM:POW:CONTROL {POW}")
'''
    
    # ======================== Getting data ========================
    
#This is a very important function. It essentially creates the segment table, used as a final function to setup the table. 
#You feed it a np array for the segments in the form of, {CSPAN,num_segments,1,1,freq,freq,.....} and repeat the last four for each segment
#The first 1 is to turn on the segment, second number is for number of points per segment, and since we chose SSTOP, freq freq are both just the same resonance freq
#1 segment state, 2 points, 3 start or center, 4 stop or span, IFBW, Dwell time, Power
def set_segment(scan_type,num_segments,segments):
    if scan_type not in ("CSPAN","SSTOP"):
        raise ValueError("Please choose scan_type between CSPAN (center span freq) or SSTOP (start stop freq)")
    else:
        session.write(f"SENS:SEGM:LIST {scan_type},{num_segments},{segments}")
        #Display the segment table
        session.write("DISP:WIND2:TABL SEGM")
        # Take a sweep of the data once, to change it, change SING. Refer to commands list
        #This will then put the VNA program reading on HOLD on the bottom right next to the Bandwidth
        session.write("SENS1:SWE:MODE SING")
        
        # Keep the controller and the VNA "synched"
        session.query("*OPC?")
 
#This is just a function to set the parameters
def paraSet(bandwidth, averages, powerLevel, edelay, phase_offset, magn_slope, magn_offset):    
    #bandwidth
    session.write(f"SENS1:BWID {bandwidth}KHZ")
    
    #Enable averaging
    session.write("SENS1:AVER ON")
    
    #Set averaging to sweep instead of point by point SWEEP or POINt
    session.write("SENS1:AVER:MODE POINt")
    
    #average
    session.write(f"SENS1:AVER:COUN {averages}")
    
    #sets the power level
    session.write(f"SOUR1:POW {powerLevel}")
    
    #sets electrical delay correction
    edelay = edelay*1E-9
    session.write(f"CALC1:MEAS2:CORR:EDEL:TIME {edelay}")
    #sets the phase offset
    session.write(f"CALC1:MEAS2:OFFS:PHAS {phase_offset}")
    #sets the slope of the magnitude correction
    session.write(f"CALC1:OFFS1:MAGN:SLOP {magn_slope}")
    #sets the magnitude offset
    session.write(f"CALC1:OFFS1:MAGN {magn_offset}")
        
#This is just a small helper function to make creating the segments easier  
def build_segment(points, freq):
    return f"1,{points},{freq},{freq}"    


#input parameters ==================================================================================================================
res_freq = [5.791241319E9,5.835566733E9,5.922292790E9,6.321501215E9,6.370486998E9,6.400447515E9,6.425018282E9,6.479538603E9] #these are centers of resonances
phase_offset = [360-345,360-350,360-20,360-15,360-0,360-10, 360-0, 360-0]


tau_split_count = 51 #essentially the num of segments
assumed_Qtot = 200000 #together with tau split cout detmermines our calculations for the spread of our segmented sweep

bandwidth = 0.01 #in khz


edelay = 57.4
magn_slope = 0
magn_offset = 2


directory = (r"C:\Users\Lab_Admin\OneDrive\Desktop\Shared Data\Stephanie_NorthwesternResonators\Resonator 2")
file_name = ("test")

power_sequence = [5, 0, -5, -10, -15, -20, -25, -30, -35, -40,
                  -5, -10, -15, -20, -25, -30, -35] # First batch to -40 for att -17, then -57
att = -17 # Starts at -17 for first batch, will change to -57 later in code

#Code start =====================================================================================================================
startup() #start up settings

Status = att.Send_SCPI(f":SETATT={att}", "") # Set attenuation
#Responses = att.Send_SCPI(":ATT?", "")
#print (str(Responses[1]))

ave_counter = 0
average = [20,20,20,20,20,20,20,20,20,20,20,20,20,80,320,1280,4000]
for powerLevel in power_sequence:
    powerLevelname = att - 68 + powerLevel
    
    res_counter = 1    
    for f in res_freq:
        paraSet(bandwidth, average[ave_counter], powerLevel, edelay, phase_offset[res_counter-1], magn_slope, magn_offset)
        startTimeLocal = time.perf_counter()
        file_name = (f"Resonator_{res_counter}_")
        #initialize freq list
        freq_list = np.empty((0,))
        #make a list of omega
        omega = np.linspace(-(24*np.pi)/25, (24*np.pi)/25, tau_split_count)
        #create a new frequency for each omega and add to list
        for t in omega:
            freq = f*(1+(1/(2*assumed_Qtot))*np.tan(t/2))
            freq_list = np.append(freq_list,freq)
        #sort from lowest to highest
        freq_list = np.sort(freq_list)
        
        
        #start adding into segment table
        segments = []
        for x in freq_list:    
            segments.append(build_segment(1, x)) 
        seg_data = ",".join(segments)
        #adds segments made from frequency list to segment list
        set_segment("SSTOP", tau_split_count, seg_data)
        
        
        #start transfering data to csv and output etc
        # Default Data Format is ASCII, we set data taking format here, this is the fastest setting and if you want to change it, refer to command tree
        session.write("FORM:DATA ASCII,0")
        # Ask for the data from the sweep, pick one of the locations to read
        #If you want more measurements, I reccomend to add more myMeas# and follow the coding convention 
        myMeas1 = session.query_ascii_values("CALC1:MEAS1:DATA:FDATA?", container=np.array)
        myMeas2 = session.query_ascii_values("CALC1:MEAS2:DATA:FDATA?", container=np.array)                
         
        #Here we attempt to put all the data into an array
        finalMeas = []
        
        #Adds columns of data to print to csv file
        #adds frequency
        finalMeas.append(freq_list)
        
        #Based on the type of form chosen in getting data, we add them as columns to final Meas
        finalMeas.append(myMeas1)
        finalMeas.append(myMeas2)
        finalMeas = np.transpose(finalMeas)
        
        #prints the data in code, can comment out
        #print(finalMeas)
        
        # Create the full file path
        file_path = os.path.join(directory, file_name)
        csv_file_path = os.path.join(directory, file_name+str(powerLevelname)+"dBm_30VARAT.csv")
        
        # Ensure the directory exists, if not, create it
        if not os.path.exists(directory):
           os.makedirs(directory)
        
        # Convert the ndarray to a DataFrame
        #Because we wanted to convert to csv, using the panda library is an easy way to do so and we did
        df = pd.DataFrame(finalMeas, columns=['frequency', 'dBm', 'phase'])
        
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        #prints file path confirmation
        print('Measurement successful. CSV file saved at:')
        print(csv_file_path+"\n")
        
        #Prints the elaspsed time for the measurement. Useful when we plan our measurements
        endTime = time.perf_counter()
        elapsed = endTime - startTime
        print(f'Total time so far to run this measurement: {elapsed:.3f} secounds\n')
        endTimeLocal = time.perf_counter()
        elapsedLocal = endTimeLocal - startTimeLocal
        print(f"Time taken to measure {file_name}{powerLevelname}dBm: {elapsedLocal:.3f} secounds\n")
        print(f"average is {average[ave_counter]}")
        print(f"power is {powerLevel}")
        print('-------------------------------------------')
        res_counter += 1

    ave_counter = ave_counter + 1

    if powerLevel == -40: # End of first batch
        att = -57
        Status = att.Send_SCPI(f":SETATT={att}", "") 
        #Responses = att.Send_SCPI(":ATT?", "")
        #print (str(Responses[1]))

        
#making a seperate folder to create plots of all saved data
for entry in os.scandir(fr"{directory}"):  
    if entry.is_file():  # check if it's a file
        df = pd.read_csv(entry)
        plt.plot(df['frequency'], df['dB'], marker='o')
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.title(os.path.splitext(entry.name)[0])
        plt.grid(True)
        print(f"Processing: {entry.name}")
        
        #make directory to save file
        # Create the directory if it doesn't exist
        plots_dir = os.path.join(directory, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Save plot
        filename = os.path.splitext(entry.name)[0] + '.png'
        full_path = os.path.join(plots_dir, filename)
        plt.savefig(full_path)
        plt.close()  # Close the figure to avoid overlap
        
att.Disconnect() # disconnects w/attenuator, not sure if this is important