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


# Importing all the necessary libraries
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import time
import datetime
import sys
import glob
from matplotlib.gridspec import GridSpec
import scipy.optimize
from scipy.interpolate import interp1d
import scipy.special
import re as regex
import uncertainties

startTime = time.perf_counter()
# Change this variable to the address of your instrument
VISA_ADDRESS = 'TCPIP0::Cryostat-Computer::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR'

# Create a connection (session) to the instrument
resourceManager = visa.ResourceManager()
session = resourceManager.open_resource(VISA_ADDRESS)

#session timeout is how long (in ms) the instrument would run one a code before it kicks you out
timeout = 43200000 #timeout after 12 hours
session.timeout = timeout
startTime = time.perf_counter()

#This will be the startup function to setup the VNA, was made seperatly for testing purposes
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


#### THIS IS THE MAIN FUNCTION THAT COLLECTS DATA!
def remote_sweep(res_freq, phase_offset, tau_split_count, assumed_Qtot, bandwidth, edelay, magn_slope, magn_offset, directory, power_sequence, attn, average):
    # This section is to connect to the remote attenuator
    import clr # pythonnet
    clr.AddReference('mcl_RUDAT_NET45')      # Reference the DLL

    from mcl_RUDAT_NET45 import USB_RUDAT
    att = USB_RUDAT()   # Create an instance of the USB control class

    Status = att.Connect()       # Connect

    if Status[0] <= 0:
        print ("Could not connect to attenuator.")
        exit


    
    Status = att.Send_SCPI(f":SETATT={attn}", "") # Set attenuation to first attenuation value

    # Create data directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    general_path = os.path.join(directory, f"{timestamp}")
    data_path = os.path.join(general_path, "Data")
    os.makedirs(data_path, exist_ok=False)

    ave_counter = 0
    for powerLevel in power_sequence:
        powerLevelname = powerLevel - 68 - attn

        res_counter = 1    
        for f in res_freq:
            paraSet(bandwidth, average[ave_counter], powerLevel, edelay, phase_offset[res_counter-1], magn_slope[res_counter-1], magn_offset)
            adjusted_Qtot = assumed_Qtot[res_counter-1] # This is the Qtot we will use to calculate the segment frequencies, can be changed to be different from the assumed Qtot if we want
            startTimeLocal = time.perf_counter()
            file_name = (f"Resonator_{res_counter}_")
            #initialize freq list
            freq_list = np.empty((0,))
            #make a list of omega
            omega = np.linspace(-(24*np.pi)/25, (24*np.pi)/25, tau_split_count)
            #create a new frequency for each omega and add to list
            for t in omega:
                freq = f*(1+(1/(2*adjusted_Qtot))*np.tan(t/2))
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
            
            # Create the csv file path
            csv_file_path = os.path.join(data_path, file_name+str(powerLevelname)+"dBm_"+{attn}+"VARAT.csv")
            
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
            print(f"attenuation is -{attn}")
            print('-------------------------------------------')
            res_counter += 1

        ave_counter = ave_counter + 1

        if powerLevel == -40: # End of first batch
            attn = 57
            Status = att.Send_SCPI(f":SETATT={attn}", "") 
            #Responses = att.Send_SCPI(":ATT?", "")
            #print (str(Responses[1]))
            
    att.Disconnect() # disconnects w/attenuator, not sure if this is important
    return data_path

# Plotting Functions

def set_xaxis_rot(ax, angle=45.):
    """Rotate x-axis labels."""
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)


def trim_s21_wings(fname_in: str, Ntrim: list, minpts: int = 100,
                   use_asymm: bool = False):
    """
    Trim the front and back of the data from fname, write to the same location
    with the appended _trimmed path returned and the data.
    """
    data_in = np.genfromtxt(fname_in, delimiter=',')

    if sum(Ntrim) > data_in.shape[0] - minpts:
        raise ValueError(f'Ntrim ({Ntrim}) too long for min pts {minpts}.')

    print(f'data_in.shape: {data_in.shape}')
    print(f'Ntrim: {Ntrim}')
    data_out = data_in[Ntrim[0]:-(Ntrim[1]+1), :]
    print(f'data_out.shape: {data_out.shape}')

    dot_split = fname_in.split('.')
    fext = dot_split[-1]
    if len(dot_split) > 1:
        fname_out = '.'.join(dot_split[0:-1]) + f'_trimmed.{fext}'
    else:
        fname_out = dot_split[0] + f'_trimmed.{fext}'
    with open(fname_out, 'w') as fid:
        fid.write('\n'.join(['%.8g, %.8g, %.8g' % (f, sdb, sph)
        for f, sdb, sph in zip(data_out[:, 0], data_out[:, 1], data_out[:, 2])]))

    return fname_out, data_out


def fit_single_res(filename, res, fsd, PLOTS_FOLDER, PLOTDATA_FOLDER, filter_points=[0, 0], preprocess_method='linear',
                   use_gauss_filt=False, use_matched_filt=False,
                   use_elliptic_filt=False, use_mov_avg_filt=False,
                   fname_ref=None):
    """Fit a single resonator from file."""
    if os.path.isabs(filename):
        fname = filename
    else:
        my_dir = os.getcwd()
        fname = os.path.join(my_dir, filename)

    if sum(filter_points) > 0:
        print(f'Trimming data to {filter_points} ...')
        fname, sdata = trim_s21_wings(fname, filter_points)

    filename = fname

    print('-------------')
    print(filename)

    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = []
    manual_init = None
    normalize = 10

    myres = res.Resonator()
    myres.from_file(filename)
    myres.preprocess_method = preprocess_method
    myres.normalize = normalize

    try:
        myres.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds,
            MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)
    except Exception as ex:
        print(f'Exception in fit_single_res for {filename}:\n{ex}')
        raise

    myres.filepath = os.path.join(PLOTS_FOLDER, 'Resonators') + os.sep
    os.makedirs(myres.filepath, exist_ok=True)

    params, conf_intervals, fig1, chi1, init1 = fsd.fit(myres)

    return params, chi1, conf_intervals


def fit_qiqcfc_vs_power(filenames, res, fsd, PLOTS_FOLDER, PLOTDATA_FOLDER, powers, filter_points=None,
                        preprocess_method='linear', phi0=0.,
                        use_gauss_filt=False, use_matched_filt=False,
                        use_elliptic_filt=False, filt_idxs=None,
                        use_mov_avg_filt=False, fname_ref=None):
    """Fits multiple resonances at different powers for a given power."""
    Npts = len(filenames)
    Qc = np.zeros(Npts); fc = np.zeros(Npts)
    Qi = np.zeros(Npts); Q = np.zeros(Npts)
    Qc_err = np.zeros(Npts); fc_err = np.zeros(Npts); Qi_err = np.zeros(Npts)
    navg = np.zeros(Npts); errs = np.zeros(Npts)
    for idx, filename in enumerate(filenames):
        filter_pts = filter_points[idx] if filter_points is not None else [0]*2
        use_matched_filt_chk = use_matched_filt and (idx in filt_idxs)
        use_elliptic_filt_chk = use_elliptic_filt and (idx in filt_idxs)
        use_mov_avg_filt_chk = use_mov_avg_filt and (idx in filt_idxs)
        params, err, conf_int = fit_single_res(filename, res, fsd, PLOTS_FOLDER, PLOTDATA_FOLDER,
                                filter_points=filter_pts,
                                preprocess_method=preprocess_method,
                                use_gauss_filt=use_gauss_filt,
                                use_matched_filt=use_matched_filt_chk,
                                use_elliptic_filt=use_elliptic_filt_chk,
                                use_mov_avg_filt=use_mov_avg_filt_chk,
                                fname_ref=fname_ref)

        Qcj = params[1] * np.exp(1j*(params[3] + phi0))
        Qij = 1. / (1. / params[0] - np.real(1. / Qcj))

        Q[idx] = params[0]
        Qc[idx] = np.real(Qcj)
        Qi[idx] = Qij
        fc[idx] = params[2]
        errs[idx] = err
        navg[idx] = power_to_navg(powers[idx], Qi[idx], Qc[0], fc[0])

        Qi_err[idx] = conf_int[1]
        Qc_err[idx] = conf_int[2]
        fc_err[idx] = conf_int[5]

        print(f'navg: {navg[idx]} photons')
        print(f'Q: {Q[idx]} +/- {conf_int[0]}')
        print(f'Qi: {Qi[idx]} +/- {Qi_err[idx]}')
        print(f'Qc: {Qc[idx]} +/- {Qc_err[idx]}')
        print(f'fc: {fc[idx]} +/- {fc_err[idx]} GHz')
        print('-------------\n')
        plt.close('all')

    df = pd.DataFrame(np.vstack((powers, navg, fc, Qi, Qc, Q,
                    errs, Qi_err, Qc_err, fc_err)).T,
            columns=['Power [dBm]', 'navg', 'fc [GHz]', 'Qi', 'Qc', 'Q',
                'error', 'Qi error', 'Qc error', 'fc error'])

    dstr = datetime.datetime.today().strftime('%y%m%d_%H_%M_%S')
    df.to_csv(os.path.join(PLOTDATA_FOLDER, f'qiqcfc_vs_power_{dstr}.csv'))

    return df


def fit_delta_tls(Qi, T, fc, Qc, p, display_scales={'QHP': 1e5,
                'nc': 1e7, 'Fdtls': 1e-6}):
    """
    Fit the loss using the expression
    delta_tls = F * delta0_tls * tanh(hbar w_c / 2 kB T) (1 + <n> / nc)^-1/2
    """
    h = 6.626069934e-34
    hbar = 1.0545718e-34
    kB = 1.3806485e-23
    fc_GHz = fc if np.any(fc >= 1e9) else fc * 1e9
    TK = T if T <= 400e-3 else T * 1e-3
    delta = 1. / Qi
    hw0 = hbar * 2 * np.pi * fc_GHz
    kT = kB * TK

    navg = power_to_navg(p, Qi, Qc, fc)
    labels = [r'$10^{%.2g}$' % x for x in np.log10(navg)]
    print(f'<n>: {labels}')
    print(f'T: {TK} K')
    print(f'fc_GHz: {fc_GHz} Hz')

    def fitfun4(n, Fdtls, nc, QHP, beta):
        num = Fdtls * np.tanh(hw0 / (2 * kT))
        den = (1. + n / nc)**beta
        return num / den + 1./QHP

    x0 = [1e-6, 1e2, np.max(Qi), 0.1]
    popt, pcov = scipy.optimize.curve_fit(fitfun4, navg, delta, p0=x0,
                                          maxfev=10000)

    Fdtls, nc, QHP, beta = popt
    errs = np.sqrt(np.diag(pcov))
    Fdtls_err, nc_err, QHP_err, beta_err = errs

    def round_sigfig(x, n):
        if not np.isfinite(x):
            return x
        if x == 0:
            return 0
        magnitude = int(np.floor(np.log10(abs(x))))
        return round(x, n - magnitude - 1)

    Fdtls_err = round_sigfig(Fdtls_err, 1)
    nc_err    = round_sigfig(nc_err, 1)
    QHP_err   = round_sigfig(QHP_err, 1)
    beta_err  = round_sigfig(beta_err, 1)

    Fdtls_un = uncertainties.ufloat(Fdtls, Fdtls_err)
    nc_un    = uncertainties.ufloat(nc, nc_err)
    QHP_un   = uncertainties.ufloat(QHP, QHP_err)
    beta_un  = uncertainties.ufloat(beta, beta_err)

    print(f'QHP: {QHP:.2f}+/-{QHP_err:.2f}')

    Fdtls_latex = f'{Fdtls_un:L}'
    nc_latex = f'{nc_un:L}'
    QHP_latex = f'{QHP_un:L}'
    beta_latex = f'{beta_un:L}'

    Fdtls_str = r'$F\delta^{0}_{TLS}: %s$' % Fdtls_latex
    nc_str    = r'$n_c: %s$' % nc_latex
    QHP_str   = r'$Q_{HP}: %s$' % QHP_latex
    beta_str  = r'$\beta: %s$' % beta_latex
    delta_fit_str = Fdtls_str + '\n' + nc_str \
            + '\n' + QHP_str + '\n' + beta_str
    print(delta_fit_str)

    return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
            fitfun4(navg, *popt), delta_fit_str, beta, beta_err


def get_powers_from_file(time_fname, temp_fname):
    """Returns the powers after reading a file."""
    powers = np.genfromtxt(time_fname, delimiter=',').T
    times, temperatures = np.genfromtxt(temp_fname, delimiter=',', dtype=str).T
    temperatures = np.asarray(temperatures)
    times = np.asarray([str(t) for t in times])
    return times, temperatures, powers


def power_to_navg(power_dBm, Qi, Qc, fc, Z0_o_Zr=1.):
    """
    Converts power to photon number following Eq. (1) of arXiv:1801.10204
    and Eq. (3) of arXiv:1912.09119
    """
    h = 6.62607015e-34
    hbar = 1.0545718e-34

    Papp = 10**(power_dBm / 10) * 1e-3
    fc_GHz = fc * 1e9
    hb_wc2 = hbar * (2 * np.pi * fc_GHz)**2

    Q = 1. / ((1. / Qi) + (1. / Qc))
    navg = (2. / hb_wc2) * (Q**2 / Qc) * Papp

    return navg


def power_sweep_fit_drv(res, fsd, PLOTS_FOLDER, PLOTDATA_FOLDER, atten=[0, -60], powers_in=None,
                        filenames_in=None,
                        temperature=0.012,
                        plot_from_file=False, use_error_bars=True,
                        temp_correction='', phi0=0., use_gauss_filt=True,
                        use_matched_filt=False, use_elliptic_filt=False,
                        use_mov_avg_filt=False, loss_scale=None,
                        preprocess_method='linear',
                        ds={'QHP': 1e4, 'nc': 1e6, 'Fdtls': 1e-6},
                        plot_twinx=True, plot_fit=False, number=0):
    """Driver for fitting the power sweep data for a given set of data."""
    if np.any(powers_in):
        powers = np.copy(powers_in)
    else:
        powers = np.linspace(15, -105, 25)

    print(f'powers: {powers}')

    tstr = '12_mK'
    if filenames_in:
        filenames = filenames_in
    else:
        filenames = [f'NIST_NBSI_CTRL01_220926_5_5p066GHz_{int(p)}dB_{tstr}.csv'
                for p in powers]
    filt_idxs = []
    fname_ref = filenames[0]
    filter_points = [[0, 0] for _ in filenames]
    print(f'filter_points:\n{filter_points}')
    dstr = datetime.datetime.today().strftime('%y%m%d')
    err_str = '_error_bars' if use_error_bars else ''
    cal_str = temp_correction + '_'
    fsize = 20
    csize = 5

    if plot_from_file:
        df = pd.read_csv('qiqcfc_vs_power_210811_17_23_44.csv')
    else:
        df = fit_qiqcfc_vs_power(filenames, res, fsd, PLOTS_FOLDER, PLOTDATA_FOLDER, powers,
                filter_points=filter_points,
                preprocess_method=preprocess_method,
                phi0=phi0, use_gauss_filt=use_gauss_filt,
                use_matched_filt=use_matched_filt,
                use_elliptic_filt=use_elliptic_filt,
                use_mov_avg_filt=use_mov_avg_filt,
                filt_idxs=filt_idxs,
                fname_ref=fname_ref)

    Qi = df['Qi']
    Qc = df['Qc']
    Q  = df['Q']
    navg = df['navg']
    delta = 1. / Qi
    fc = df['fc [GHz]']
    Qi_err = df['Qi error']
    Qc_err = df['Qc error']
    delta_err = Qi_err / Qi**2
    fc_err = df['fc error']

    powers += sum(atten)

    def pdBm_to_navg_ticks(p):
        n = power_to_navg(powers[0::2], Qi[0::2], Qc[0], fc[0])
        labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
        print(f'labels:\n{labels}')
        return labels

    T = temperature
    doff = 0
    if plot_fit:
        if doff > 0:
            Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str, beta, beta_err \
                    = fit_delta_tls(Qi[0:-doff], T, fc[0], Qc[0], powers[0:-doff],
                    display_scales=ds)
        else:
            Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str, beta, beta_err \
                    = fit_delta_tls(Qi, T, fc[0], Qc[0], powers,
                    display_scales=ds)

        if loss_scale:
            delta_fit /= loss_scale

        print('\n')
        pillow = nc
        print(f'F * d0_tls: {Fdtls:.2g} +/- {Fdtls_err:.2g}')
        print(f'nc: {nc:.2g} +/- {nc_err:.2g}')
        print('\n')

    if loss_scale:
        delta /= loss_scale
        delta_err /= loss_scale

    fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)
    ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_fc.set_ylabel('Resonance Frequency [GHz]', fontsize=fsize)
    ax_fc_top = ax_fc.twiny()

    fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True)
    fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True)
    fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True)
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)

    if not plot_twinx:
        powers = power_to_navg(powers, Qi, Qc[0], fc[0])

    if use_error_bars:
        markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
        colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax_fc.errorbar(powers, fc, yerr=fc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qc.errorbar(powers, Qc, yerr=Qc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qiqc.errorbar(powers, Qi, yerr=Qi_err, marker='h', ls='', ms=10,
                capsize=csize, color=colors[5],
                label=r'$Q_i$')
        ax_qiqc.errorbar(powers, Qc, yerr=Qc_err, marker='^', ls='', ms=10,
                capsize=csize, color=colors[6],
                label=r'$Q_c$')
        ax_qi.errorbar(powers, Qi, yerr=Qi_err, marker='o', ls='', ms=10,
                capsize=csize)
        if doff > 0:
            ax_d.errorbar(powers[0:-doff], delta[0:-doff],
                    yerr=delta_err[0:-doff], marker='d', ls='',
                    color=colors[1], ms=10, capsize=csize)
            if plot_fit:
                ax_d.plot(powers[0:-doff], delta_fit, ls='-',
                        label=delta_fit_str, color=colors[1])
        else:
            ax_d.errorbar(powers, delta,
                    yerr=delta_err, marker='d', ls='', color=colors[1],
                    ms=10, capsize=csize)
            if plot_fit:
                ax_d.plot(powers, delta_fit, ls='-', label=delta_fit_str,
                    color=colors[1])

    else:
        ax_fc.plot(powers, fc, marker='o', ms=10, ls='')
        ax_qc.plot(powers, Qc, marker='o', ms=10, ls='')
        ax_qi.plot(powers, Qi, marker='o', ms=10, ls='')
        ax_d.plot(powers, delta, marker='o', ms=10, ls='')

    ax_qc.set_ylabel(r'$Q_c$', fontsize=fsize)
    ax_qi.set_ylabel(r'$Q_i$', fontsize=fsize)
    ax_qiqc.set_ylabel(r'$Q_i, Q_c$', fontsize=fsize)

    if loss_scale:
        ax_d.set_ylabel(r'$Q_i^{-1}\times 10^{%d}$'
                         % int(np.log10(loss_scale)), fontsize=fsize)
    else:
        ax_d.set_ylabel(r'$Q_i^{-1}$', fontsize=fsize)

    power_str = f'{atten[0]} dB ext, {atten[1]} dB int attenuation'
    ax_qc.set_title(power_str, fontsize=fsize)
    ax_fc.set_title(power_str, fontsize=fsize)
    ax_qi.set_title(power_str, fontsize=fsize)

    if plot_twinx:
        ax_d_top  = ax_d.twiny()
        ax_qc_top = ax_qc.twiny()
        ax_qi_top = ax_qi.twiny()
        ax_qiqc_top = ax_qiqc.twiny()

        ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_d.set_xlabel('Power [dBm]', fontsize=fsize)

        ax_qc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qi_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qiqc_top.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
        ax_d_top.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)
        ax_fc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)

        ax_qc.set_xticks(powers[0::2])
        ax_qi.set_xticks(powers[0::2])
        ax_qiqc.set_xticks(powers[0::2])
        ax_d.set_xticks(powers[0::2])
        ax_fc.set_xticks(powers[0::2])

        ax_qc_top.set_xticks(ax_qc.get_xticks())
        ax_qi_top.set_xticks(ax_qi.get_xticks())
        ax_qiqc_top.set_xticks(ax_qi.get_xticks())
        ax_d_top.set_xticks(ax_d.get_xticks())
        ax_fc_top.set_xticks(ax_fc.get_xticks())

        ax_qc_top.set_xbound(ax_qc.get_xbound())
        ax_qi_top.set_xbound(ax_qi.get_xbound())
        ax_qiqc_top.set_xbound(ax_qi.get_xbound())
        ax_d_top.set_xbound(ax_d.get_xbound())
        ax_fc_top.set_xbound(ax_fc.get_xbound())

        ax_qc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qc.get_xticks()))
        ax_qi_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
        ax_qiqc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
        ax_d_top.set_xticklabels(pdBm_to_navg_ticks(ax_d.get_xticks()))
        ax_fc_top.set_xticklabels(pdBm_to_navg_ticks(ax_fc.get_xticks()))

        set_xaxis_rot(ax_d_top, 45)
        set_xaxis_rot(ax_d, 45)

    else:
        ax_qc.set_xscale('log')
        ax_qi.set_xscale('log')
        ax_qiqc.set_xscale('log')
        ax_fc.set_xscale('log')
        ax_d.set_xscale('log')

        ax_qc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_qi.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_qiqc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_fc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_d.get_xaxis().get_major_formatter().labelOnlyBase = False

        ax_qc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qi.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qiqc.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
        ax_fc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_d.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)

    qiqc_lbls, qiqc_hdls = ax_qiqc.get_legend_handles_labels()
    ax_qiqc.legend(qiqc_lbls, qiqc_hdls, loc=(0.4, 0.4), fontsize=fsize)
    d_lbls, d_hdls = ax_d.get_legend_handles_labels()
    ax_d.legend(d_lbls, d_hdls, loc='upper right', fontsize=fsize)

    # Save the loss tangent plot (other plots commented out as in original)
    tand_path = os.path.join(
        PLOTS_FOLDER,
        f'tand_vs_power{cal_str}{dstr}_{temperature}{err_str}_Resonator{number}.pdf'
    )
    fig_d.savefig(tand_path, format='pdf')

    plt.close('all')
    return QHP, QHP_err, pillow, nc_err, Fdtls, Fdtls_err, np.mean(Qc), beta, beta_err

def user_fit(directory, DEPENDENCIES_FOLDER, DATA_FOLDER):
    HOST_FOLDER = os.path.join(DATA_FOLDER, "..")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Subfolder names (dont really need to change)
    PLOTS_SUBDIR   = 'Plots'
    PLOTDATA_SUBDIR = 'Plot Data'
    RESULTS_SUBDIR = 'Results'

    RESULTS_FOLDER = os.path.join(HOST_FOLDER, RESULTS_SUBDIR)
    PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, PLOTS_SUBDIR)
    PLOTDATA_FOLDER = os.path.join(RESULTS_FOLDER, PLOTDATA_SUBDIR)


    # Make sure the resonator library is importable
    assert os.path.isdir(DATA_FOLDER), \
        f'DATA_FOLDER does not exist: {DATA_FOLDER}'
    assert os.path.isdir(DEPENDENCIES_FOLDER), \
        f'DEPENDENCIES_FOLDER does not exist: {DEPENDENCIES_FOLDER}'
    sys.path.append(DEPENDENCIES_FOLDER)

    # Make sure the output folders exist
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(PLOTDATA_FOLDER, exist_ok=True)

    import fit_resonator.resonator as res
    import fit_resonator.Sdata as fsd
    np.set_printoptions(precision=4, suppress=True)

    if __name__ == '__main__':
        # 1. Find every CSV file in the data directory
        all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

        if not all_files:
            print(f"ERROR: No files found in {DATA_FOLDER}")
        else:
            # 2. Group files by Resonator Number
            resonator_data = {}

            for f in all_files:
                match = regex.search(r'Resonator_(\d+)_(-?\d+\.?\d*)dB', f)

                if match:
                    res_num = int(match.group(1))
                    power_val = float(match.group(2))

                    if res_num not in resonator_data:
                        resonator_data[res_num] = []

                    resonator_data[res_num].append((power_val, f))
                else:
                    print(f"Skipping file (regex mismatch): {os.path.basename(f)}")

            with open(os.path.join(PLOTDATA_FOLDER, 'plotData.csv'),
                    mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Resonator', 'QHP', 'QHP error', 'nc', 'nc error',
                                'Fdtls', 'Fdtls error', 'mean Qc', 'beta',
                                'beta error'])

                for res_numb in range(1, 10):
                    if res_numb not in resonator_data:
                        continue

                    # Sort by power (high to low)
                    data_list = sorted(resonator_data[res_numb],
                                    key=lambda x: x[0], reverse=True)

                    final_powers = [x[0] for x in data_list]
                    final_files = [x[1] for x in data_list]

                    print(f"--- Processing Resonator {res_numb} ---")
                    print(f"Found {len(final_files)} files. "
                        f"Power range: {max(final_powers)} dBm "
                        f"to {min(final_powers)} dBm")

                    try:
                        print(f"--- Attempting Resonator {res_numb} ---")
                        QHP, QHP_err, nc, nc_err, Fdtls, Fdtls_err, mean_Qc, beta, beta_err = power_sweep_fit_drv(
                            res,
                            fsd,
                            PLOTS_FOLDER,
                            PLOTDATA_FOLDER,
                            atten=[0, 0],
                            temperature=12e-3,
                            powers_in=np.array(final_powers),
                            filenames_in=final_files,
                            plot_from_file=False,
                            use_error_bars=True,
                            loss_scale=1e-6,
                            preprocess_method='circle',
                            ds={'QHP': 1e6, 'nc': 1e1, 'Fdtls': 1e-6},
                            plot_twinx=False,
                            plot_fit=True,
                            number=res_numb,
                        )
                        writer.writerow([res_numb, QHP, QHP_err, nc, nc_err,
                                        Fdtls, Fdtls_err, mean_Qc, beta, beta_err])

                    except (Exception, SystemExit) as e:
                        print(f"\n[!] Resonator {res_numb} failed. Moving to next.")
                        print(f"Error details: {e}\n")
                        continue


# =============================================================================
# CONFIG: CHANGE THESE INPUT PARAMETERS
# =============================================================================
res_freq = [3.805798615E9, 4.142051723E9, 4.451010129E9, 4.7334244E9, 5.09801776E9, 5.45630723E9, 5.80264483E9, 6.12921038E9] #these are centers of resonances
phase_offset = [360-70, 360-125, 360-75, 360-10, 360-180, 360-175, 360-100, 360-100]

tau_split_count = 51 #essentially the num of segments
assumed_Qtot = [200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000] #together with tau split cout detmermines our calculations for the spread of our segmented sweep

bandwidth = 0.01 #in khz

edelay = 56
magn_slope = [-11.5, -10.9 ,-10, -9.1, -8.2, -7.4, -6.8, -6.2]
magn_offset = 64

average = [10,10,10,10,10,10,10,10,10,10,10,5**2,10**2,15**2,20**2,25**2,30**2] # Should be 17 power levels, so 17 averaging levels

directory = (r"C:\Users\cryoq\Documents\Resonator_Measurements\FINALTEST")
dependency_directory = os.path.join(directory, "Dependencies")


# Do not change values below unless you understand how power level changes
attn = 17
power_sequence = [5, 0, -5, -10, -15, -20, -25, -30, -35, -40, # First batch at attn -17
                  -5, -10, -15, -20, -25, -30, -35] # Second batch at attn -57

# =============================================================================

if len(res_freq) == len(phase_offset) == len(assumed_Qtot) == len(magn_slope) and len(average) == len(power_sequence):
     print("Input parameter lengths are consistent.")
else:
    print("Error: Inconsistent input parameter lengths.")
    exit()

sys.path.append(dependency_directory) # Make sure all dependencies are in 'Dependencies' folder!
startup()
data_directory = remote_sweep(res_freq, phase_offset, tau_split_count, assumed_Qtot, bandwidth, edelay, magn_slope, magn_offset, directory, power_sequence, attn, average)
user_fit(directory, dependency_directory, data_directory)