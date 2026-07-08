# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:44:52 2025

@author: Paul Song and Khubilai Bayarsaikhan

Analysis side of the resonator sweep: fits resonator data (Qi, Qc, fc, etc.)
from saved CSV files, produces plots, and writes a summary CSV.

This file has no hardware/VISA dependencies, so it can be safely imported or
run on its own against a folder of previously-collected CSV data.

Can be run on its own (see the commented block at the bottom of this file)
or imported and called from main.py alongside measurement.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import time
import datetime
import sys
import glob
import scipy.optimize
import re as regex
import uncertainties

# Fixed, one-time setup: folder containing mcl_RUDAT_NET45.dll and the
# fit_resonator package. Shared across all experiments — not something to
# change per experiment (that's what config.py is for). If you ever move
# this folder, also update the matching constant in measurement.py.
DEPENDENCY_DIR = r"C:\Users\cryoq\Documents\ResonatorMeasurementAndAnalysis\Dependencies"


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
    """
    Finds all CSVs in DATA_FOLDER, groups them by resonator number, fits each
    resonator's power sweep, and writes a summary CSV of the fit results.
    """
    HOST_FOLDER = os.path.join(DATA_FOLDER, "..")

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

    # 1. Find every CSV file in the data directory
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    if not all_files:
        print(f"ERROR: No files found in {DATA_FOLDER}")
        return

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
# Uncomment the block below to run analysis.py on its own against an
# existing folder of measurement CSVs (no VNA/attenuator connection needed).
# =============================================================================
# import config
#
# data_directory = os.path.join(config.directory, "Data")
# user_fit(config.directory, DEPENDENCY_DIR, data_directory)
