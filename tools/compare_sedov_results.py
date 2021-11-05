#!/usr/bin/env python3

import os
import sys

import datetime as dt
import subprocess as sbp
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

from pathlib import Path

import sedov

plt.style.use('ggplot')

#####----- CONFIGURATION VALUES
# Conserved quantities bins
N_BINS = 50

# Index of Z-value at which to get slice of final solution
# This should be close to the center of the explosion
#Z_IDX = 128
Z_IDX = 0

#####----- AESTHETICS - These should be used consistently throughout
FONTSIZE   = 16
MARKERSIZE = 5

def print_usage(msg):
    """
    Print the given message as well as general program and usage information.

    Parameters:
        msg - a message to express to the user the error that occurred.
    """
    print()
    print(f'ERROR - {msg}')
    print()
    print('Program Information')
    print('-' * 80)
    print()
    print('Command Line Usage')
    print('-' * 80)
    print()

if __name__ == '__main__':
    """
    Run the program with no arguments to obtain general program and usage
    information.
    """
    if len(sys.argv[1:]) != 2:
        print_usage('Two and only two command line arguments please')
        exit(1)
    result_A = Path(sys.argv[1])
    result_B = Path(sys.argv[2])

    FCOMPARE_EXE = os.getenv('FCOMPARE_EXE')
    if FCOMPARE_EXE == None:
        print_usage('Please set FCOMPARE_EXE')
        exit(2)

    if   not result_A.is_dir():
        print_usage(f'{result_A} does not exist or is not a folder')
        exit(3)
    elif not result_B.is_dir():
        print_usage(f'{result_B} does not exist or is not a folder')
        exit(4)

    fname_data_A   = result_A.joinpath('sedov.dat')
    fname_log_A    = result_A.joinpath('sedov.log')
    fname_plot_A   = result_A.joinpath('sedov_plt_final')
    fname_timing_A = result_A.joinpath('sedov_timings.dat')

    fname_data_B   = result_B.joinpath('sedov.dat')
    fname_log_B    = result_B.joinpath('sedov.log')
    fname_plot_B   = result_B.joinpath('sedov_plt_final')
    fname_timing_B = result_B.joinpath('sedov_timings.dat')

    print()
    print('Execution started at {} UTC'.format(dt.datetime.utcnow()))
    print()
    print('Sedov package metadata')
    print('-' * 80)
    sedov.print_versions()

    # Comparison fails if individual integral quantity checks fail.
    # The checks are run automatically when acquiring the IQ.
    did_iq_fail = False
    try:
        r_A = sedov.Result(fname_plot_A, fname_data_A, \
                           fname_log_A, fname_timing_A)
        iq_A_df = r_A.integral_quantities
        print()
        print(f'Integral Quantities A Check - SUCCESS')
        print()
        print('Sedov A Result Summary')
        print('-' * 80)
        print(r_A)
        print()
    except ValueError as err:
        print()
        print(f'Integral Quantities A Check - FAILED')
        print(err)
        print()
        did_iq_fail = True
    except:
        print()
        print('An unknown error was trapped on A')
        print()
        did_iq_fail = True

    try:
        r_B = sedov.Result(fname_plot_B, fname_data_B, \
                           fname_log_B, fname_timing_B)
        iq_B_df = r_B.integral_quantities
        print()
        print(f'Integral Quantities B Check - SUCCESS')
        print()
        print('Sedov B Result Summary')
        print('-' * 80)
        print(r_B)
        print()
    except ValueError as err:
        print()
        print(f'Integral Quantities B Check - FAILED')
        print(err)
        print()
        did_iq_fail = True
    except:
        print()
        print('An unknown error was trapped on B')
        print()
        did_iq_fail = True

    if did_iq_fail:
        print()
        print(f'A/B Comparison Failed - CANNOT PROCEED')
        print()
        exit(5)

    #####----- VISUALIZE CONSERVED QUANTITIES TIMESERIES
    subp = mfig.SubplotParams(left=0.06, right=0.97, top=0.84, bottom=0.10, \
                              hspace=0.35, wspace=0.25)
    fig = plt.figure(num=1, FigureClass=sedov.MplConservedQuantities, \
                            figsize=(15, 7), subplotpars=subp)
    fig.draw_plot(iq_A_df, N_BINS, f'{result_A}')

    subp = mfig.SubplotParams(left=0.06, right=0.97, top=0.84, bottom=0.10, \
                              hspace=0.35, wspace=0.25)
    fig = plt.figure(num=2, FigureClass=sedov.MplConservedQuantities, \
                            figsize=(15, 7), subplotpars=subp)
    fig.draw_plot(iq_B_df, N_BINS, f'{result_B}')

    # Non-conservative integral quantities should track each other closely
    # across runs.  Also the conserved quantities should have similar 
    # values.
    try:
        sedov.compare_integral_quantities(r_A, r_B)
        print()
        print(f'A/B Integral Quantities Comparison - SUCCESS')
        print()
    except ValueError as err:
        print()
        print(f'A/B Integral Quantities Comparison - FAILED')
        print(err)
        print()
        did_iq_fail = True

    plt.show()

    did_fcompare_fail = False
    try:
        cmd = [FCOMPARE_EXE, \
               '--norm', str(sedov.FCOMPARE_NORM), \
               '--rel_tol', str(sedov.FCOMPARE_TOLERANCE_RELERR), \
               fname_plot_A, fname_plot_B]
        sbp.run(cmd, check=True)
        print()
        print(f'A/B Plotfile Comparison - SUCCESS')
        print()
    except sbp.CalledProcessError as err:
        print()
        print(f'A/B Plotfile Comparison - FAILED')
        print()
        did_fcompare_fail = True

    #####----- VISUALIZE FINAL SOLUTION
    z_coords_A = r_A.z_coordinates
    z_coords_B = r_B.z_coordinates
    assert(z_coords_A == z_coords_B)

    subp = mfig.SubplotParams(left=0.04, right=0.96, top=0.84, bottom=0.11, \
                              hspace=0.5, wspace=0.97)
    fig = plt.figure(num=3, FigureClass=sedov.MplSolutionComparison, \
                            figsize=(18, 4), subplotpars=subp)
    fig.fontsize = FONTSIZE - 4
    fig.draw_plot(z_coords_A[Z_IDX], \
                  r_A, f'{result_A.name}', \
                  r_B, f'{result_B.name}')

    subp = mfig.SubplotParams(left=0.09, right=0.975, top=0.92, bottom=0.11)
    fig = plt.figure(num=4, FigureClass=sedov.MplWalltimesByStep, \
                            figsize=(12, 6), subplotpars=subp)
    fig.draw_plot(r_A, N_BINS, f'{result_A.name}', 'ms')

    subp = mfig.SubplotParams(left=0.09, right=0.975, top=0.92, bottom=0.11)
    fig = plt.figure(num=5, FigureClass=sedov.MplWalltimesByStep, \
                            figsize=(12, 6), subplotpars=subp)
    fig.draw_plot(r_B, N_BINS, f'{result_B.name}', 'ms')

    plt.show()

    # Let users see fcompare results and difference regardless of success
    if did_iq_fail or did_fcompare_fail:
        print()
        print('Total A/B Comparison - FAILED')
        print()
        exit(6)
    else:
        print()
        print('Total A/B Comparison - SUCCESS')
        print()

