#!/usr/bin/env python3

import sys

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

from pathlib import Path

import sedov

plt.style.use('ggplot')

#####----- CONFIGURATION VALUES
# Conserved quantities bins
N_BINS = 50

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
    if len(sys.argv[1:]) != 1:
        print_usage('One and only one command line argument please')
        exit(1)
    result_path = Path(sys.argv[1])

    if not result_path.is_dir():
        print_usage(f'{result_path} does not exist or is not a folder')
        exit(2)

    fname_data   = result_path.joinpath('sedov.dat')
    fname_log    = result_path.joinpath('sedov.log')
    fname_plot   = result_path.joinpath('sedov_plt_final')
    fname_timing = result_path.joinpath('sedov_timings.dat')

    print()
    print('Execution started at {} UTC'.format(dt.datetime.utcnow()))
    print()
    print('Test Sedov package')
    print('-' * 80)
    sedov.test()

    result = sedov.Result(fname_plot, fname_data, fname_log, fname_timing)
    iq_df = result.integral_quantities

    print()
    print('Sedov Result Summary')
    print('-' * 80)
    print(result)
    print()

    #####----- VISUALIZE CONSERVED QUANTITIES TIMESERIES
    subp = mfig.SubplotParams(left=0.06, right=0.97, top=0.84, bottom=0.10, \
                              hspace=0.35, wspace=0.25)
    fig = plt.figure(num=1, FigureClass=sedov.MplConservedQuantities, \
                            figsize=(15, 7), subplotpars=subp)
    fig.draw_plot(iq_df, N_BINS, f'{result_path.name}')

    #####----- VISUALIZE FINAL SOLUTION
    z_coords = result.z_coordinates
    z_idx = int(np.floor(0.5 * len(z_coords)))

    subp = mfig.SubplotParams(left=0.06, right=0.95, top=0.88, bottom=0.08, \
                              hspace=0.2, wspace=0.6)
    fig = plt.figure(num=2, FigureClass=sedov.MplFinalSolution, \
                            figsize=(15, 6), subplotpars=subp)
    fig.draw_plot(result, z_coords[z_idx], f'{result_path.name}')

    #####----- VISUALIZE PERFORMANCE DATA
    subp = mfig.SubplotParams(left=0.09, right=0.975, top=0.92, bottom=0.11)
    fig = plt.figure(num=3, FigureClass=sedov.MplWalltimesByStep, \
                            figsize=(12, 6), subplotpars=subp)
    fig.draw_plot(result, N_BINS, f'{result_path.name}', 'ms')

    plt.show()

