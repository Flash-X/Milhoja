#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import matplotlib.figure as mfig

from pathlib import Path

import sedov

#####----- CONFIGURATION VALUES
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

    fname_data = result_path.joinpath('sedov.dat')
    fname_log  = result_path.joinpath('sedov.log')
    fname_plot = result_path.joinpath('sedov_plt_final')
    if   not fname_data.is_file():
        print_usage(f'{fname_data} does not exist')
        exit(3)
    elif not fname_log.is_file():
        print_usage(f'{fname_log} does not exist')
        exit(4)
    elif not fname_plot.is_dir():
        fname_plt = ''

    result = sedov.Result(fname_plot, fname_data, fname_log)
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
    fig.draw_plot(iq_df, N_BINS, f'{result_path}')

    plt.show()

