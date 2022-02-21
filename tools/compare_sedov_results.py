#!/usr/bin/env python3

"""
Run the script with -h to obtain more information regarding the script.
"""

import os
import argparse

import numpy as np
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

#####----- AESTHETICS - These should be used consistently throughout
FONTSIZE   = 16
MARKERSIZE = 5

#####----- PROGRAM USAGE INFO
DESCRIPTION = \
    "Perform a regression test on results obtained with Milhoja's Sedov\n" \
    "test problem.  This test should work for both 2D and 3D test setups\n" \
    "and regardless of whether or not the runtime has been used or which\n" \
    "thread team configurations were used.\n\n" \
    "Note that for graphical visualizations, only a single 2D slice at the\n" \
    "approximate midpoint along the z-direction will be shown.\n"
FNAME_A_HELP = \
    'The name of the folder that contains the results of all reference\n' \
    'Sedov results to use in the comparison\n'
FNAME_B_HELP = \
    'The name of the folder that contains the results of all new\n' \
    'Sedov results to use in the comparison\n'

if __name__ == '__main__':
    #####----- SPECIFY COMMAND LINE USAGE
    parser = argparse.ArgumentParser(description=DESCRIPTION, \
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('A', nargs=1, help=FNAME_A_HELP)
    parser.add_argument('B', nargs=1, help=FNAME_B_HELP)

    #####----- CONFIRM PROPER EXECUTION ENVIRONMENT
    FCOMPARE_EXE = os.getenv('FCOMPARE_EXE')
    if FCOMPARE_EXE == None:
        print()
        print('ERROR: Please set FCOMPARE_EXE')
        print()
        parser.print_help()
        exit(2)

    #####----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()

    result_A = Path(args.A[0]).resolve()
    result_B = Path(args.B[0]).resolve()

    if   not result_A.is_dir():
        print()
        print(f'{result_A} does not exist or is not a folder')
        print()
        parser.print_help()
        exit(3)
    elif not result_B.is_dir():
        print()
        print(f'{result_B} does not exist or is not a folder')
        print()
        parser.print_help()
        exit(4)

    # Select the name to show in visualizations based on regression
    # testing needs.  This should be <commit>/<result folder>/<test name>
    fname_A = os.sep.join(result_A.parts[-3:])
    fname_B = os.sep.join(result_B.parts[-3:])

    fname_data_A   = result_A.joinpath('sedov.dat')
    fname_log_A    = result_A.joinpath('sedov.log')
    fname_plot_A   = result_A.joinpath('sedov_plt_final')
    fname_timing_A = result_A.joinpath('sedov_timings_hydro.dat')

    fname_data_B   = result_B.joinpath('sedov.dat')
    fname_log_B    = result_B.joinpath('sedov.log')
    fname_plot_B   = result_B.joinpath('sedov_plt_final')
    fname_timing_B = result_B.joinpath('sedov_timings_hydro.dat')

    # The timing filename has changed over time.
    # TODO: Get rid of this once the name is used consistently between 
    # current baselines and all tests generating new results to be tested.
    if not fname_timing_A.exists():
        fname_timing_A = Path(str(fname_timing_A).replace('_hydro', ''))
    if not fname_timing_B.exists():
        fname_timing_B = Path(str(fname_timing_B).replace('_hydro', ''))

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

    print()
    print('Sedov A Result Summary')
    print('-' * 80)
    print(r_A)
    print()
    print('Sedov B Result Summary')
    print('-' * 80)
    print(r_B)
    print()

    #####----- VISUALIZE CONSERVED QUANTITIES TIMESERIES
    subp = mfig.SubplotParams(left=0.06, right=0.97, top=0.84, bottom=0.10, \
                              hspace=0.35, wspace=0.25)
    fig = plt.figure(num=1, FigureClass=sedov.MplConservedQuantities, \
                            figsize=(15, 7), subplotpars=subp)
    fig.draw_plot(iq_A_df, N_BINS, fname_A)

    subp = mfig.SubplotParams(left=0.06, right=0.97, top=0.84, bottom=0.10, \
                              hspace=0.35, wspace=0.25)
    fig = plt.figure(num=2, FigureClass=sedov.MplConservedQuantities, \
                            figsize=(15, 7), subplotpars=subp)
    fig.draw_plot(iq_B_df, N_BINS, fname_B)

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
               '--allow_diff_grids', \
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
    z_idx = int(np.floor(0.5 * len(z_coords_A)))

    subp = mfig.SubplotParams(left=0.04, right=0.96, top=0.84, bottom=0.11, \
                              hspace=0.5, wspace=0.97)
    fig = plt.figure(num=3, FigureClass=sedov.MplSolutionComparison, \
                            figsize=(18, 4), subplotpars=subp)
    fig.fontsize = FONTSIZE - 4
    fig.draw_plot(z_coords_A[z_idx], \
                  r_A, fname_A, \
                  r_B, fname_B)

    subp = mfig.SubplotParams(left=0.09, right=0.975, top=0.92, bottom=0.11)
    fig = plt.figure(num=4, FigureClass=sedov.MplWalltimesByStep, \
                            figsize=(12, 6), subplotpars=subp)
    fig.draw_plot(r_A, N_BINS, fname_A, 'ms')

    subp = mfig.SubplotParams(left=0.09, right=0.975, top=0.92, bottom=0.11)
    fig = plt.figure(num=5, FigureClass=sedov.MplWalltimesByStep, \
                            figsize=(12, 6), subplotpars=subp)
    fig.draw_plot(r_B, N_BINS, fname_B, 'ms')

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

