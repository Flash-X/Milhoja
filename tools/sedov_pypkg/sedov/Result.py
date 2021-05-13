import yt

import numpy as np
import pandas as pd
import itertools as it

from pathlib import Path

import sedov

class Result(object):
    """
    A class for accessing the results of a single Sedov test run and that
    automatically applies some sanity checks to the results.
    """
    def __init__(self, filename):
        """
        Create an instance of the class for accessing the results associated
        with the given individual result files.  It is generally intended that
        all the given files were been created as part of the same execution of
        the Sedov test problem.

        Parameters:
            fname_plot   - the full path to the AMReX-format folder containing
                           the solution at the end of the simulation
            fname_data   - the filename including full path of the ASCII-format
                           file that contains the integral quantities as
                           computed at each timestep.
            fname_log    - the filename including full path of the ASCII-format
                           file that contains the simulation log information.
            fname_timing - the filename including full path of the ASCII-format
                           file that contains the computation timing data.
        Returns
            The object
        """
        self.__path         = filename
        self.__fname_plot   = filename.joinpath('sedov_plt_final')
        self.__fname_data   = filename.joinpath('sedov.dat')
        self.__fname_log    = filename.joinpath('sedov.log')
        self.__fname_hydro  = filename.joinpath('sedov_timings.dat')

        if   not self.__fname_data.is_file():
            raise ValueError(f'{self.__fname_data} does not exist')
        elif not self.__fname_log.is_file():
            raise ValueError(f'{self.__fname_log} does not exist')
        elif not self.__fname_hydro.is_file():
            raise ValueError(f'{self.__fname_hydro} does not exist')

        self.__dataset = None
        if not self.__fname_plot.is_dir():
            self.__fname_plot = ''
        else:
            self.__dataset = yt.load(str(self.__fname_plot))

        assert(self.__dataset.coordinates.axis_id['x'] == sedov.YT_XAXIS)
        assert(self.__dataset.coordinates.axis_id['y'] == sedov.YT_YAXIS)
        assert(self.__dataset.coordinates.axis_id['z'] == sedov.YT_ZAXIS)

        with open(self.__fname_hydro, 'r') as fptr:
            comments = (each.lstrip().lstrip('#') for each in fptr.readlines() \
                                                  if each.lstrip()[0] == '#')
            comments = (each.split() for each in comments \
                                     if len(each.split()) == 3)
            pairs = ((each[0], each[2]) for each in comments \
                                        if each[1] == '=')
            self.__hdr = dict(pairs)

        # Create structure for caching 2D z-slices
        self.__frbs = {}

        # Combine all packet timing results into a single DataFrame
        self.__packet_timings = sedov.PacketTimings(self.__path)

        super().__init__()

    @property
    def study_name(self):
        return self.__hdr['Testname']

    @property
    def dimension(self):
        return int(self.__hdr['Dimension'])

    @property
    def blocks_shape(self):
        return (int(self.__hdr['NXB']), \
                int(self.__hdr['NYB']), \
                int(self.__hdr['NZB']))

    @property
    def domain_shape(self):
        return (int(self.__hdr['N_BLOCKS_X']), \
                int(self.__hdr['N_BLOCKS_Y']), \
                int(self.__hdr['N_BLOCKS_Z'])) 

    @property
    def n_mpi_processes(self):
        raw_data = np.loadtxt(self.__fname_hydro, delimiter=',')
        assert(raw_data.shape[1] % 2 == 1)
        return int(0.5 * (raw_data.shape[1] - 1))

    @property
    def n_steps(self):
        raw_data = np.loadtxt(self.__fname_hydro, delimiter=',')
        n_rows = raw_data.shape[0]
        assert(all(raw_data[:, 0] == range(1, n_rows+1)))
        return raw_data.shape[0]

    @property
    def n_distributor_threads(self):
        return int(self.__hdr['n_distributor_threads'])

    @property
    def n_cpu_threads(self):
        return int(self.__hdr['n_cpu_threads'])

    @property
    def n_gpu_threads(self):
        return int(self.__hdr['n_gpu_threads'])

    @property
    def n_blocks_per_packet(self):
        return int(self.__hdr['n_blocks_per_packet'])

    @property
    def n_blocks_per_cpu_turn(self):
        return int(self.__hdr['n_blocks_per_cpu_turn'])

    @property
    def timer_resolution_sec(self):
        return float(self.__hdr['MPI_Wtick_sec'])

    @property
    def raw_timings(self):
        n_procs = self.n_mpi_processes

        raw_data = np.loadtxt(self.__fname_hydro, delimiter=',')

        steps = raw_data[:, 0].astype(int)
        procs = range(1, n_procs+1)
        idx = pd.MultiIndex.from_tuples(it.product(steps, procs), \
                                        names=['step', 'proc'])

        columns = ['n_blocks', 'wtime_sec']
        data = np.full([len(idx), len(columns)], 0.0, float)
        for row, (s_idx, p_idx) in enumerate(idx):
            data[row, 0] = raw_data[s_idx-1, 2*(p_idx - 1) + 1]
            data[row, 1] = raw_data[s_idx-1, 2* p_idx]

        df = pd.DataFrame(data=data, index=idx, columns=columns)
        df['n_blocks'] = df.n_blocks.astype(int)
        df['wtime_per_blk_sec'] = df.wtime_sec / df.n_blocks

        return df

    @property
    def timings_per_step(self):
        """
        For each timestep, obtain the maximum walltime per timestep across all
        MPI processes.
        """
        raw_df = self.raw_timings
        steps = raw_df.index.get_level_values('step').unique().values
        max_wtime_sec = np.zeros(len(steps))
        for j, n in enumerate(steps):
            max_wtime_sec[j] = raw_df.xs(n, level='step').wtime_sec.max()

        columns = ['max_wtime_sec']
        return pd.DataFrame(data=max_wtime_sec, index=steps, columns=columns)

    @property
    def packet_timings(self):
        """
        """
        return self.__packet_timings.timings

    @property
    def z_coordinates(self):
        """
        """
        data = self.__dataset.all_data()
        return sorted(np.unique(data['z']))

    def z_data_slice(self, z, var):
        """
        Obtain a 2D slice of the data for the given variable as a 2D array
        that can be visualized with Matplotlib.

        https://yt-project.org/doc/analyzing/generating_processed_data.html
        """
        z_coords = self.z_coordinates
        if z not in z_coords:
            raise ValueError(f'{z} not a valid z-coordinate')
        z_flt = float(z)

        xmin, ymin, zmin = [float(e) for e in self.__dataset.domain_left_edge]
        xmax, ymax, zmax = [float(e) for e in self.__dataset.domain_right_edge]
        extent = (xmin, xmax, ymin, ymax)

        if z_flt not in self.__frbs:
            sl = self.__dataset.slice(sedov.YT_ZAXIS, z)
            n_cells_x, n_cells_y, n_cells_z = self.__dataset.domain_dimensions
            self.__frbs[z_flt] = yt.FixedResolutionBuffer(sl, extent, (n_cells_x, n_cells_y))

        return extent, self.__frbs[z_flt][var]

    @property
    def integral_quantities(self):
        """
        Obtain the raw integral quantities timeseries data.

        Returns
            The timeseries as a DataFrame
        """
        with open(self.__fname_data, 'r') as fptr:
            comments = [line.lstrip().lstrip('#') for line in fptr.readlines() \
                                                  if line.lstrip()[0] == '#']
        assert(len(comments) == 1)
        columns = [each.replace('-', '_') for each in comments[0].split()]

        with open(self.__fname_data, 'r') as fptr:
            data = np.loadtxt(fptr)

        # IC + all steps
        assert(data.shape[0] == self.n_steps + 1)

        # Conserved quantities
        # TODO: These should all be put under test
        checks = [('x', sedov.IQ_XMOM_IDX), \
                  ('y', sedov.IQ_YMOM_IDX), \
                  ('z', sedov.IQ_ZMOM_IDX)]
        for axis, idx in checks:
            max_mom_abserr = np.max(np.abs(data[:, idx]))
            if data[0, idx] != 0.0:
                raise ValueError(f'Initial {axis}-momentum is non-zero')
            elif max_mom_abserr > sedov.IQ_THRESHOLD_VEL:
                msg = '{}-momentum conservation absolute error {} exceeds threshold {}'
                raise ValueError(msg.format(axis, max_mom_abserr, sedov.IQ_THRESHOLD_VEL))

        mass_err = data[1:, sedov.IQ_MASS_IDX] - data[0, sedov.IQ_MASS_IDX]
        max_mass_abserr = np.max(np.abs(mass_err))
        if max_mass_abserr > sedov.IQ_THRESHOLD_MASS:
            msg = 'Mass conservation absolute error {} exceeds threshold {}'
            raise ValueError(msg.format(max_mass_abserr, sedov.IQ_THRESHOLD_MASS))

        etot_err = data[1:, sedov.IQ_ETOT_IDX] - data[0, sedov.IQ_ETOT_IDX]
        max_etot_abserr = np.max(np.abs(etot_err))
        if max_etot_abserr > sedov.IQ_THRESHOLD_ETOT:
            msg = 'Total energy conservation absolute error {} exceeds threshold {}'
            raise ValueError(msg.format(max_etot_abserr, sedov.IQ_THRESHOLD_ETOT))

        df = pd.DataFrame(data=data, columns=columns)
        df.index = df.index.rename('step')

        return df

    @property
    def integral_quantities_statistics(self):
        """
        Obtain basic summary statistics computed for the mass and total energy
        errors.

        Returns
            The statistics as a DataFrame
        """
        index = ['Initial_Mass',
                 'Mean_Mass_Err',
                 'Std_Mass_Err',
                 'Max_Mass_Err',
                 'Initial_Energy',
                 'Mean_Energy_Err',
                 'Std_Energy_Err',
                 'Max_Energy_Err']

        intQuantities = np.full(len(index), 0.0, float)

        with open(self.__fname_data, 'r') as fptr:
            data = np.loadtxt(fptr)

        mass_err = data[1:, sedov.IQ_MASS_IDX] - data[0, sedov.IQ_MASS_IDX]
        etot_err = data[1:, sedov.IQ_ETOT_IDX] - data[0, sedov.IQ_ETOT_IDX]

        intQuantities[0] = data[0, sedov.IQ_MASS_IDX]
        intQuantities[1] = np.mean(mass_err)
        intQuantities[2] = np.std(mass_err)
        intQuantities[3] = np.max(np.abs(mass_err))

        intQuantities[4] = data[0, sedov.IQ_ETOT_IDX]
        intQuantities[5] = np.mean(etot_err)
        intQuantities[6] = np.std(etot_err)
        intQuantities[7] = np.max(np.abs(etot_err))

        columns = ['summary_stats']
        return pd.DataFrame(data=intQuantities, index=index, columns=columns)

    @property
    def __timers_results(self):
        n_blocks = np.prod(self.domain_shape)
        n_proc = self.n_mpi_processes
        n_steps = self.n_steps
        n_blocks_per_proc = np.ceil(n_blocks / float(n_proc))

        timers = ['Set initial conditions',
                  'computeLocalIQ',
                  'Reduce/Write',
                  'Gather/Write',
                  'sedov simulation',
                  'GC Fill']

        times_sec = {}
        for key in timers:
            assert(key not in times_sec)
            times_sec[key] = {}
            times_sec[key]['start'] = []
            times_sec[key]['end']   = []

        with open(self.__fname_log, 'r') as fptr:
            for line in fptr.readlines():
                if '[Logger] Terminated at' in line:
                    wtime_total_sec = float(line.split()[0])

                for key in timers:
                    if '[Timer] Start timing ' + key in line:
                        times_sec[key]['start'].append( float(line.split()[0]) )
                    if '[Timer] Stop timing ' + key in line:
                        times_sec[key]['end'].append( float(line.split()[0]) )

        for key in timers:
            start = np.array(times_sec[key]['start'])
            end   = np.array(times_sec[key]['end'])
            assert(len(start) == len(end))
        for key in ['Set initial conditions', 'sedov simulation']:
            assert(len(times_sec[key]['start']) == 1)

        wtimes_sec = {}
        for key in times_sec:
            assert(key not in wtimes_sec)
            start = np.array(times_sec[key]['start'])
            end   = np.array(times_sec[key]['end'])
            wtimes_sec[key] = np.sum(end - start)

        wtime_setup_sec    = times_sec['Set initial conditions']['start'][0]
        wtime_init_sec     = times_sec['sedov simulation']['start'][0]
        wtime_ic_sec       = wtimes_sec['Set initial conditions']
        wtime_sim_sec      = wtimes_sec['sedov simulation']
        wtime_gc_sec       = wtimes_sec['GC Fill']
        wtime_finalize_sec = wtime_total_sec - times_sec['sedov simulation']['end'][0]
        wtime_comp_sec     = self.timings_per_step.max_wtime_sec.sum()
        wtime_iq_sec       = wtimes_sec['computeLocalIQ']
        wtime_gather_sec   = wtimes_sec['Gather/Write']
        wtime_reduce_sec   = wtimes_sec['Reduce/Write']

        msg  = f'Total walltime\t\t\t\t\t{wtime_total_sec} s\n'
        msg += f'\tInitialization walltime\t\t\t\t{wtime_init_sec} s\n'
        msg += f'\t\tSetup walltime\t\t\t\t\t{wtime_setup_sec} s\n'
        msg += f'\t\tICs walltime\t\t\t\t\t{wtime_ic_sec} s\n'
        msg += f'\t\tICs walltime/block\t\t\t\t{wtime_ic_sec / float(n_blocks_per_proc*n_steps)} s\n'
        msg += f'\tSimulation walltime\t\t\t\t{wtime_sim_sec} s\n'
        msg += f'\t\tGC fill walltime\t\t\t\t{wtime_gc_sec} s\n'
        msg += f'\t\tComputation walltime\t\t\t\t{wtime_comp_sec} s\n'
        msg += f'\t\tComputation walltime/block\t\t\t{wtime_comp_sec / float(n_blocks_per_proc*n_steps)} s\n'
        msg += f'\t\tCompute Local IQ walltime\t\t\t{wtime_iq_sec} s\n'
        msg += f'\t\tGather walltime\t\t\t\t\t{wtime_gather_sec} s\n'
        msg += f'\t\tReduction walltime\t\t\t\t{wtime_reduce_sec} s\n'
        msg += f'\tFinalization walltime\t\t\t\t{wtime_finalize_sec} s\n'

        return msg

    def __str__(self):
        """
        """
        MSEC_TO_SEC = 1.0e3
        USEC_TO_SEC = 1.0e6

        nxb, nyb, nzb = self.blocks_shape
        n_blocks_x, n_blocks_y, n_blocks_z = self.domain_shape

        iq_df = self.integral_quantities_statistics
        M0     = iq_df.loc['Initial_Mass', 'summary_stats']
        M_mean = iq_df.loc['Mean_Mass_Err', 'summary_stats']
        M_std  = iq_df.loc['Std_Mass_Err', 'summary_stats']
        M_max  = iq_df.loc['Max_Mass_Err', 'summary_stats']
        E0     = iq_df.loc['Initial_Energy', 'summary_stats']
        E_mean = iq_df.loc['Mean_Energy_Err', 'summary_stats']
        E_std  = iq_df.loc['Std_Energy_Err', 'summary_stats']
        E_max  = iq_df.loc['Max_Energy_Err', 'summary_stats']

        msg  = f'Filename\t\t\t\t{self.__path}\n'
        msg += f'Log  Filename\t\t\t{self.__fname_log.name}\n'
        msg += f'Data Filename\t\t\t{self.__fname_data.name}\n'
        if self.__fname_plot == '':
            msg += 'Plofile Name\t\t\tNot Given\n'
        else:
            msg += f'Plotfile Name\t\t\t{self.__fname_plot.name}\n'
        msg += f'Hydro Filename\t\t\t{self.__fname_hydro.name}\n'
        msg += f'Study name\t\t\t{self.study_name}\n'
        msg += f'N MPI Processes\t\t\t{self.n_mpi_processes}\n'
        msg += f'N Steps\t\t\t\t{self.n_steps}\n'
        msg += f'Dimension\t\t\t{self.dimension}\n'
        msg += f'NXB\t\t\t\t{nxb}\n'
        msg += f'NYB\t\t\t\t{nyb}\n'
        msg += f'NZB\t\t\t\t{nzb}\n'
        msg += f'N_BLOCKS_X\t\t\t{n_blocks_x}\n'
        msg += f'N_BLOCKS_Y\t\t\t{n_blocks_y}\n'
        msg += f'N_BLOCKS_Z\t\t\t{n_blocks_z}\n'
        msg += f'N Distributor Threads\t\t{self.n_distributor_threads}\n'
        msg += f'N CPU         Threads\t\t{self.n_cpu_threads}\n'
        msg += f'N GPU         Threads\t\t{self.n_gpu_threads}\n'
        msg += f'N Blocks/Packet\t\t\t{self.n_blocks_per_packet}\n'
        msg += f'N Blocks/CPU Turn\t\t{self.n_blocks_per_cpu_turn}\n'
        msg += f'Timer Resolution\t\t{self.timer_resolution_sec} sec\n'

        if self.__packet_timings.n_packets > 0:
            packet_df = self.__packet_timings.timings

            n_blocks_all = sorted(packet_df.nblocks.unique())
            msg += '\n'
            msg += f'N Packets\t\t\t{self.__packet_timings.n_packets}\n'
            msg += f'N Blocks\t\t\t{self.__packet_timings.n_blocks}\n'
            msg += f'Packet Sizes\t\t\t{n_blocks_all} Blocks/Packet\n'

            todo = [('wtime_pack_sec',   'Pack'), \
                    ('wtime_async_sec',  'Async'), \
                    ('wtime_packet_sec', 'Packet')]
            for col, name in todo:
                data_sec = packet_df[col] / packet_df.nblocks
                min_wtime_us    = data_sec.min()    * USEC_TO_SEC
                mean_wtime_us   = data_sec.mean()   * USEC_TO_SEC
                median_wtime_us = data_sec.median() * USEC_TO_SEC
                max_wtime_us    = data_sec.max()    * USEC_TO_SEC
                std_wtime_us    = data_sec.std()    * USEC_TO_SEC
                msg += '\n'
                msg += f'Min    {name} Walltime/Block\t{min_wtime_us} us\n'
                msg += f'Mean   {name} Walltime/Block\t{mean_wtime_us} us\n'
                msg += f'Median {name} Walltime/Block\t{median_wtime_us} us\n'
                msg += f'Max    {name} Walltime/Block\t{max_wtime_us} us\n'
                msg += f'StdDev {name} Walltime/Block\t{std_wtime_us} us\n'

        msg += '\n'
        msg += f'Initial Mass\t\t\t{M0}\n'
        msg += f'Mean(Mass Error)\t\t{M_mean}\n'
        msg += f'Std(Mass Error)\t\t\t{M_std}\n'
        msg += f'Max(|Mass Error|)\t\t{M_max}\n'
        msg += f'Initial Total Energy\t\t{E0}\n'
        msg += f'Mean(Total Energy Error)\t{E_mean}\n'
        msg += f'Std(Total Energy Error)\t\t{E_std}\n'
        msg += f'Max(|Total Energy Error|)\t{E_max}\n'
        msg += '\n'
        msg += self.__timers_results

        return msg

