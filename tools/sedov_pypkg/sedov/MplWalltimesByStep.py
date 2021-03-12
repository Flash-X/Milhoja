import matplotlib.figure as mfig

class MplWalltimesByStep(mfig.Figure):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        """
        self.__linewidth = 2
        self.__markersize = 5
        self.__fontsize = 16

        # Provide default geometry of Figure
        if not 'subplotpars' in kwargs.keys():
            subp = mfig.SubplotParams(left=0.10, top=0.95, \
                                      bottom=0.08, right=0.95, \
                                      hspace=0.001, wspace=0.001)
            kwargs['subplotpars'] = subp

        mfig.Figure.__init__(self, *args, **kwargs)

    def draw_plot(self, result, n_bins, file_info, time_unit):
        """
        """
        if   time_unit.lower() ==  's':
            conv = 1.0
        elif time_unit.lower() == 'ms':
            conv = 1.0e3
        elif time_unit.lower() == 'us':
            conv = 1.0e6
        else:
            raise ValueError(f'Invalid time unit {time_unit}')

        wtime_sec_df = result.raw_timings
        max_wtime_df = result.timings_per_step

        title = f'Walltime Results by Step - {file_info}'

        self.clear()

        gs = self.add_gridspec(2, 2,  \
                               width_ratios=(7, 2), \
                               height_ratios=(1, 1),
                               wspace=0.05, hspace=0.2)

        self.suptitle(title, fontsize=self.__fontsize)

        subpy = self.add_subplot(gs[0, 0])
        for proc in range(1, result.n_mpi_processes + 1):
            df = wtime_sec_df.xs(proc, level='proc')
            subpy.plot(df.index, \
                       df.wtime_per_blk_sec.values * conv, '.')
        subpy.set_ylabel(f'Time/Blk/Step ({time_unit})', \
                         fontsize=self.__fontsize)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

        subp = self.add_subplot(gs[0, 1], sharey=subpy)
        subp.hist(wtime_sec_df.wtime_per_blk_sec * conv, \
                  bins=n_bins, density=True, orientation='horizontal')
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_visible(False)

        subpy = self.add_subplot(gs[1, 0], sharex=subpy)
        subpy.plot(max_wtime_df.index, \
                   max_wtime_df.max_wtime_sec.values * conv, '.')
        subpy.set_xlabel('Timestep', \
                         fontsize=self.__fontsize)
        subpy.set_ylabel(f'Max Time/Step ({time_unit})', \
                         fontsize=self.__fontsize)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

        subp = self.add_subplot(gs[1, 1], sharey=subpy)
        subp.hist(max_wtime_df.max_wtime_sec.values * conv, \
                  bins=n_bins, density=True, orientation='horizontal')
        subp.set_xlabel('PDF', fontsize=self.__fontsize)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_visible(False)

