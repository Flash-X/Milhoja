import matplotlib.figure as mfig

class MplConservedQuantities(mfig.Figure):
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

    def draw_plot(self, iq_df, n_bins, file_info):
        """
        """
        title = f'Sedov Conserved Integral Quantity Timeseries\n{file_info}'

        steps    = iq_df.loc[1:].index
        mass_err = iq_df.loc[1:].mass.values    - iq_df.loc[0].mass
        etot_err = iq_df.loc[1:].E_total.values - iq_df.loc[0].E_total

        self.clear()

        self.suptitle(title, fontsize=self.__fontsize)

        subp = self.add_subplot(221)
        subp.set_title('Mass', fontsize=self.__fontsize)
        subp.plot(steps, mass_err, '.k', markersize=self.__markersize)
        subp.set_xlabel('Step', fontsize=self.__fontsize)
        subp.set_ylabel('Error', fontsize=self.__fontsize)
        subp.grid(True)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

        subp = self.add_subplot(222)
        subp.set_title('Total Energy', fontsize=self.__fontsize)
        subp.plot(steps, etot_err, '.r', markersize=self.__markersize)
        subp.set_xlabel('Step', fontsize=self.__fontsize)
        subp.set_ylabel('Error', fontsize=self.__fontsize)
        subp.grid(True)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

        subp = self.add_subplot(223)
        subp.hist(mass_err, bins=n_bins, color='black')
        subp.set_xlabel('Error', fontsize=self.__fontsize)
        subp.set_ylabel('Counts', fontsize=self.__fontsize)
        subp.grid(True)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

        subp = self.add_subplot(224)
        subp.hist(etot_err, bins=n_bins, color='red')
        subp.set_xlabel('Error', fontsize=self.__fontsize)
        subp.set_ylabel('Counts', fontsize=self.__fontsize)
        subp.grid(True)
        for each in self.gca().get_xticklabels():
            each.set_fontsize(self.__fontsize)
        for each in self.gca().get_yticklabels():
            each.set_fontsize(self.__fontsize)

