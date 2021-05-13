import matplotlib.cm as cm
import matplotlib.figure as mfig

class MplPacketWalltimesByBlock(mfig.Figure):
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

    def draw_plot(self, packet_df, bins, logbins, file_info):
        """
        """
        USEC_TO_SEC = 1.0e6
       
        title = f'Packet Walltime Information By Block\n{file_info}'

        self.clear()

        self.suptitle(title, fontsize=self.__fontsize)

        subp = self.add_subplot(131)
        subp.set_title(f'Pack', fontsize=self.__fontsize)
        subp.hist(packet_df.wtime_pack_sec * USEC_TO_SEC / packet_df.nblocks, \
                  bins=bins, density=True)
        subp.set_xlabel('Walltime/Block ($\mu$s)', fontsize=self.__fontsize)
        subp.set_ylabel('Density', fontsize=self.__fontsize)

        subp = self.add_subplot(132)
        subp.set_title(f'Initiate Async Xfer', fontsize=self.__fontsize)
        subp.hist(packet_df.wtime_async_sec * USEC_TO_SEC / packet_df.nblocks, \
                  bins=logbins, density=True)
        subp.set_xscale('log')
        subp.set_yscale('log', nonpositive='clip')
        subp.set_xlabel('Walltime/Block ($\mu$s)', fontsize=self.__fontsize)

        subp = self.add_subplot(133)
        subp.set_title(f'Total Time', fontsize=self.__fontsize)
        subp.hist(packet_df.wtime_packet_sec * USEC_TO_SEC / packet_df.nblocks, \
                  bins=bins, density=True)
        subp.set_xlabel('Walltime/Block ($\mu$s)', fontsize=self.__fontsize)

