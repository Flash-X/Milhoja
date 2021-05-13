import matplotlib.cm as cm
import matplotlib.figure as mfig

class MplPacketWalltimes(mfig.Figure):
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
        MSEC_TO_SEC = 1.0e3
        USEC_TO_SEC = 1.0e6
        
        n_blocks_all = sorted(packet_df.nblocks.unique())
        n_rows = len(n_blocks_all)
        n_cols = 3

        title = f'Packet Walltime Information\n{file_info}'

        self.clear()

        self.suptitle(title, fontsize=self.__fontsize)

        for j, n_blocks in enumerate(n_blocks_all):
            df = packet_df[packet_df.nblocks == n_blocks]
        
            if j == 0:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+1)
                subp.set_title(f'Pack - {n_blocks}/Packet', \
                               fontsize=self.__fontsize)
                subp1 = subp
            else:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+1, sharex=subp1)
                subp.set_title(f'{n_blocks} Blocks/Packet', \
                               fontsize=self.__fontsize)
            subp.hist(df.wtime_pack_sec * MSEC_TO_SEC, bins=bins, density=True)
            if j == n_rows - 1:
                subp.set_xlabel('Walltime (ms)', fontsize=self.__fontsize)
            subp.set_ylabel('Density', fontsize=self.__fontsize)

            if j == 0:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+2)
                subp.set_title(f'Initiate Async Xfer - {n_blocks}/Packet', \
                               fontsize=self.__fontsize)
                subp2 = subp
            else:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+2, sharex=subp2)
                subp.set_title(f'{n_blocks} Blocks/Packet', \
                               fontsize=self.__fontsize)
            subp.hist(df.wtime_async_sec * USEC_TO_SEC, \
                      bins=logbins, density=True)
            subp.set_xscale('log')
            subp.set_yscale('log', nonpositive='clip')
            if j == n_rows - 1:
                subp.set_xlabel('Walltime ($\mu$s)', fontsize=self.__fontsize)

            if j == 0:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+3)
                subp.set_title(f'Total Time - {n_blocks}/Packet', \
                               fontsize=self.__fontsize)
                subp3 = subp
            else:
                subp = self.add_subplot(n_rows,n_cols,j*n_cols+3, sharex=subp3)
                subp.set_title(f'{n_blocks} Blocks/Packet', fontsize=self.__fontsize)
            subp.hist(df.wtime_packet_sec * MSEC_TO_SEC, bins=bins, density=True)
            if j == n_rows - 1:
                subp.set_xlabel('Walltime (ms)', fontsize=self.__fontsize)

