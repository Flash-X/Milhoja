import matplotlib.cm as cm
import matplotlib.figure as mfig

from sedov import YT_VAR_LUT
from sedov import YT_N_VARS

class MplSolutionComparison(mfig.Figure):
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

    @property
    def fontsize(self):
        return self.__fontsize

    @fontsize.setter
    def fontsize(self, size):
        self.__fontsize = size

    def draw_plot(self, z, result_A, file_info_A, result_B, file_info_B):
        """
        """
        title  = f'Sedov Solution Z-Slice Comparison (z = {z})\n'
        title += f'A = {file_info_A} / B = {file_info_B}'

        self.clear()

        self.suptitle(title, fontsize=self.__fontsize)

        for j, (var, var_name) in enumerate(YT_VAR_LUT):
            var_idx = 'var{:04}'.format(var)
            extent_A, data_A = result_A.z_data_slice(z, var_idx)
            extent_B, data_B = result_B.z_data_slice(z, var_idx)
            assert(extent_A == extent_B)

            subp = self.add_subplot(2, YT_N_VARS, j+1)
            im = subp.imshow(data_A, interpolation='nearest', \
                             extent=extent_A, cmap=cm.Blues_r)
            subp.set_xlabel('X (a.u.)', fontsize=self.__fontsize)
            subp.set_ylabel('Y (a.u.)', fontsize=self.__fontsize)
            cbar = self.colorbar(im)
            cbar.set_label(label=f'{var_name} A', size=self.__fontsize)
            subp.grid(False)
            for each in self.gca().get_xticklabels():
                each.set_fontsize(self.__fontsize - 2)
            for each in self.gca().get_yticklabels():
                each.set_fontsize(self.__fontsize - 2)

            subp = self.add_subplot(2, YT_N_VARS, YT_N_VARS+j+1)
            im = subp.imshow(data_A - data_B, interpolation='nearest', \
                             extent=extent_A, cmap=cm.Blues_r)
            subp.set_xlabel('X (a.u.)', fontsize=self.__fontsize)
            subp.set_ylabel('Y (a.u.)', fontsize=self.__fontsize)
            cbar = self.colorbar(im)
            cbar.set_label(label='A - B', size=self.__fontsize)
            subp.grid(False)
            for each in self.gca().get_xticklabels():
                each.set_fontsize(self.__fontsize - 2)
            for each in self.gca().get_yticklabels():
                each.set_fontsize(self.__fontsize - 2)

