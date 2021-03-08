import matplotlib.cm as cm
import matplotlib.figure as mfig

from sedov import YT_VAR_LUT

class MplFinalSolution(mfig.Figure):
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

    def draw_plot(self, result, z, file_info):
        """
        """
        title = f'Sedov Final Solution Z-Slices (z = {z})\n{file_info}'

        self.clear()

        self.suptitle(title, fontsize=self.__fontsize)

        for j, (var, var_name) in enumerate(YT_VAR_LUT):
            var_idx = 'var{:04}'.format(var)
            extent, data = result.z_data_slice(z, var_idx)

            subp = self.add_subplot(2, 4, j+1)
            im = subp.imshow(data, interpolation='nearest', extent=extent, cmap=cm.Blues_r)
            subp.set_xlabel('X (a.u.)', fontsize=self.__fontsize)
            subp.set_ylabel('Y (a.u.)', fontsize=self.__fontsize)
            cbar = self.colorbar(im)
            cbar.set_label(label=var_name, size=self.__fontsize)
            subp.grid(False)
            for each in self.gca().get_xticklabels():
                each.set_fontsize(self.__fontsize - 2)
            for each in self.gca().get_yticklabels():
                each.set_fontsize(self.__fontsize - 2)

