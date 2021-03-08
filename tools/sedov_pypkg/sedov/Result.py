import numpy as np
import pandas as pd

from pathlib import Path

from sedov import IQ_XMOM_IDX
from sedov import IQ_YMOM_IDX
from sedov import IQ_ZMOM_IDX
from sedov import IQ_MASS_IDX
from sedov import IQ_ETOT_IDX
from sedov import IQ_THRESHOLD_VEL
from sedov import IQ_THRESHOLD_MASS
from sedov import IQ_THRESHOLD_ETOT

class Result(object):
    """
    A class for accessing the results of a single Sedov test run and that
    automatically applies some sanity checks to the results.
    """
    def __init__(self, fname_plot, fname_data, fname_log):
        """
        Create an instance of the class for accessing the results associated
        with the given individual result files.  It is generally intended that
        all the given files were been created as part of the same execution of
        the Sedov test problem.

        Parameters:
            fname_plot - the full path to the AMReX-format folder containing the
                         solution at the end of the simulation
            fname_data - the filename including full path of the ASCII-format
                         file that contains the integral quantities as computed
                         at each timestep.
            fname_log  - the filename including full path of the ASCII-format
                         file that contains the simulation log information.
        Returns
            The object
        """
        self.__fname_plot = Path(fname_plot)
        self.__fname_data = Path(fname_data)
        self.__fname_log  = Path(fname_log)

        super().__init__()

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
#        assert(data.shape[0] == self.n_steps + 1)

        # Conserved quantities
        assert(data[0, IQ_XMOM_IDX] == 0.0)
        assert(data[0, IQ_YMOM_IDX] == 0.0)
        assert(data[0, IQ_ZMOM_IDX] == 0.0)
        assert(np.max(np.abs(data[:, IQ_XMOM_IDX])) < IQ_THRESHOLD_VEL)
        assert(np.max(np.abs(data[:, IQ_YMOM_IDX])) < IQ_THRESHOLD_VEL)
        assert(np.max(np.abs(data[:, IQ_ZMOM_IDX])) < IQ_THRESHOLD_VEL)

        mass_err = data[1:, IQ_MASS_IDX] - data[0, IQ_MASS_IDX]
        assert(np.max(np.abs(mass_err)) < IQ_THRESHOLD_MASS)

        etot_err = data[1:, IQ_ETOT_IDX] - data[0, IQ_ETOT_IDX]
        assert(np.max(np.abs(etot_err)) < IQ_THRESHOLD_ETOT)

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

        mass_err = data[1:, IQ_MASS_IDX] - data[0, IQ_MASS_IDX]
        etot_err = data[1:, IQ_ETOT_IDX] - data[0, IQ_ETOT_IDX]

        intQuantities[0] = data[0, IQ_MASS_IDX]
        intQuantities[1] = np.mean(mass_err)
        intQuantities[2] = np.std(mass_err)
        intQuantities[3] = np.max(np.abs(mass_err))

        intQuantities[4] = data[0, IQ_ETOT_IDX]
        intQuantities[5] = np.mean(etot_err)
        intQuantities[6] = np.std(etot_err)
        intQuantities[7] = np.max(np.abs(etot_err))

        columns = ['summary_stats']
        return pd.DataFrame(data=intQuantities, index=index, columns=columns)

    def __str__(self):
        """
        """
        iq_df = self.integral_quantities_statistics
        M0     = iq_df.loc['Initial_Mass', 'summary_stats']
        M_mean = iq_df.loc['Mean_Mass_Err', 'summary_stats']
        M_std  = iq_df.loc['Std_Mass_Err', 'summary_stats']
        M_max  = iq_df.loc['Max_Mass_Err', 'summary_stats']
        E0     = iq_df.loc['Initial_Energy', 'summary_stats']
        E_mean = iq_df.loc['Mean_Energy_Err', 'summary_stats']
        E_std  = iq_df.loc['Std_Energy_Err', 'summary_stats']
        E_max  = iq_df.loc['Max_Energy_Err', 'summary_stats']

        msg  = f'Log  Filename\t\t\t{self.__fname_log.name}\n'
        msg += f'Log  Filename Path\t\t{self.__fname_log.parent}\n'
        msg += f'Data Filename\t\t\t{self.__fname_data.name}\n'
        msg += f'Data Filename Path\t\t{self.__fname_data.parent}\n'
        if self.__fname_plot == '':
            msg += 'Plofile Name\t\t\tNot Given\n'
        else:
            msg += f'Plotfile Name\t\t\t{self.__fname_plot.name}\n'
            msg += f'Plotfile Name Path\t\t{self.__fname_plot.parent}\n'

        msg += '\n'
        msg += f'Initial Mass\t\t\t{M0}\n'
        msg += f'Mean(Mass Error)\t\t{M_mean}\n'
        msg += f'Std(Mass Error)\t\t\t{M_std}\n'
        msg += f'Max(|Mass Error|)\t\t{M_max}\n'
        msg += f'Initial Total Energy\t\t{E0}\n'
        msg += f'Mean(Total Energy Error)\t{E_mean}\n'
        msg += f'Std(Total Energy Error)\t\t{E_std}\n'
        msg += f'Max(|Total Energy Error|)\t{E_max}\n'

        return msg

