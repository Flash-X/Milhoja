import numpy as np

from sedov import IQ_THRESHOLD_MASS
from sedov import IQ_THRESHOLD_ETOT
from sedov import IQ_THRESHOLD_KE
from sedov import IQ_THRESHOLD_EINT

def compare_integral_quantities(result_A, result_B):
    """
    Given two Sedov test results, compare all non-conserved quantities to
    confirm that these are sufficiently similar at all timesteps.  This function
    also confirms that the initial value of all conserved quantities are
    sufficiently similar.  Thresholds used to assess if two quantities are
    sufficiently similar are defined in the sedov package with prefix
    IQ_THRESHOLD.

    It is assumed that sufficient conservation of all conserved quantities has
    already been confirmed for each individual result.

    A ValueError exception is raised if a significant difference is discovered.

    Parameters:
        result_A - the Result object for the first test result
        result_B - the Result object for the second test result
    Returns:
        Nothing
    """
    # TODO: These should all be put under test
    iq_A_df = result_A.integral_quantities
    iq_B_df = result_B.integral_quantities

    # This should fail if the two contain a different number of timesteps
    if   not all(iq_A_df.index == iq_B_df.index):
        raise ValueError('A and B have different steps')
    elif not all(iq_A_df.time == iq_B_df.time):
        raise ValueError('A and B have different times')

    max_ke_abserr = np.max(np.abs(iq_A_df.E_kinetic - iq_B_df.E_kinetic))
    if   max_ke_abserr > IQ_THRESHOLD_KE:
        msg = 'Kinetic energy absolute error {} exceeds threshold {}'
        raise ValueError(msg.format(max_ke_abserr, IQ_THRESHOLD_KE))

    max_eint_abserr = np.max(np.abs(iq_A_df.E_internal - iq_B_df.E_internal))
    if   max_eint_abserr > IQ_THRESHOLD_EINT:
        msg = 'Internal energy absolute error {} exceeds threshold {}'
        raise ValueError(msg.format(max_eint_abserr, IQ_THRESHOLD_EINT))

    abserr = np.abs(iq_A_df.loc[0].mass - iq_B_df.loc[0].mass)
    if abserr > IQ_THRESHOLD_MASS:
        msg = 'Initial mass absolute error {} larger than threshold {}'
        raise ValueError(msg.format(abserr, IQ_THRESHOLD_MASS))

    abserr = np.abs(iq_A_df.loc[0].E_total - iq_B_df.loc[0].E_total)
    if abserr > IQ_THRESHOLD_ETOT:
        msg = 'Initial total energy absolute error {} larger than threshold {}'
        raise ValueError(msg.format(abserr, IQ_THRESHOLD_ETOT))

    for axis in ['x', 'y', 'z']:
        if   iq_A_df.loc[0, f'{axis}_momentum'] != 0.0:
            raise ValueError(f'Initial A {axis}-momentum is non-zero')
        elif iq_B_df.loc[0, f'{axis}_momentum'] != 0.0:
            raise ValueError(f'Initial B {axis}-momentum is non-zero')

