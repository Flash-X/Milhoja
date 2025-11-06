from os import getenv

opts = {}

nxyzb_args = 'nxyzb_args'
nxyzt_args = 'nxyzt_args'
nxyzb_mod = 'nxyzb_mod'

opts[nxyzb_args] = False
opts[nxyzt_args] = False
opts[nxyzb_mod] = False

opts['computation_offloading'] = getenv('ORCHA_GPU_OFFLOADING')
if not opts['computation_offloading']:  # fallback if envvar not set
    opts['computation_offloading'] = 'OpenACC'  # alternative: 'OpenMP'

opts['MILHOJA_USE_TARGET_ASYNC'] = False
if getenv('MILHOJA_USE_TARGET_ASYNC'):
    opts['MILHOJA_USE_TARGET_ASYNC'] = True

opts['emit_HDA'] = True

opts['use_omp_requires'] = (opts['computation_offloading'] == 'OpenMP')
