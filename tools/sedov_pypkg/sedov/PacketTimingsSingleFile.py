import numpy as np
import pandas as pd

from pathlib import Path

# TODO: This needs to be written and tested properly.

class PacketTimingsSingleFile(object):
    def __init__(self, filename):
        super().__init__()

        self.__fname = Path(filename)
        with open(self.__fname, 'r') as fptr:
            comments = (each.lstrip().lstrip('#') for each in fptr.readlines() \
                                                  if each.lstrip()[0] == '#')
            comments = (each.split('=') for each in comments \
                                        if len(each.split('=')) == 2)
            pairs = ((each[0].lstrip().rstrip(), each[1].lstrip().rstrip()) \
                        for each in comments)
            self.__hdr = dict(pairs)

        # TODO: Confirm that header contains the line
        # thread,packet,nblocks,walltime_pack_sec,walltime_async_sec,walltime_packet_sec
        # since we have hardcoded that assumption here
        columns = ['nblocks', \
                   'wtime_pack_sec', \
                   'wtime_async_sec', \
                   'wtime_packet_sec']
        data = np.loadtxt(self.__fname, delimiter=',')

        names = ['step', 'proc', 'thread', 'packet']
        step = [int(self.__hdr['Step'])]     * data.shape[0]
        rank = [int(self.__hdr['MPI rank'])] * data.shape[0]
        threads = data[:, 0].astype(int)
        packets = data[:, 1].astype(int)
        idx = zip(step, rank, threads, packets)
        idx = pd.MultiIndex.from_tuples(idx, names=names)

        self.__df = pd.DataFrame(data[:, 2:], index=idx, columns=columns)
        self.__df['nblocks'] = self.__df.nblocks.astype(int)
        
    @property
    def filename(self):
        return self.__fname

    @property
    def study_name(self):
        return self.__hdr['Testname']
    
    @property
    def step(self):
        return int(self.__hdr['Step'])
    
    @property
    def mpi_rank(self):
        return int(self.__hdr['MPI rank'])

    @property
    def dimension(self):
        return int(self.__hdr['Dimension'])

    @property
    def blocks_shape(self):
        return (int(self.__hdr['NXB']), \
                int(self.__hdr['NYB']), \
                int(self.__hdr['NZB']))

    @property
    def n_packets(self):
        return len(self.__df)

    @property
    def n_blocks(self):
        return self.__df.nblocks.sum()

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
    def timings(self):
        return self.__df

    def __str__(self):
        nxb, nyb, nzb = self.blocks_shape

        msg  = f'Filename\t\t\t{self.filename.name}\n'
        msg += f'File Path\t\t\t{self.filename.parent}\n'
        msg += f'Study name\t\t\t{self.study_name}\n'
        msg += f'Step\t\t\t\t{self.step}\n'
        msg += f'Rank\t\t\t\t{self.mpi_rank}\n'
        msg += f'Dimension\t\t\t{self.dimension}\n'
        msg += f'NXB\t\t\t\t{nxb}\n'
        msg += f'NYB\t\t\t\t{nyb}\n'
        msg += f'NZB\t\t\t\t{nzb}\n'
        msg += f'N Distributor Threads\t\t{self.n_distributor_threads}\n'
        msg += f'N GPU         Threads\t\t{self.n_gpu_threads}\n'
        msg += f'N CPU         Threads\t\t{self.n_cpu_threads}\n'
        msg += f'N Blocks/Packet\t\t\t{self.n_blocks_per_packet}\n'
        msg += f'N Blocks/CPU Turn\t\t{self.n_blocks_per_cpu_turn}\n'
        msg += f'N Packets\t\t\t{self.n_packets}\n'
        msg += f'N Blocks\t\t\t{self.n_blocks}\n'

        return msg

