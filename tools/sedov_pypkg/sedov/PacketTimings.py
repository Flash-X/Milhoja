import pandas as pd

from pathlib import Path

from sedov import PacketTimingsSingleFile

# TODO: This needs to be written and tested properly.

class PacketTimings(object):
    def __init__(self, folder):
        super().__init__()

        self.__path = Path(folder)

        idx = []
        for fname in self.__path.glob('timings_packet_step*_rank*.dat'):
            info = fname.name.replace('.dat', '').replace('timings_packet_', '').split('_')
            # TODO: Get these from the header
            step = int(info[0].replace('step', ''))
            rank = int(info[1].replace('rank', ''))
            idx.append( (step, rank) )

        # TODO: Confirm that all have the same setup
        dfs = []
        for step, rank in sorted(idx):
            fname = self.__path.joinpath(f'timings_packet_step{step}_rank{rank}.dat')
            result = PacketTimingsSingleFile(fname)
            dfs.append( result.timings )

        self.__df = pd.concat(dfs)
        self.__hdr = result

    @property
    def filename(self):
        return self.__path

    @property
    def study_name(self):
        return self.__hdr.study_name

    @property
    def dimension(self):
        return self.__hdr.dimension

    @property
    def blocks_shape(self):
        return self.__hdr.blocks_shape

    @property
    def n_packets(self):
        return len(self.__df)
    
    @property
    def n_blocks(self):
        return self.__df.nblocks.sum()

    @property
    def n_distributor_threads(self):
        return self.__hdr.n_distributor_threads

    @property
    def n_cpu_threads(self):
        return self.__hdr.n_cpu_threads

    @property
    def n_gpu_threads(self):
        return self.__hdr.n_gpu_threads

    @property
    def n_blocks_per_packet(self):
        return self.__hdr.n_blocks_per_packet

    @property
    def n_blocks_per_cpu_turn(self):
        return self.__hdr.n_blocks_per_cpu_turn

    @property
    def timer_resolution_sec(self):
        return self.__hdr.timer_resolution_sec

    @property
    def timings(self):
        return self.__df

    def __str__(self):
        nxb, nyb, nzb = self.blocks_shape

        msg  = f'Source Folder\t\t\t{self.filename}\n'
        msg += f'Study name\t\t\t{self.study_name}\n'
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

