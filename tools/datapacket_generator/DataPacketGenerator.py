from milhoja import BaseCodeGenerator

class DataPacketGenerator(BaseCodeGenerator):
    
    def __init__(
        self,
        tf_spec,
        header_filename,
        source_filename,
        log_tag,
        log_level,
        indent
    ):
        super().__init__();1
