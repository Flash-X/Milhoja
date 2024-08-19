from dataclasses import dataclass

# this is my attempt at making it easier to add new tile source types.


@dataclass
class TileSourceData:
    name: str
    size: str
    dtype: str

##    def get_formattable_memcpy_string(self):
##        formatted = f"{self.dtype} " + "{name}" + f"[{self.size}] = "


TILE_LO_DATA = TileSourceData("lo", "MILHOJA_MDIM", "real")
TILE_LEVEL_DATA = TileSourceData("level", "1", "unsigned int")
