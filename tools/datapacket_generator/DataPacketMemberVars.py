class DataPacketMemberVars:
    """
    This class stores information from items in the data packet for keeping data mapping consistent.
    """

    def __init__(self, item: str, dtype: str, size_eq: str, per_tile: bool):
        """
        Initializer for DataPacketMemberVars.

        :param str item: The name of the data packet item.
        :param str dtype: The data type of *item*.
        :param str size_eq: The equation for determining the size of *item*.
        :param bool per_tile: Flag if the item appears per tile.
        """
        self.ITEM = item
        self.dtype = dtype
        self.SIZE_EQ = size_eq
        self.PER_TILE = per_tile

    @property
    def size(self):
        """
        Returns the size **variable** for *item*. This is not the size equation.

        :param bool use_total_size: Returns the size variable times nTiles instead of just the size variable.
        """
        return f'SIZE_{self.ITEM.upper()}'
    
    @property
    def total_size(self):
        total = '' if not self.PER_TILE else '_nTiles_h * '
        return f'{total}SIZE_{self.ITEM.upper()}'
    
    @property
    def host(self):
        """Returns the item formatted for as a host variable."""
        return f'_{self.ITEM}_h'

    @property
    def device(self):
        """Returns the item name formatted as a device pointer variable."""
        return f'_{self.ITEM}_d'
    
    @property
    def pinned(self):
        """Returns the item name formatted as a pinned pointer variable."""
        return f'_{self.ITEM}_p'