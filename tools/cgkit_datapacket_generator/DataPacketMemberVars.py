class DataPacketMemberVars:
    """
    This class stores information from items in the data packet for keeping data mapping consistent.
    """

    #: The name of the item.
    ITEM: str
    #: The data type of the item.
    dtype: str
    #: The equation for the size of the item.
    SIZE_EQ: str
    #: Whether or not the item appears per tile or only once.
    PER_TILE: bool

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

    def get_size(self, use_total_size: bool):
        """
        Returns the size **variable** for *item*. This is not the size equation.

        :param bool use_total_size: Returns the size variable times nTiles instead of just the size variable.
        """
        total = '' if not use_total_size else '_nTiles_h * '
        return f'{total}SIZE_{self.ITEM.upper()}'

    def get_host(self):
        """Returns the item formatted for as a host variable."""
        return f'_{self.ITEM}_h'

    def get_device(self):
        """Returns the item name formatted as a device pointer variable."""
        return f'_{self.ITEM}_d'
    
    def get_pinned(self):
        """Returns the item name formatted as a pinned pointer variable."""
        return f'_{self.ITEM}_p'