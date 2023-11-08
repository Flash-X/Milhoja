import numpy as np

# ----- SETS OF ITEMS OF WRONG TYPE FOR TYPE CHECKING A SINGLE VARIABLE
NOT_STR_LIST = [
    None,
    -1, 0, 1, 1.1, -2.2, np.nan, np.inf,
    (), (1,), [], [1], set(), {1}, {}, {"a": 1}
]
NOT_INT_LIST = [
    None,
    "fail",
    1.1, -2.2, np.nan, np.inf,
    (), (1,), [], [1], set(), {1}, {}, {"a": 1}
]
NOT_LIST_LIST = [
    None,
    "fail",
    -1, 0, 1, 1.1, -2.2, np.nan, np.inf,
    (), (1,), set(), {1}, {}, {"a": 1}
]
NOT_DICT_LIST = [
    None,
    "fail",
    -1, 0, 1, 1.1, -2.2, np.nan, np.inf,
    (), (1,), [], [1], set(), {1}
]
NOT_CLASS_LIST = [
    None,
    "fail",
    -1, 0, 1, 1.1, -2.2, np.nan, np.inf,
    (), (1,), [], [1], set(), {1}, {}, {"a": 1}
]
