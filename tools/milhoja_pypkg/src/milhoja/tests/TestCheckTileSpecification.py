"""
Automatic unit testing of check_tile_specification()
"""

import copy
import unittest

import numpy as np
import itertools as it

from milhoja import (
    LOG_LEVEL_NONE,
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    SCRATCH_ARGUMENT,
    LogicError,
    BasicLogger,
    check_tile_specification
)


class TestCheckTileSpecification(unittest.TestCase):
    def setUp(self):
        self.__singletons = [
            TILE_GRID_INDEX_ARGUMENT,
            TILE_LEVEL_ARGUMENT,
            TILE_DELTAS_ARGUMENT,
            TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
            TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
            TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT
        ]

        self.__logger = BasicLogger(LOG_LEVEL_NONE)

        self.__name = "lo"
        self.__good_single = {"source": TILE_LO_ARGUMENT}
        check_tile_specification(self.__name, self.__good_single, self.__logger)

        self.__good_coords = {
            "source": TILE_COORDINATES_ARGUMENT,
            "axis": "I", "edge": "center",
            "lo": TILE_LO_ARGUMENT, "hi": TILE_UBOUND_ARGUMENT
        }
        check_tile_specification("coords", self.__good_coords, self.__logger)

        self.__good_areas = {
            "source": TILE_FACE_AREAS_ARGUMENT,
            "axis": "I", "lo": TILE_LO_ARGUMENT, "hi": TILE_UBOUND_ARGUMENT
        }
        check_tile_specification("areas", self.__good_areas, self.__logger)

        self.__good_volumes = {
            "source": TILE_CELL_VOLUMES_ARGUMENT,
            "lo": TILE_LO_ARGUMENT, "hi": TILE_UBOUND_ARGUMENT
        }
        check_tile_specification("volumes", self.__good_volumes, self.__logger)

    def testBadSource(self):
        bad_spec = copy.deepcopy(self.__good_single)
        del bad_spec["source"]
        with self.assertRaises(ValueError):
            check_tile_specification(self.__name, bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good_single)
        for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
            bad_spec["source"] = bad
            with self.assertRaises(TypeError):
                check_tile_specification(self.__name, bad_spec, self.__logger)

        bad_spec = copy.deepcopy(self.__good_single)
        bad_spec["source"] = SCRATCH_ARGUMENT
        with self.assertRaises(LogicError):
            check_tile_specification(self.__name, bad_spec, self.__logger)

    def testBadLogger(self):
        for bad in [None, 1, 1.1, "fail", np.nan, np.inf, [], [1], (), (1,)]:
            with self.assertRaises(TypeError):
                check_tile_specification(self.__name, self.__good_single, bad)

    def testSingletonKeys(self):
        for each in self.__singletons:
            self.assertTrue(each.startswith("tile_"))
            name = each.replace("tile_", "")

            good_spec = {"source": each}
            check_tile_specification(name, good_spec, self.__logger)

            # Too few keys
            bad_spec = copy.deepcopy(good_spec)
            del bad_spec["source"]
            self.assertTrue(len(bad_spec) < len(good_spec))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

            # Too many keys
            bad_spec = copy.deepcopy(good_spec)
            bad_spec["fail"] = 1.1
            self.assertTrue(len(bad_spec) > len(good_spec))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

            # Right number of keys, but bad key
            bad_spec = copy.deepcopy(good_spec)
            del bad_spec["source"]
            bad_spec["fail"] = 1.1
            self.assertEqual(len(bad_spec), len(good_spec))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileCoordinatesKeys(self):
        name = "coords"

        # Too few keys
        for key in ["axis", "edge", "lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_coords)
            del bad_spec[key]
            self.assertTrue(len(bad_spec) < len(self.__good_coords))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good_coords)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good_coords))
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Right number of keys, but bad key
        for key in ["axis", "edge", "lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_coords)
            del bad_spec[key]
            bad_spec["fail"] = 1.1
            self.assertEqual(len(bad_spec), len(self.__good_coords))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileCoordinates(self):
        # Confirm all possible combinations accepted
        name = "coords"
        good_axis = ["I", "i", "J", "j", "K", "k"]
        good_edge = ["CeNTeR", "center", "LEFt", "left", "RiGHt", "right"]
        good_lo = [TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]
        good_hi = [TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]
        good_all = list(it.product(good_axis, good_edge, good_lo, good_hi))
        n_cases = len(good_axis) * len(good_edge) * len(good_lo) * len(good_hi)
        self.assertEqual(len(good_all), n_cases)

        for axis, edge, lo, hi in good_all:
            good_spec = {
                "source": TILE_COORDINATES_ARGUMENT,
                "axis": axis, "edge": edge, "lo": lo, "hi": hi
            }
            check_tile_specification(name, good_spec, self.__logger)

    def testTileCoordinatesErrors(self):
        name = "coords"

        # Bad types
        for key in ["axis", "edge", "lo", "hi"]:
            for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
                bad_spec = copy.deepcopy(self.__good_coords)
                bad_spec[key] = bad
                with self.assertRaises(TypeError):
                    check_tile_specification(name, bad_spec, self.__logger)

        # Bad axis value
        bad_spec = copy.deepcopy(self.__good_coords)
        bad_spec["axis"] = "iaxis"
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Bad edge value
        bad_spec = copy.deepcopy(self.__good_coords)
        bad_spec["edge"] = "middle"
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Bad lo value
        bad_spec = copy.deepcopy(self.__good_coords)
        for bad in ["lo", "lbound", TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]:
            bad_spec["lo"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Bad hi value
        bad_spec = copy.deepcopy(self.__good_coords)
        for bad in ["hi", "ubound", TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]:
            bad_spec["hi"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileFaceAreasKeys(self):
        name = "areas"

        # Too few keys
        for key in ["axis", "lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_areas)
            del bad_spec[key]
            self.assertTrue(len(bad_spec) < len(self.__good_areas))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good_areas)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good_areas))
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Right number of keys, but bad key
        for key in ["axis", "lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_areas)
            del bad_spec[key]
            bad_spec["fail"] = 1.1
            self.assertEqual(len(bad_spec), len(self.__good_areas))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileFaceAreas(self):
        name = "areas"

        # Confirm all possible combinations accepted
        good_axis = ["I", "i", "J", "j", "K", "k"]
        good_lo = [TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]
        good_hi = [TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]
        good_all = list(it.product(good_axis, good_lo, good_hi))
        n_cases = len(good_axis) * len(good_lo) * len(good_hi)
        self.assertEqual(len(good_all), n_cases)

        for axis, lo, hi in good_all:
            good_spec = {
                "source": TILE_FACE_AREAS_ARGUMENT,
                "axis": axis, "lo": lo, "hi": hi
            }
            check_tile_specification(name, good_spec, self.__logger)

    def testTileFaceAreasErrors(self):
        name = "areas"

        # Bad types
        for key in ["axis", "lo", "hi"]:
            for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
                bad_spec = copy.deepcopy(self.__good_areas)
                bad_spec[key] = bad
                with self.assertRaises(TypeError):
                    check_tile_specification(name, bad_spec, self.__logger)

        # Bad axis value
        bad_spec = copy.deepcopy(self.__good_areas)
        bad_spec["axis"] = "iaxis"
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Bad lo value
        bad_spec = copy.deepcopy(self.__good_areas)
        for bad in ["lo", "lbound", TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]:
            bad_spec["lo"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Bad hi value
        bad_spec = copy.deepcopy(self.__good_areas)
        for bad in ["hi", "ubound", TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]:
            bad_spec["hi"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileCellVolumesKeys(self):
        name = "volumes"

        # Too few keys
        for key in ["lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_volumes)
            del bad_spec[key]
            self.assertTrue(len(bad_spec) < len(self.__good_volumes))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Too many keys
        bad_spec = copy.deepcopy(self.__good_volumes)
        bad_spec["fail"] = 1.1
        self.assertTrue(len(bad_spec) > len(self.__good_volumes))
        with self.assertRaises(ValueError):
            check_tile_specification(name, bad_spec, self.__logger)

        # Right number of keys, but bad key
        for key in ["lo", "hi"]:
            bad_spec = copy.deepcopy(self.__good_volumes)
            del bad_spec[key]
            bad_spec["fail"] = 1.1
            self.assertEqual(len(bad_spec), len(self.__good_volumes))
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

    def testTileCellVolumes(self):
        name = "volumes"

        # Confirm all possible correct combinations accepted
        good_lo = [TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]
        good_hi = [TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]
        good_all = list(it.product(good_lo, good_hi))
        n_cases = len(good_lo) * len(good_hi)
        self.assertEqual(len(good_all), n_cases)

        for lo, hi in good_all:
            good_spec = {
                "source": TILE_CELL_VOLUMES_ARGUMENT,
                "lo": lo, "hi": hi
            }
            check_tile_specification(name, good_spec, self.__logger)

    def testTileCellVolumesErrors(self):
        name = "volumes"

        # Bad types
        for key in ["lo", "hi"]:
            for bad in [None, 1, 1.1, np.nan, np.inf, [], [1], (), (1,)]:
                bad_spec = copy.deepcopy(self.__good_volumes)
                bad_spec[key] = bad
                with self.assertRaises(TypeError):
                    check_tile_specification(name, bad_spec, self.__logger)

        # Bad lo value
        bad_spec = copy.deepcopy(self.__good_volumes)
        for bad in ["lo", "lbound", TILE_HI_ARGUMENT, TILE_UBOUND_ARGUMENT]:
            bad_spec["lo"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)

        # Bad hi value
        bad_spec = copy.deepcopy(self.__good_volumes)
        for bad in ["hi", "ubound", TILE_LO_ARGUMENT, TILE_LBOUND_ARGUMENT]:
            bad_spec["hi"] = bad
            with self.assertRaises(ValueError):
                check_tile_specification(name, bad_spec, self.__logger)
