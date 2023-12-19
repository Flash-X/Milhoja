import os
import milhoja.tests

from pathlib import Path
from collections import OrderedDict
from collections import defaultdict

from milhoja import (
    LOG_LEVEL_NONE,
    LogicError,
    DataPacketGenerator,
    BasicLogger,
    TaskFunction
)
from milhoja.Cpp2CLayerGenerator import Cpp2CLayerGenerator
from milhoja.C2FortranLayerGenerator import C2FortranLayerGenerator
from milhoja.FortranTemplateUtility import FortranTemplateUtility
from milhoja.TemplateUtility import TemplateUtility
from milhoja.DataPacketC2FModuleGenerator import DataPacketC2FModuleGenerator

_FILE_PATH = Path(__file__).resolve().parent
_DATA_PATH = _FILE_PATH.joinpath("data")
_SEDOV_PATH = _DATA_PATH.joinpath("Sedov")
_RUNTIME_PATH = _DATA_PATH.joinpath("runtime")
_FLASHX_PATH = _DATA_PATH.joinpath("FlashX")


class TestDataPacketGenerator(milhoja.tests.TestCodeGenerators):
    """
    Unit test of DataPacketGenerator class.
    """
    # keys for test dictionaries.
    JSON = "json"
    HEADER = "header"
    HDD = "header_dim_dependent"
    SOURCE = "source",
    SDD = "source_dim_dependent"
    SIZES = "sizes"
    FOLDER = "folder"

    SUMMIT_SIZES_2D = {
        "FArray1D": 24,
        "FArray2D": 24,
        "FArray3D": 40,
        "FArray4D": 48,
        "IntVect": 8,
        "RealVect": 16,
        "byte_align": 16,
        "int": 4,
        "real": 8,
        "std::size_t": 8,
        "unsigned int": 4,
        "bool": 1
    }

    SUMMIT_SIZES_3D = {
        "FArray1D": 24,
        "FArray2D": 24,
        "FArray3D": 40,
        "FArray4D": 48,
        "IntVect": 12,
        "RealVect": 24,
        "byte_align": 16,
        "int": 4,
        "real": 8,
        "std::size_t": 8,
        "unsigned int": 4
    }

    def setUp(self):
        self._runtime = [
            {
                self.JSON: _RUNTIME_PATH.joinpath("gpu_tf_dens.json"),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                # temp use summit sizes
                self.SIZES: self.SUMMIT_SIZES_2D
            },
            {
                self.JSON: _RUNTIME_PATH.joinpath("gpu_tf_ener.json"),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_2D
            },
            {
                self.JSON: _RUNTIME_PATH.joinpath("gpu_tf_fused_actions.json"),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_2D
            },
            {
                self.JSON: _RUNTIME_PATH.joinpath("gpu_tf_fused_kernels.json"),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_2D
            }
        ]

        self._sedov = [
            {
                self.JSON: _SEDOV_PATH.joinpath("REF_gpu_tf_hydro_2D.json"),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_2D
            },
            {
                self.JSON: _SEDOV_PATH.joinpath("REF_gpu_tf_hydro_3D.json"),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_3D
            }
        ]

        self._flashx = [
            {
                self.JSON: _FLASHX_PATH.joinpath(
                    "REF_gpu_tf_hydro_Wesley_2D.json"
                ),
                self.FOLDER: "FlashX",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_2D
            },
            {
                self.JSON: _FLASHX_PATH.joinpath(
                    "REF_gpu_tf_hydro_Wesley_3D.json"
                ),
                self.FOLDER: "FlashX",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: self.SUMMIT_SIZES_3D
            }
        ]

    def tearDown(self):
        pass

    def check_generated_files(self, generated, correct):
        """
        Checks the generated file by comparing it to correct
        with no whitespace.

        TODO: Find a better way to remove whitespaces and compare.
        """
        try:
            generated_string = generated.read().replace(' ', '') \
                                    .replace('\n', '').replace('\t', '')
            correct_string = correct.read().replace(' ', '') \
                                    .replace('\n', '').replace('\t', '')

            self.assertTrue(
                len(generated_string) == len(correct_string),
                f"Generated length: {len(generated_string)}, "
                f"correct length: {len(correct_string)}"
            )
            self.assertTrue(
                generated_string == correct_string,
                f"Comparison between {generated.name}"
                f"and {correct.name} returned false."
            )
        except IOError:
            raise RuntimeError(f"{generated} could not be read.")

    def testPacketGeneration(self):
        # Runs through all tests in test_set.
        # This function generates all necessary files for a complete
        # data packet and compares them with existing reference benchmarks.
        for test_set in [self._runtime, self._sedov, self._flashx]:
            for test in test_set:
                json_path = test[self.JSON]
                sizes = test[self.SIZES]
                self.assertTrue(isinstance(sizes, dict))
                tf_spec = TaskFunction.from_milhoja_json(json_path)
                # use default logging value for now
                logger = BasicLogger(LOG_LEVEL_NONE)
                destination = Path.cwd()
                generator = DataPacketGenerator(tf_spec, 4, logger, sizes)

                # check for no template generation
                with self.assertRaises(
                    RuntimeError,
                    msg="Templates were not generated before the code."
                ):
                    generator.generate_header_code(
                        destination, overwrite=True
                    )

                # check for no template generation
                with self.assertRaises(
                    RuntimeError,
                    msg="Templates were not generated before the code."
                ):
                    generator.generate_source_code(
                        destination, overwrite=True
                    )

                # generate source code
                generator.generate_templates(destination, overwrite=True)
                generator.generate_header_code(destination, overwrite=True)
                generator.generate_source_code(destination, overwrite=True)

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote templates, overwrite==False!!!!"
                ):
                    generator.generate_templates(destination, overwrite=False)

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote header, overwrite==False!!!!"
                ):
                    generator.generate_header_code(
                        destination, overwrite=False
                    )

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote source, overwrite==False!!!!"
                ):
                    generator.generate_source_code(
                        destination, overwrite=False
                    )

                generated_name_cpp = Path(
                    destination,
                    generator.source_filename
                )
                correct_name_cpp = json_path.stem.replace("REF_", "")
                correct_name_cpp = Path(
                    _DATA_PATH,
                    test[self.FOLDER],
                    "REF_DataPacket_" + str(correct_name_cpp) + ".cpp"
                )

                # check c++ source code
                with open(generated_name_cpp, 'r') as generated_cpp:
                    with open(correct_name_cpp, 'r') as correct:
                        # Test generated files.
                        self.check_generated_files(generated_cpp, correct)

                generated_name_h = Path(
                    destination,
                    generator.header_filename
                )
                correct_name_h = json_path.stem.replace("REF_", "")
                correct_name_h = Path(
                    _DATA_PATH,
                    test[self.FOLDER],
                    "REF_DataPacket_" + str(correct_name_h) + ".h"
                )

                # check c++ headers
                with open(generated_name_h, 'r') as generated_h:
                    with open(correct_name_h, 'r') as correct:
                        self.check_generated_files(generated_h, correct)

                # ..todo::
                #   * Generator should generate TaskFunction

                # ..todo::
                #       * currently the cpp2c layer is only generated when
                #         using a fortran task function there should be
                #         another "cpp2c layer" that's just for cpp task
                #         functions.
                generated_cpp2c = None
                generated_c2f = None
                generated_dp_mod = None
                if generator.language == "fortran":
                    # check cpp2c layer.
                    generated_cpp2c = Path(
                        destination,
                        generator.cpp2c_file_name
                    )
                    correct_cpp2c = json_path.stem.replace("REF_", "")
                    correct_cpp2c = Path(
                        _DATA_PATH,
                        test[self.FOLDER],
                        "REF_" + str(correct_cpp2c) + "_Cpp2C.cpp"
                    )
                    with open(generated_cpp2c, 'r') as generated:
                        with open(correct_cpp2c, 'r') as correct:
                            self.check_generated_files(generated, correct)

                    # check c2f layer.
                    generated_c2f = Path(
                        destination,
                        generator.c2f_file_name
                    )
                    correct_c2f = json_path.stem.replace("REF_", "")
                    correct_c2f = Path(
                        _DATA_PATH,
                        test[self.FOLDER],
                        "REF_" + str(correct_c2f) + "_C2F.F90"
                    )
                    with open(generated_c2f, 'r') as generated:
                        with open(correct_c2f, 'r') as correct:
                            self.check_generated_files(generated, correct)

                    # check module file
                    generated_dp_mod = Path(
                        destination,
                        generator.module_file_name
                    )
                    correct_dp_mod = json_path.stem.replace("REF_", "")
                    correct_dp_mod = Path(
                        _DATA_PATH,
                        test[self.FOLDER],
                        "REF_DataPacket_" + str(correct_dp_mod) + "_c2f_mod.F90"
                    )
                    with open(generated_dp_mod, 'r') as generated:
                        with open(correct_dp_mod, 'r') as correct:
                            self.check_generated_files(generated, correct)

                # clean up generated files if test passes.
                try:
                    os.remove(generated_name_cpp)
                    os.remove(generated_name_h)
                    if generated_cpp2c:
                        os.remove(generated_cpp2c)
                    if generated_c2f:
                        os.remove(generated_c2f)
                    if generated_dp_mod:
                        os.remove(generated_dp_mod)
                    # be careful when cleaning up here
                    for file in Path(destination).glob("cg-tpl.*.cpp"):
                        os.remove(file)
                except FileNotFoundError:
                    print("Could not find files. Continue.")

    def testTemplateUtility(self):
        """
        Function for testing the template utility classes.
        """
        connectors = {}
        size_connectors = {}

        # give a sample external variable set.
        mock_external = OrderedDict({
            "external_example": {
                "source": "external",
                "type": "int",
                "extents": "(1,5)"
            }
        })
        with self.assertRaises(
            NotImplementedError,
            msg="External with extents was used without error."
        ):
            TemplateUtility._common_iterate_externals(
                connectors, mock_external, ["external_example"]
            )

        # sample tile_in variable set.
        mock_tile_in = OrderedDict({
            "test1": {
                "source": "tile_in",
                "extents": ['16', '15', '14', '1'],
                "variables_in": [1, 5]
            }
        })
        with self.assertRaises(
            NotImplementedError,
            msg="No test case for fortran tile_in."
        ):
            FortranTemplateUtility.iterate_tile_in(
                connectors, size_connectors, mock_tile_in
            )

        mock_tile_out = OrderedDict({
            "test2": {
                "source": "tile_out",
                "extents": ['16', '15', '14', '1'],
                "variables_out": [1, 5]
            }
        })
        with self.assertRaises(
            NotImplementedError,
            msg="No test cases for fortran tile_out."
        ):
            FortranTemplateUtility.iterate_tile_out(
                connectors, size_connectors, mock_tile_out
            )

        # test proper ordering
        mock_externals = OrderedDict({
            "dt": {
                "source": "external",
                "type": "real",
                "extents": "()"
            },
            "spark_coef": {
                "source": "external",
                "type": "int",
                "extents": "()"
            },
            "spark_coef2": {
                "source": "external",
                "type": "real",
                "extents": "()"
            },
            "another_coef": {
                "source": "external",
                "type": "int",
                "extents": "()"
            }
        })

        connectors = defaultdict(list)
        TemplateUtility._common_iterate_externals(
            connectors, mock_externals,
            ["dt", "another_coef", "spark_coef2", "spark_coef"]
        )

        assert TemplateUtility._CON_ARGS in connectors
        assert TemplateUtility._HOST_MEMBERS in connectors
        constructor_args = [
            'real dt',
            'int another_coef',
            'real spark_coef2',
            'int spark_coef'
        ]
        host_args = [
            "_dt_h",
            "_another_coef_h",
            "_spark_coef2_h",
            "_spark_coef_h"
        ]

        self.assertEqual(
            connectors[TemplateUtility._CON_ARGS], constructor_args
        )
        self.assertEqual(
            connectors[TemplateUtility._HOST_MEMBERS], host_args
        )

    def testCpp2CGenerator(self):
        for test in self._sedov:
            json_path = test[self.JSON]
            sizes = test[self.SIZES]
            self.assertTrue(isinstance(sizes, dict))
            tf_spec = TaskFunction.from_milhoja_json(json_path)

            # only testing fortran files with cpp2c for now.
            # Some files in the Sedov test use C++ so this should be here.
            if tf_spec.language.lower() == "c++":
                continue

            # use default logging value for now
            logger = BasicLogger(LOG_LEVEL_NONE)
            destination = Path.cwd()

            datapacket_generator = DataPacketGenerator(
                tf_spec, 4, logger, sizes
            )

            outer = datapacket_generator.cpp2c_outer_template_name
            helper = datapacket_generator.cpp2c_helper_template_name

            cpp2c = Cpp2CLayerGenerator(
                tf_spec, outer, helper,
                4, LOG_LEVEL_NONE, datapacket_generator.n_extra_streams,
                datapacket_generator.external_args
            )

            with self.assertRaises(
                LogicError,
                msg="Cpp2c generate_header was not called?"
            ):
                cpp2c.generate_header_code(destination, False)

            cpp2c.generate_source_code(destination, overwrite=True)
            with self.assertRaises(
                FileExistsError,
                msg="File was overwritten!"
            ):
                cpp2c.generate_source_code(destination, overwrite=False)

            # cleanup
            try:
                for file in Path(destination).glob("cg-tpl.*.cpp"):
                    os.remove(file)
            except FileNotFoundError:
                print("Could not find files. Continue.")

    def testModGeneration(self):
        for test in self._sedov:
            json_path = test[self.JSON]
            tf_spec = TaskFunction.from_milhoja_json(json_path)
            logger = BasicLogger(LOG_LEVEL_NONE)
            destination = Path.cwd()

            # todo::
            #   * Need a smaller sample reference file for checking
            #     all types of arguments (bool, int, real, arrays)
            sample_externals = {
                "dt": {
                    "source": "external",
                    "type": "real"
                }
            }

            if tf_spec.language.lower() == "c++":
                with self.assertRaises(LogicError, msg="Wrong language"):
                    mod_generator = DataPacketC2FModuleGenerator(
                        tf_spec, 4, logger, sample_externals
                    )
                continue

            mod_generator = DataPacketC2FModuleGenerator(
                tf_spec, 4, logger, sample_externals
            )

            with self.assertRaises(LogicError, msg="Header gen should fail."):
                mod_generator.generate_header_code(destination, True)

            mod_generator.generate_source_code(destination, True)

            with self.assertRaises(
                FileExistsError,
                msg="File was overwritten!"
            ):
                mod_generator.generate_source_code(destination, False)
            os.remove(Path(destination, mod_generator.source_filename))

    def testC2fGenerator(self):
        for test in self._sedov:
            json_path = test[self.JSON]
            sizes = test[self.SIZES]
            self.assertTrue(isinstance(sizes, dict))
            tf_spec = TaskFunction.from_milhoja_json(json_path)

            # only testing fortran TFs.
            if tf_spec.language.lower() == "c++":
                continue

            # use default logging value for now
            logger = BasicLogger(LOG_LEVEL_NONE)
            destination = Path.cwd()

            datapacket_generator = DataPacketGenerator(
                tf_spec, 4, logger, sizes
            )

            int_scratch = {
                "auxC": {
                    "source": "scratch",
                    "type": "int",
                    "extents": ['1', '1', '1'],
                    "lbound": ["(tile_lo)"]
                }
            }

            c2f = C2FortranLayerGenerator(
                tf_spec, 4, logger,
                datapacket_generator.n_extra_streams,
                datapacket_generator.external_args,
                datapacket_generator.tile_metadata_args,
                datapacket_generator.tile_in_args,
                datapacket_generator.tile_in_out_args,
                datapacket_generator.tile_out_args,
                int_scratch
            )

            with self.assertRaises(
                LogicError,
                msg="C2F generate_header was not called?"
            ):
                c2f.generate_header_code(destination, overwrite=True)

            with self.assertRaises(
                NotImplementedError,
                msg="Int scratch did not raise error."
            ):
                c2f.generate_source_code(destination, overwrite=True)

            with self.assertRaises(
                FileExistsError,
                msg="C2F file was overwritten!"
            ):
                c2f.generate_source_code(destination, overwrite=False)

            try:
                for file in Path(destination).glob("*.F90"):
                    os.remove(file)
            except FileNotFoundError:
                print("Could not find files. Continue.")

    def testGetArraySizes(self):
        # test none on both
        mask_in = None
        mask_out = None
        with self.assertRaises(TypeError):
            TemplateUtility.get_array_size([], [])

        mask_in = [1, 2]
        mask_out = [1, 2]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 2)

        mask_in = [2, 8]
        mask_out = [2, 2]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 8)

        with self.assertRaises(
            NotImplementedError,
            msg="No test cases for out size > in size, but no error thrown."
        ):
            mask_in = [1, 2]
            mask_out = [1, 6]
            TemplateUtility.get_array_size(mask_in, mask_out)

        mask_in = [1, 10]
        mask_out = []
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 10)

        mask_in = []
        mask_out = [1, 10]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 10)
