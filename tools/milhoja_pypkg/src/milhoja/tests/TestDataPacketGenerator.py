import os
import milhoja.tests
import glob

from pathlib import Path
from collections import OrderedDict

from milhoja import LOG_LEVEL_NONE
from milhoja.DataPacketGenerator import DataPacketGenerator
from milhoja.TaskFunction import TaskFunction
from milhoja.BasicLogger import BasicLogger
from milhoja.Cpp2CLayerGenerator import Cpp2CLayerGenerator
from milhoja.C2FortranLayerGenerator import C2FortranLayerGenerator
from milhoja.FortranTemplateUtility import FortranTemplateUtility
from milhoja.TemplateUtility import TemplateUtility

_FILE_PATH = Path(__file__).resolve().parent
# temporary
_TEST_PATH = _FILE_PATH.joinpath("data")


class TestDataPacketGenerator(milhoja.tests.TestCodeGenerators):
    """
    Unit test of DataPacketGenerator class.

    ..todo::
        * move lbound parser tests into package test.
        * create tests for extents parser.
    """
    # keys for test dictionaries.
    JSON = "json"
    HEADER = "header"
    HDD = "header_dim_dependent"
    SOURCE = "source",
    SDD = "source_dim_dependent"
    SIZES = "sizes"
    FOLDER = "folder"

    def setUp(self):
        # load task function spec
        # TODO: Once this is in milhoja package change path
        self._runtime = [
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_dens.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                # temp use summit sizes
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_fused_actions.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "runtime",
                    "gpu_tf_fused_kernels.json"
                ),
                self.FOLDER: "runtime",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            }
        ]

        self._sedov = [
            {
                self.JSON: _TEST_PATH.joinpath(
                    "Sedov",
                    "gpu_tf_hydro_2D.json"
                ),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
            },
            {
                self.JSON: _TEST_PATH.joinpath(
                    "Sedov",
                    "gpu_tf_hydro_3D.json"
                ),
                self.FOLDER: "Sedov",
                self.HDD: False,
                self.SDD: False,
                self.SIZES: {
                    "real": 8,
                    "int": 4,
                    "unsigned int": 4,
                    "std::size_t": 8,
                    "IntVect": 8,
                    "RealVect": 16,
                    "bool": 1
                }
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

    def testPacketGeneration(self):
        # Tests all files in the test data folder.
        for test_set in [self._runtime, self._sedov]:
            for test in test_set:
                json_path = test[self.JSON]
                sizes = test[self.SIZES]
                self.assertTrue(isinstance(sizes, dict))
                tf_spec = TaskFunction.from_milhoja_json(json_path)
                # use default logging value for now
                logger = BasicLogger(LOG_LEVEL_NONE)
                destination = "./"

                generator = DataPacketGenerator(tf_spec, 4, logger, sizes)

                # check for no template generation
                with self.assertRaises(
                    RuntimeError,
                    msg="Templates were not generated before the code."
                ):
                    generator.generate_header_code(
                        destination, overwrite=True
                    )
                    self.assertTrue(False)

                # check for no template generation
                with self.assertRaises(
                    RuntimeError,
                    msg="Templates were not generated before the code."
                ):
                    generator.generate_source_code(
                        destination, overwrite=True
                    )
                    self.assertTrue(False)

                generator.generate_templates(destination, overwrite=True)
                generator.generate_header_code(destination, overwrite=True)
                generator.generate_source_code(destination, overwrite=True)

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote templates, overwrite==False!!!!"
                ):
                    generator.generate_templates(destination, overwrite=False)
                    self.assertTrue(False)

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote header, overwrite==False!!!!"
                ):
                    generator.generate_header_code(
                        destination, overwrite=False
                    )
                    self.assertTrue(False)

                with self.assertRaises(
                    FileExistsError,
                    msg="Generator overwrote source, overwrite==False!!!!"
                ):
                    generator.generate_source_code(
                        destination, overwrite=False
                    )
                    self.assertTrue(False)

                generated_name_cpp = Path(
                    destination,
                    generator.source_filename
                )
                correct_name_cpp = Path(
                    _TEST_PATH,
                    test[self.FOLDER],
                    "REF_" + os.path.basename(generator.source_filename)
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
                correct_name_h = Path(
                    _TEST_PATH,
                    test[self.FOLDER],
                    "REF_" + os.path.basename(generator.header_filename)
                )

                # check c++ headers
                with open(generated_name_h, 'r') as generated_h:
                    with open(correct_name_h, 'r') as correct:
                        self.check_generated_files(generated_h, correct)

                # ..todo::
                #   * Generator should generate TaskFunction call

                # ..todo::
                #       * currently the cpp2c layer is only generated when
                #         using a fortran task function there should be
                #         another "cpp2c layer" that's just for cpp task
                #         functions.
                generated_cpp2c = None
                generated_c2f = None
                if generator.language == "fortran":
                    generated_cpp2c = Path(
                        destination,
                        generator.cpp2c_file
                    )
                    correct_cpp2c = Path(
                        _TEST_PATH,
                        test[self.FOLDER],
                        "REF_" + os.path.basename(generator.cpp2c_file)
                    )
                    with open(generated_cpp2c, 'r') as generated:
                        with open(correct_cpp2c, 'r') as correct:
                            self.check_generated_files(generated, correct)

                    generated_c2f = Path(
                        destination,
                        generator.c2f_file
                    )
                    correct_c2f = Path(
                        _TEST_PATH,
                        test[self.FOLDER],
                        "REF_" + os.path.basename(generator.c2f_file)
                    )
                    with open(generated_c2f, 'r') as generated:
                        with open(correct_c2f, 'r') as correct:
                            self.check_generated_files(generated, correct)

                # clean up generated files if test passes.
                try:
                    os.remove(generated_name_cpp)
                    os.remove(generated_name_h)
                    if generated_cpp2c:
                        os.remove(generated_cpp2c)
                    if generated_c2f:
                        os.remove(generated_c2f)
                    # be careful when cleaning up here
                    for file in glob.glob(
                        str(Path(destination, "cg-tpl.*.cpp"))
                    ):
                        os.remove(file)
                except FileNotFoundError:
                    print("Could not find files. Continue.")

    def testTemplateUtility(self):
        connectors = {}
        size_connectors = {}

        mock_external = OrderedDict({
            "external_example": {
                "source": "external",
                "type": "int",
                "extents": ['1', '5']
            }
        })
        with self.assertRaises(
            NotImplementedError,
            msg="External with extents was used without error."
        ):
            TemplateUtility._common_iterate_externals(
                connectors, size_connectors, mock_external
            )

        mock_tile_in = OrderedDict({
            "test1": {
                "source": "tile_in",
                "extents": ['16', '16', '16', '1'],
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
                "extents": ['16', '16', '16', '1'],
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

    def testCpp2CGenerator(self):
        for test in self._sedov:
            json_path = test[self.JSON]
            sizes = test[self.SIZES]
            self.assertTrue(isinstance(sizes, dict))
            tf_spec = TaskFunction.from_milhoja_json(json_path)
            # use default logging value for now
            logger = BasicLogger(LOG_LEVEL_NONE)
            destination = "./"

            datapacket_generator = DataPacketGenerator(
                tf_spec, 4, logger, sizes
            )

            cpp2c = Cpp2CLayerGenerator(
                tf_spec, datapacket_generator.cpp2c_outer_template,
                datapacket_generator.cpp2c_helper_template,
                4, LOG_LEVEL_NONE, datapacket_generator.n_extra_streams,
                datapacket_generator.external_args
            )

            with self.assertRaises(
                NotImplementedError,
                msg="Cpp2c generate_header was not called?"
            ):
                cpp2c.generate_header_code(destination, False)
                self.assertTrue(False)

            cpp2c.generate_source_code(destination, overwrite=True)
            with self.assertRaises(
                FileExistsError,
                msg="File was overwritten!"
            ):
                cpp2c.generate_source_code(destination, overwrite=False)
                self.assertTrue(False)

            try:
                for file in glob.glob(
                    str(Path(destination, "cg-tpl.*.cpp"))
                ):
                    os.remove(file)
            except FileNotFoundError:
                print("Could not find files. Continue.")

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
            destination = "./"

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
                NotImplementedError,
                msg="C2F generate_header was not called?"
            ):
                c2f.generate_header_code(destination, overwrite=True)
                self.assertTrue(False)

            with self.assertRaises(
                NotImplementedError,
                msg="Int scratch did not raise error."
            ):
                c2f.generate_source_code(destination, overwrite=True)
                self.assertTrue(False)

            with self.assertRaises(
                FileExistsError,
                msg="C2F file was overwritten!"
            ):
                c2f.generate_source_code(destination, overwrite=False)
                self.assertTrue(False)

            try:
                for file in glob.glob(
                    str(Path(destination, "cg-tpl.*.cpp"))
                ):
                    os.remove(file)
            except FileNotFoundError:
                print("Could not find files. Continue.")
