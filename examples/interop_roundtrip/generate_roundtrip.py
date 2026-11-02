#!/usr/bin/env python3
"""Generate the Milhoja interoperability round trip."""

import argparse
import json

from pathlib import Path

import milhoja


LAYOUTS = ("variable-last", "variable-first")
EXECUTIONS = ("hostmem", "cuda-openacc")


def specification(layout, execution):
    use_openacc = execution == "cuda-openacc"
    return {
        "format": ["Milhoja-JSON", "1.0.0"],
        "grid": {
            "dimension": 2,
            "nxb": 3,
            "nyb": 2,
            "nzb": 1,
            "nguardcells": 1,
        },
        "task_function": {
            "language": "Fortran",
            "processor": "GPU" if use_openacc else "CPU",
            "computation_offloading": "OpenACC" if use_openacc else "",
            "variable_index_base": 1,
            "cpp_header": "roundtrip_tf.h",
            "cpp_source": "roundtrip_tf.cxx",
            "c2f_source": "roundtrip_tf_C2F.F90",
            "fortran_source": "roundtrip_tf_mod.F90",
            "name": "roundtrip_tf",
            "argument_list": [
                "external_scale",
                "tile_lbound",
                "CC_1",
                "scratch_checksum",
            ],
            "argument_specifications": {
                "external_scale": {
                    "type": "real",
                    "extents": "()",
                    "source": "external",
                },
                "tile_lbound": {"source": "tile_lbound"},
                "CC_1": {
                    "source": "grid_data",
                    "structure_index": ["CENTER", 1],
                    "variables_in": [1, 4],
                    "variables_out": [1, 4],
                },
                "scratch_checksum": {
                    "type": "real",
                    "extents": "(1)",
                    "lbound": "(1)",
                    "source": "scratch",
                },
            },
            "subroutine_call_graph": ["roundtrip_kernel"],
        },
        "data_item": {
            "type": "DataPacket",
            "byte_alignment": 16,
            "header": "DataPacket_roundtrip_tf.h",
            "module": "DataPacket_roundtrip_tf_mod.F90",
            "source": "DataPacket_roundtrip_tf.cxx",
            "data_layout": layout,
        },
        "subroutines": {
            "roundtrip_kernel": {
                "interface_file": "roundtrip_kernel_mod.F90",
                "argument_list": [
                    "tile_lbound", "field", "scale", "checksum"
                ],
                "argument_mapping": {
                    "tile_lbound": "tile_lbound",
                    "field": "CC_1",
                    "scale": "external_scale",
                    "checksum": "scratch_checksum",
                },
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milhoja_prefix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument(
        "--execution", choices=EXECUTIONS, default="hostmem"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "roundtrip.taskfunction.json"
    if spec_path.exists() and not args.overwrite:
        raise FileExistsError("Output already exists: {!s}".format(spec_path))
    spec_path.write_text(
        json.dumps(specification(args.layout, args.execution), indent=2) + "\n",
        encoding="utf-8",
    )

    task_function = milhoja.TaskFunction.from_milhoja_json(spec_path)
    logger = milhoja.BasicLogger(milhoja.LOG_LEVEL_NONE)
    milhoja.generate_data_item(
        task_function, output, args.overwrite, args.milhoja_prefix, 4, logger
    )
    milhoja.generate_task_function(
        task_function, output, args.overwrite, 4, logger
    )
    print("Generated {} {} round trip in {}".format(
        args.execution, args.layout, output
    ))


if __name__ == "__main__":
    main()
