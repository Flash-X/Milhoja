#!/usr/bin/env python3
"""Verify deterministic HOSTMEM and CUDA/OpenACC source generation."""

import argparse
import filecmp
import json
import subprocess
import sys
import tempfile

from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_roundtrip.py"
LAYOUTS = ("variable-last", "variable-first")
EXECUTIONS = ("hostmem", "cuda-openacc")
EXPECTED = {
    "DataPacket_roundtrip_tf.cxx",
    "DataPacket_roundtrip_tf.h",
    "DataPacket_roundtrip_tf_mod.F90",
    "cg-tpl.helper_DataPacket_roundtrip_tf.cpp",
    "cg-tpl.helper_DataPacket_roundtrip_tf_cpp2c.cpp",
    "cg-tpl.outer_DataPacket_roundtrip_tf.cpp",
    "cg-tpl.outer_DataPacket_roundtrip_tf_cpp2c.cpp",
    "roundtrip.taskfunction.json",
    "roundtrip_tf.cxx",
    "roundtrip_tf_C2F.F90",
    "roundtrip_tf_mod.F90",
}


def generate(prefix, output, execution, layout):
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            str(prefix),
            str(output),
            "--layout",
            layout,
            "--execution",
            execution,
        ],
        check=True,
    )


def compare(first, second):
    comparison = filecmp.dircmp(first, second)
    assert not comparison.left_only
    assert not comparison.right_only
    assert not comparison.funny_files
    for name in comparison.common_files:
        assert filecmp.cmp(first / name, second / name, shallow=False), name


def verify_sources(output, execution, layout):
    assert {path.name for path in output.iterdir() if path.is_file()} == EXPECTED
    spec = json.loads((output / "roundtrip.taskfunction.json").read_text())
    assert spec["grid"] == {
        "dimension": 2,
        "nxb": 3,
        "nyb": 2,
        "nzb": 1,
        "nguardcells": 1,
    }
    assert spec["data_item"]["data_layout"] == layout
    assert spec["data_item"]["type"] == "DataPacket"
    if execution == "cuda-openacc":
        assert spec["task_function"]["processor"] == "GPU"
        assert spec["task_function"]["computation_offloading"] == "OpenACC"
    else:
        assert spec["task_function"]["processor"] == "CPU"
        assert spec["task_function"]["computation_offloading"] == ""
    assert spec["task_function"]["subroutine_call_graph"] == [
        "roundtrip_kernel"
    ]
    field = spec["task_function"]["argument_specifications"]["CC_1"]
    assert field == {
        "source": "grid_data",
        "structure_index": ["CENTER", 1],
        "variables_in": [1, 4],
        "variables_out": [1, 4],
    }
    checksum = spec["task_function"]["argument_specifications"][
        "scratch_checksum"
    ]
    assert checksum == {
        "type": "real",
        "extents": "(1)",
        "lbound": "(1)",
        "source": "scratch",
    }

    c2f = (output / "roundtrip_tf_C2F.F90").read_text()
    packet = (output / "DataPacket_roundtrip_tf.cxx").read_text()
    wrapper = (output / "roundtrip_tf.cxx").read_text()
    fortran = (output / "roundtrip_tf_mod.F90").read_text()
    combined = (c2f + packet + wrapper).lower()
    if execution == "cuda-openacc":
        assert "!$acc parallel loop gang" in fortran.lower()
        assert "use openacc" in combined
        assert "milhoja_openacc_offloading" in combined
        assert "asynchronousqueue()" in wrapper.lower()
        assert "milhoja_hostmem_runtime_backend" not in combined
    else:
        assert "!$acc" not in combined
        assert "use openacc" not in combined
        assert "milhoja_openacc_offloading" not in combined
        assert "milhoja_hostmem_runtime_backend" in combined
    assert "CALL roundtrip_kernel" in fortran
    assert "intent(INOUT) :: CC_1_d" in fortran
    assert "shape=[1, F_nTiles_h]" in c2f

    if layout == "variable-first":
        assert (
            "shape=[4, 3 + 2 * 1 * MILHOJA_K1D, "
            "2 + 2 * 1 * MILHOJA_K2D, "
            "1 + 2 * 1 * MILHOJA_K3D, F_nTiles_h]"
        ) in c2f
        assert "CC_1_p[variable + nVars_CC_1 * cell]" in packet
        assert "CC_1_data_h[cell + nCells_CC_1 * variable]" in packet
    else:
        assert (
            "shape=[3 + 2 * 1 * MILHOJA_K1D, "
            "2 + 2 * 1 * MILHOJA_K2D, "
            "1 + 2 * 1 * MILHOJA_K3D, 4, F_nTiles_h]"
        ) in c2f
        assert "std::memcpy" in packet
        assert "nCells_CC_1" not in packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milhoja_prefix", type=Path)
    args = parser.parse_args()

    for execution in EXECUTIONS:
        for layout in LAYOUTS:
            with tempfile.TemporaryDirectory() as first_name, \
                    tempfile.TemporaryDirectory() as second_name:
                first = Path(first_name)
                second = Path(second_name)
                generate(args.milhoja_prefix, first, execution, layout)
                generate(args.milhoja_prefix, second, execution, layout)
                verify_sources(first, execution, layout)
                compare(first, second)

    print("Milhoja Step 5E HOSTMEM/CUDA-OpenACC generation verification passed")


if __name__ == "__main__":
    main()
