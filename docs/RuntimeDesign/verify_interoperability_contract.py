#!/usr/bin/env python3
"""Verify source facts recorded by the Step 5A interoperability contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return ROOT.joinpath(relative).read_text(encoding="utf-8")


def _require(text, tokens, source):
    for token in tokens:
        if token not in text:
            raise AssertionError("{} omits {!r}".format(source, token))


def main():
    header_tool = _read("tools/write_library_header.py")
    _require(
        header_tool,
        [
            "_VALID_RUNTIME = ['none', 'cuda', 'hostmem']",
            "_VALID_OFFLOAD = ['none', 'openacc']",
            "MILHOJA_CUDA_RUNTIME_BACKEND",
            "MILHOJA_HOSTMEM_RUNTIME_BACKEND",
            "MILHOJA_OPENACC_OFFLOADING",
        ],
        "write_library_header.py",
    )

    configure = _read("configure.py")
    _require(
        configure,
        [
            "RUNTIME_BACKEND = {runtime_backend}",
            "COMPUTATION_OFFLOADING = {computation_offloading}",
        ],
        "configure.py",
    )

    runtime = _read("src/Milhoja_RuntimeBackend.cpp")
    _require(
        runtime,
        [
            "#ifdef MILHOJA_CUDA_RUNTIME_BACKEND",
            "#elif defined(MILHOJA_HOSTMEM_RUNTIME_BACKEND)",
            "static CudaBackend singleton",
            "static FakeCudaBackend singleton",
        ],
        "Milhoja_RuntimeBackend.cpp",
    )

    hostmem = _read("src/FakeCudaBackend/Milhoja_FakeCudaBackend.cpp")
    _require(
        hostmem,
        [
            "memcpy(packet.copyToGpuStart_gpu()",
            "memcpy(packet.returnToHostStart_host()",
            "callback(callbackData);",
        ],
        "Milhoja_FakeCudaBackend.cpp",
    )

    cuda = _read("src/CudaBackend/Milhoja_CudaBackend.cu")
    _require(
        cuda,
        [
            "cudaMemcpyAsync(packet.copyToGpuStart_gpu()",
            "cudaMemcpyAsync(packet.returnToHostStart_host()",
            "packet.stream()",
            "cudaLaunchHostFunc(stream, callback, callbackData)",
        ],
        "Milhoja_CudaBackend.cu",
    )

    packet = _read("src/Milhoja_DataPacket.cpp")
    _require(
        packet,
        [
            "RuntimeBackend::instance().releaseStream(stream_)",
            "RuntimeBackend::instance().releaseGpuMemory(&packet_p_, &packet_d_)",
            "tiles_.push_front( std::move(tileDesc) )",
        ],
        "Milhoja_DataPacket.cpp",
    )

    packet_template = _read(
        "tools/milhoja_pypkg/src/milhoja/templates/cg-tpl.datapacket.cpp"
    )
    _require(
        packet_template,
        [
            "RuntimeBackend::instance().requestGpuMemory",
            "RuntimeBackend::instance().requestStream(true)",
            "RuntimeBackend::instance().releaseStream(stream_)",
        ],
        "cg-tpl.datapacket.cpp",
    )

    dispatch = _read("tools/milhoja_pypkg/src/milhoja/generate_task_function.py")
    _require(
        dispatch,
        [
            '(processor == "cpu")',
            '(data_item == "datapacket")',
            '(offloading == "openacc")',
            'logger.warn("Milhoja TF", "No TF generation for use with DataPackets")',
        ],
        "generate_task_function.py",
    )

    cpp2c = _read(
        "tools/milhoja_pypkg/src/milhoja/"
        "TaskFunctionCpp2CGenerator_OpenACC_F.py"
    )
    _require(
        cpp2c,
        [
            '"MILHOJA_OPENACC_OFFLOADING"',
            '"MILHOJA_HOSTMEM_RUNTIME_BACKEND"',
            "packet_h->asynchronousQueue()",
            "if self._use_openacc else '0'",
        ],
        "TaskFunctionCpp2CGenerator_OpenACC_F.py",
    )

    c2f = _read(
        "tools/milhoja_pypkg/src/milhoja/"
        "TaskFunctionC2FGenerator_OpenACC_F.py"
    )
    _require(
        c2f,
        [
            "bind(c, name=",
            "C_PTR, C_F_POINTER",
            "MILHOJA_OPENACC_OFFLOADING",
            "MILHOJA_HOSTMEM_RUNTIME_BACKEND",
        ],
        "TaskFunctionC2FGenerator_OpenACC_F.py",
    )

    cuda_streams = _read(
        "src/CudaBackend/Milhoja_CudaStreamManager.cu"
    )
    _require(
        cuda_streams,
        ["acc_get_cuda_stream(stream.accAsyncQueue)"],
        "Milhoja_CudaStreamManager.cu",
    )

    constants = _read("tools/milhoja_pypkg/src/milhoja/constants.py")
    _require(
        constants,
        [
            'VARIABLE_LAST = "variable-last"',
            'VARIABLE_FIRST = "variable-first"',
            "SUPPORTED_DATA_LAYOUTS = [VARIABLE_LAST, VARIABLE_FIRST]",
        ],
        "constants.py",
    )

    print("Milhoja Step 5A interoperability source audit passed")


if __name__ == "__main__":
    main()
