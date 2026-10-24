"""Runtime-compiled CUDA C++ kernels: NVRTC and the CUDA driver API through
ctypes, with no build step and no extra Python dependency.

Kernels that Triton cannot express (data-dependent shared-memory gathers, warp
shuffles) are compiled from source strings by the NVRTC library that ships
with PyTorch's CUDA wheels (``nvidia/cu13/lib`` or ``nvidia/cuda_nvrtc/lib``),
to a cubin for the device's architecture, cached on disk, loaded with
``cuModuleLoadData`` and launched with ``cuLaunchKernel`` on torch's current
stream, so CUDA-graph capture records them like any other kernel. Compile and
load before a capture (the warmup does): module loading is not a stream
operation.

Environment: ``PYJUICE_NVRTC_LIB`` (path of ``libnvrtc.so``), ``PYJUICE_NVRTC_CACHE``
(cubin cache directory, default ``~/.cache/pyjuice/nvrtc``; empty disables it).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import hashlib
import os
import threading
from typing import Dict, Optional, Sequence, Tuple

import torch

_LOCK = threading.Lock()
_NVRTC: Optional[ctypes.CDLL] = None
_CUDA: Optional[ctypes.CDLL] = None

_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

class TensorMap(ctypes.Structure):
    """A ``CUtensorMap`` (TMA descriptor, 128 bytes), passed by value to a
    ``const __grid_constant__`` kernel parameter (``"tmap"``)."""
    _pack_ = 64
    _fields_ = [("opaque", ctypes.c_uint64 * 16)]


_ARG_TYPES = {
    "ptr": ctypes.c_uint64,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "f32": ctypes.c_float,
    "tmap": TensorMap,
}

_TMAP_DTYPES = {torch.float32: 7, torch.int32: 3}
_TMAP_SWIZZLE = {0: 0, 32: 1, 64: 2, 128: 3}


def tensor_map_2d(t: torch.Tensor, offset: int, rows: int, cols: int, row_stride: int,
                  box_rows: int, box_cols: int, swizzle: int = 0) -> TensorMap:
    """TMA descriptor of the row-major 2D view ``[rows, cols]`` (row stride
    ``row_stride`` elements) of ``t`` starting at element ``offset``, loaded
    in boxes of ``[box_rows, box_cols]`` with the given smem swizzle (bytes:
    0 / 32 / 64 / 128). The base address must be 16-byte aligned and the row
    stride a multiple of 16 bytes; rows past ``rows`` read as zeros."""
    cuda = _driver()
    tm = TensorMap()
    es = t.element_size()
    addr = t.data_ptr() + offset * es
    assert addr % 16 == 0 and (row_stride * es) % 16 == 0, "TMA needs 16-byte aligned rows"
    dims = (ctypes.c_uint64 * 2)(cols, rows)
    strides = (ctypes.c_uint64 * 1)(row_stride * es)
    box = (ctypes.c_uint32 * 2)(box_cols, box_rows)
    estr = (ctypes.c_uint32 * 2)(1, 1)
    # interleave none, L2 promotion 256 B, no OOB NaN fill
    _cu_check(cuda.cuTensorMapEncodeTiled(ctypes.byref(tm), _TMAP_DTYPES[t.dtype], 2, ctypes.c_void_p(addr),
                                          dims, strides, box, estr, 0, _TMAP_SWIZZLE[swizzle], 3, 0),
              "cuTensorMapEncodeTiled")
    return tm


def _nvrtc_candidates():
    env = os.environ.get("PYJUICE_NVRTC_LIB")
    if env:
        yield env
    major = (torch.version.cuda or "").split(".")[0]
    try:
        import nvidia
        roots = list(nvidia.__path__)
    except ImportError:
        roots = []
    for root in roots:
        # CUDA 13 wheels: nvidia/cu13/lib, CUDA 12 wheels: nvidia/cuda_nvrtc/lib
        for pat in (f"cu{major}/lib/libnvrtc.so.{major}*", f"cuda_nvrtc/lib/libnvrtc.so.{major}*",
                    "*/lib/libnvrtc.so*"):
            for path in sorted(glob.glob(os.path.join(root, pat))):
                if "builtins" not in path and ".alt." not in path:
                    yield path
    found = ctypes.util.find_library("nvrtc")
    if found:
        yield found
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if home:
            yield from sorted(glob.glob(os.path.join(home, "lib64", "libnvrtc.so*")))


def _nvrtc() -> ctypes.CDLL:
    global _NVRTC
    with _LOCK:
        if _NVRTC is None:
            errors = []
            for path in _nvrtc_candidates():
                try:
                    _NVRTC = ctypes.CDLL(path)
                    break
                except OSError as e:
                    errors.append(f"{path}: {e}")
            if _NVRTC is None:
                raise RuntimeError("pyjuice.utils.nvrtc: no loadable libnvrtc found (set "
                                   "PYJUICE_NVRTC_LIB); tried " + "; ".join(errors or ["nothing"]))
            _NVRTC.nvrtcGetErrorString.restype = ctypes.c_char_p
    return _NVRTC


def _driver() -> ctypes.CDLL:
    global _CUDA
    with _LOCK:
        if _CUDA is None:
            _CUDA = ctypes.CDLL("libcuda.so.1")
            _CUDA.cuInit(0)
    return _CUDA


def _nvrtc_check(nvrtc, status, prog=None):
    if status == 0:
        return
    msg = nvrtc.nvrtcGetErrorString(status).decode()
    if prog is not None:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        msg += "\n" + log.value.decode(errors="replace")
    raise RuntimeError(f"NVRTC: {msg}")


def _cu_check(status, what):
    if status == 0:
        return
    cuda = _driver()
    s = ctypes.c_char_p()
    cuda.cuGetErrorString(status, ctypes.byref(s))
    raise RuntimeError(f"CUDA driver: {what} failed ({status}: {s.value.decode() if s.value else '?'})")


def nvrtc_version() -> Tuple[int, int]:
    nvrtc = _nvrtc()
    major, minor = ctypes.c_int(), ctypes.c_int()
    _nvrtc_check(nvrtc, nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
    return major.value, minor.value


def ptxas_info(source: str, name: str, arch: str = "sm_90", options: Sequence[str] = ()) -> str:
    """The compiler's resource report (registers, spills, shared memory) of
    ``source``, uncached; needs no GPU."""
    nvrtc = _nvrtc()
    opts = [f"--gpu-architecture={arch}", "-std=c++17", "--ptxas-options=-v", *options]
    prog = ctypes.c_void_p()
    _nvrtc_check(nvrtc, nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(),
                                                 f"{name}.cu".encode(), 0, None, None))
    try:
        c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        _nvrtc_check(nvrtc, nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts), prog)
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        return log.value.decode(errors="replace")
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def compile_cubin(source: str, name: str, arch: str, options: Sequence[str] = ()) -> bytes:
    """Compile ``source`` to a cubin for ``arch`` (``"sm_90"``), through the
    on-disk cache."""
    nvrtc = _nvrtc()
    opts = [f"--gpu-architecture={arch}", "-std=c++17", *options]
    key = hashlib.sha256("\0".join([source, name, *opts, repr(nvrtc_version())]).encode()).hexdigest()
    cache = os.environ.get("PYJUICE_NVRTC_CACHE", os.path.expanduser("~/.cache/pyjuice/nvrtc"))
    path = os.path.join(cache, f"{key}.cubin") if cache else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    prog = ctypes.c_void_p()
    _nvrtc_check(nvrtc, nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(),
                                                 f"{name}.cu".encode(), 0, None, None))
    try:
        c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        _nvrtc_check(nvrtc, nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts), prog)
        size = ctypes.c_size_t()
        _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
        buf = ctypes.create_string_buffer(size.value)
        _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBIN(prog, buf))
        cubin = buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    if path:
        try:
            os.makedirs(cache, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(cubin)
            os.replace(tmp, path)
        except OSError:
            pass
    return cubin


class CudaKernel:
    """One ``extern "C" __global__`` function of a CUDA C++ source string.

    ``arg_types`` lists the kernel's parameters (``"ptr"`` for a tensor or a
    device address, ``"i32"`` / ``"u32"`` / ``"i64"`` / ``"f32"``). The cubin
    is built and loaded per device on first use; ``max_shared`` raises the
    dynamic shared-memory limit of the function (bytes, above 48 KB).
    """

    def __init__(self, source: str, name: str, arg_types: Sequence[str],
                 options: Sequence[str] = (), max_shared: int = 0):
        self.source, self.name = source, name
        self.options = tuple(options)
        self.max_shared = max_shared
        self._types = [_ARG_TYPES[t] for t in arg_types]
        self._fns: Dict[int, Tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}

    def _function(self, device: torch.device) -> Tuple[ctypes.c_void_p, ctypes.c_void_p]:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        fn = self._fns.get(idx)
        if fn is not None:
            return fn
        cuda = _driver()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"CudaKernel {self.name}: first use during a CUDA-graph capture; "
                               "run the kernel once before capturing (module loading is not "
                               "capturable)")
        major, minor = torch.cuda.get_device_capability(idx)
        cubin = compile_cubin(self.source, self.name, f"sm_{major}{minor}", self.options)
        dev = ctypes.c_int()
        _cu_check(cuda.cuDeviceGet(ctypes.byref(dev), idx), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _cu_check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        _cu_check(cuda.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")
        try:
            mod = ctypes.c_void_p()
            _cu_check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
            func = ctypes.c_void_p()
            _cu_check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, self.name.encode()),
                      f"cuModuleGetFunction({self.name})")
            if self.max_shared > 48 * 1024:
                _cu_check(cuda.cuFuncSetAttribute(func, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                  self.max_shared), "cuFuncSetAttribute")
        finally:
            _cu_check(cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())), "cuCtxPopCurrent")
        self._fns[idx] = (func, ctx)
        return func, ctx

    def __call__(self, grid: Sequence[int], block: Sequence[int], args: Sequence,
                 shared_mem: int = 0, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
            device = device or torch.device("cuda", torch.cuda.current_device())
        func, ctx = self._function(device)
        cuda = _driver()
        assert len(args) == len(self._types), f"{self.name}: {len(args)} args, expected {len(self._types)}"
        vals = [a if isinstance(a, TensorMap) else t(a.data_ptr() if isinstance(a, torch.Tensor) else a)
                for t, a in zip(self._types, args)]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        stream = torch.cuda.current_stream(device).cuda_stream
        cur = ctypes.c_void_p()
        cuda.cuCtxGetCurrent(ctypes.byref(cur))
        if cur.value != ctx.value:
            _cu_check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        _cu_check(cuda.cuLaunchKernel(func, g[0], g[1], g[2], b[0], b[1], b[2], shared_mem,
                                      ctypes.c_void_p(stream), params, None),
                  f"cuLaunchKernel({self.name})")
