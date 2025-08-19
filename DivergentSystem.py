#!/usr/bin/env python3
#
# Note to self: had to pip upgrade typing_extensions from 4.9.0 to 4.14.1

from abc import ABC
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numbers import Number
from pycuda.compiler import SourceModule
from threading import Thread
from typing import List, Tuple, Union, Optional, Any
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pycuda.driver as drv
import pyvista as pv
import shutil
import time
import vtk


@dataclass(frozen=True)
class DivergentPlotHandle:
    fig: Figure
    ax: Any          # generic: 2D/3D
    line: Any        # e.g. Line2D or Line3D

    def close(self) -> None:
        """Close the figure. Safe to call multiple times."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception:
                # Be quiet during interpreter shutdown or unusual backends.
                pass

    def __del__(self) -> None:
        # Best-effort finaliser so `del handle` eventually closes the window.
        if self.fig:
            try:
                self.close()
            except Exception:
                pass


class DivergentSystem(ABC):
    # Dark theme
    _fg_col   = "#eaecef"
    _bg_col   = "#0a0f1c"
    _line_col = "#84c968"
    _fill_col = (0.25, 0.35, 0.65, 0.15)
    _grid_col = (0.7, 0.7, 0.7, 0.5)

    @staticmethod
    def _is_scalar(x):
        # Accepts ints/floats (incl. numpy numeric types), rejects booleans
        return isinstance(x, Number) and not isinstance(x, bool)

    @classmethod
    def _normalise_sequence(cls, name: str, value, expected_len: int) -> tuple:
        """
        Convert `value` to a tuple of length `expected_len`.
        - If fewer elements are supplied, pad with zeros.
        - If more are supplied, raise ValueError.
        - Elements must be numeric scalars.
        Accepts: None, a single scalar, or any non-string Iterable of scalars.
        """

        if value is None:
            seq = []
        elif cls._is_scalar(value):
            seq = [value]
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            seq = list(value)
        else:
            raise TypeError(f"{name} must be a numeric scalar or an iterable of numeric scalars.")

        if len(seq) > expected_len:
            raise ValueError(f"{name} accepts at most {expected_len} value(s); got {len(seq)}.")

        # pad with zeros as required
        seq += [0] * (expected_len - len(seq))

        # validate element types
        if not all(cls._is_scalar(x) for x in seq):
            raise TypeError(f"All elements of {name} must be numeric scalars.")

        return tuple(seq)

    def __init__(self, system_kernel: str = None, distance_kernel: str = None):
        # Subclasses override this attribute with the textual names of the control variable(s)
        # for this system
        #
        # (e.g. ['sigma', 'rho', 'beta'] for a 3-DOF Lorenz system)
        #
        self._variables = []

        # Tuple of values to assign to each of our control variables
        self._config = None

        # 3D tuple containing the starting coordinates of the state space
        self._base = None

        # 3D tuple containing the end coordinates of the state space
        self._limit = None

        # 3D tuple containing the stride size through the state space
        self._stride = None

        # Number of iteration steps to use when computing a heatmap or trajectory
        self._steps = None

        # Information about any CUDA-compatible GPUs that are present
        self._base_dev, self._base_cc, self._devices = DivergentSystem._check_CUDA()

        # Directory of this Python file
        self._script_dir = os.path.dirname(os.path.abspath(__file__))

        # Cache directory: <script_dir>/CUDA_cache
        self._cache_dir = os.path.join(self._script_dir, "CUDA_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

        # Build directory used with nvcc --keep-dir
        self._build_dir = os.path.join(self._cache_dir, "build")
        os.makedirs(self._build_dir, exist_ok=True)

        # Subclasses initialise these dict entries with CUDA kernel sources...
        self._kernel_sources = {
            "system_kernel": system_kernel,
            "distance_kernel": distance_kernel,
        }

        # PTX cache in memory: kernel_name -> dev_index -> bytes
        self._kernel_ptx: dict[str, dict[int, bytes]] = {
            "system_kernel": {},
            "distance_kernel": {},
        }

        # Track which kernels have been compiled for which devices
        self._compiled: set[tuple[str, int]] = set()

        # Track if we have a trajectory plot spawned from the heatmap plot
        self._plot = None

    @property
    def variables(self) -> List:
        """Textual names of the control variables for this system."""
        return self._variables

    @property
    def config(self) -> tuple:
        """Tuple of control variable values, length matches number of variables."""
        return self._config

    @config.setter
    def config(self, value):
        if not hasattr(self, "_variables") or self._variables is None:
            raise RuntimeError("Variables not initialised in subclass before setting config.")
        expected = len(self._variables)
        if expected < 0:
            raise RuntimeError("Invalid variable count.")
        self._config = self._normalise_sequence("config", value, expected)

    @property
    def base(self) -> tuple:
        """3D starting coordinates of the state space."""
        return self._base

    @base.setter
    def base(self, value):
        self._base = self._normalise_sequence("base", value, 3)

    @property
    def limit(self) -> tuple:
        """3D end coordinates of the state space."""
        return self._limit

    @limit.setter
    def limit(self, value):
        self._limit = self._normalise_sequence("limit", value, 3)

    @property
    def stride(self) -> tuple:
        """3D stride through the state space."""
        return self._stride

    @stride.setter
    def stride(self, value):
        self._stride = self._normalise_sequence("stride", value, 3)

    @property
    def system_kernel(self) -> str:
        """CUDA kernel source for the system algorithm returning a trajectory."""
        return self._kernel_sources["system_kernel"]

    @property
    def distance_kernel(self) -> str:
        """CUDA kernel source for the system algorithm returning a distance metric."""
        return self._kernel_sources["distance_kernel"]

    def calc_stride(self, divisions: Union[int, Tuple[int, int, int], List[int]]) -> Tuple[float, float, float]:
        """
        Calculate and set the stride values for the state space.

        The stride for each dimension is computed as:
            stride[i] = (limit[i] - base[i]) / divisions[i]

        Args:
            divisions (int | tuple[int, int, int] | list[int]):
                Either a single positive integer, which is expanded to all
                dimensions, or a sequence of exactly three positive integers
                specifying the subdivisions for each dimension (x, y, z).

        Returns:
            tuple[float, float, float]: The calculated stride for each dimension.

        Raises:
            TypeError: If `divisions` is not an int, tuple of three ints,
                or list of three ints.
            ValueError: If any subdivision count is <= 0, or if a stride
                evaluates to zero.
            RuntimeError: If `base` or `limit` has not been set before calling
                this method.
        """
        # Normalise divisions into a tuple of 3 positive integers
        if isinstance(divisions, int):
            divs = (divisions,) * 3
        elif isinstance(divisions, (tuple, list)) and len(divisions) == 3:
            if not all(isinstance(d, int) for d in divisions):
                raise TypeError("All subdivision counts must be integers.")
            divs = tuple(divisions)
        else:
            raise TypeError("divisions must be an int or a tuple/list of 3 ints.")

        if any(d <= 0 for d in divs):
            raise ValueError("Subdivision counts must be > 0.")

        # Ensure base and limit are set
        if not self.base or not self.limit:
            raise RuntimeError("base and limit properties must be set before calculating stride.")

        # Calculate strides
        strides = []
        for i in range(3):
            span = self.limit[i] - self.base[i]
            stride_val = span / divs[i]
            # If the span is zero, the stride will be zero
            strides.append(stride_val)

        self._stride = tuple(strides)
        return self._stride

    @staticmethod
    def _check_CUDA() -> Tuple[drv.Device, List[int]]:
        """
        Identify all CUDA-capable GPUs with matching compute capability.

        Returns:
            tuple: (base_device, list_of_device_indices)
        """
        drv.init()
        n_devices = drv.Device.count()
        if n_devices == 0:
            return None, []

        base_dev = drv.Device(0)
        base_cc = base_dev.compute_capability()

        devices = []
        for i in range(n_devices):
            dev = drv.Device(i)
            if dev.compute_capability() == base_cc:
                devices.append(i)
            else:
                print(f"Skipping Device {i}: {dev.name()} CC {dev.compute_capability()}")

        return base_dev, base_cc, devices

    def _cc_tag_for_device(self, dev_index: int) -> str:
        """
        Return the compute capability tag (e.g. 'sm75') for a CUDA device.

        Args:
            dev_index: Index of the target device.
        """

        maj, min_ = drv.Device(dev_index).compute_capability()
        return f"sm{maj}{min_}"

    def _stable_ptx_path(self, kernel_name: str, dev_index: int, src: str) -> str:
        """
        Build a stable PTX cache path for a kernel on a specific device.

        Args:
            - kernel_name: Kernel identifier.
            - dev_index: CUDA device index.
            - src: Kernel source code.

        Returns:
            Path to the PTX file in the cache directory.
        """

        cc = self._cc_tag_for_device(dev_index)
        h  = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self._cache_dir, f"{kernel_name}-{cc}-{h}.ptx")

    def _make_unique_build_subdir(self, kernel_name: str) -> str:
        """
        Create and return a unique build subdirectory for a kernel.

        Args:
            kernel_name: Kernel identifier.

        Returns:
            Path to the created build subdirectory.
        """

        pid = os.getpid()
        try:
            tid = str(threading.get_ident())  # optional; if threading wasn’t imported, fallback
        except NameError:
            tid = "main"
        ts  = time.time_ns()
        d   = os.path.join(self._build_dir, f"{kernel_name}-{pid}-{tid}-{ts}")
        os.makedirs(d, exist_ok=True)
        return d

    def _wait_for_ptx(self, keep_dir: str, timeout_s: float = 5.0, poll_ms: int = 25) -> str:
        """
        Wait briefly for nvcc to materialise a .ptx in keep_dir. Returns the path.
        Raises on timeout.
        """

        preferred = os.path.join(keep_dir, "kernel.ptx")
        deadline  = time.time() + timeout_s
        while time.time() < deadline:
            # Prefer kernel.ptx, else first *.ptx found
            if os.path.exists(preferred):
                return preferred
            for name in os.listdir(keep_dir):
                if name.lower().endswith(".ptx"):
                    return os.path.join(keep_dir, name)
            time.sleep(poll_ms / 1000.0)
        raise RuntimeError(f"Timed out waiting for .ptx in {keep_dir}")

    def _clean_dir_tree(self, path: str) -> None:
        """
        Recursively delete a directory tree, ignoring errors.

        Args:
            path: Path to the directory to remove.
        """

        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _ensure_kernel_compiled_for_device(self, kernel_name: str, dev_index: int) -> drv.Function:
        """
        Ensure PTX exists for (kernel_name, dev_index), move it to a stable path in CUDA_cache,
        then load it into the *current* context and return its Function.

        Caller must have already made a context current for dev_index.
        """

        # Validate current context matches the device
        cur = drv.Context.get_current()
        if cur is None:
            raise RuntimeError("No active CUDA context; create a context before loading the kernel.")
        if drv.Context.get_device() != drv.Device(dev_index):
            raise RuntimeError("Active context device does not match requested dev_index.")

        src = self._kernel_sources.get(kernel_name)
        if not src:
            raise RuntimeError(f"{kernel_name} source not provided")

        stable_ptx = self._stable_ptx_path(kernel_name, dev_index, src)

        # Build PTX if not present
        if not os.path.exists(stable_ptx):
            keep_dir = self._make_unique_build_subdir(kernel_name)
            try:
                # Compile in the current context; nvcc leaves intermediates in keep_dir
                _ = SourceModule(
                    src,
                    options=["--keep", "--source-in-ptx", f"--keep-dir={keep_dir}"],
                    cache_dir=self._cache_dir,
                )
                # Wait for PTX to appear and move to stable filename
                ptx_path = self._wait_for_ptx(keep_dir)
                os.makedirs(self._cache_dir, exist_ok=True)
                tmp_target = stable_ptx + ".tmp"
                shutil.copyfile(ptx_path, tmp_target)
                os.replace(tmp_target, stable_ptx)
            finally:
                self._clean_dir_tree(keep_dir)

        # Load PTX into the current context
        mod = drv.module_from_file(stable_ptx)
        return mod.get_function(kernel_name)

    def _sanitise_volume(
            self,
            base: Tuple = None,
            limit: Tuple = None,
            stride: Tuple = None,
            config: Tuple = None,
        ) -> Tuple:
        """
        Ensure volume parameters (base, limit, stride, config) are set, filling
        from instance defaults if needed.

        Args:
            - base: Volume base coordinates.
            - limit: Volume limit coordinates.
            - stride: Step sizes.
            - config: Configuration tuple.

        Returns:
            tuple: (base, limit, stride, config)

        Raises:
            RuntimeError: If config is unset or any volume parameter is missing.
        """

        if base is None:
            base = self.base
        if limit is None:
            limit = self.limit
        if stride is None:
            stride = self.stride
        if config is None:
            config = self.config

        if config is None:
            raise RuntimeError("Configuration has not been set.")
        if base is None or limit is None or stride is None:
            raise RuntimeError("Volume information is mandatory (base, limit, stride).")

        return base, limit, stride, config

    def get_trajectory(
            self,
            origin: Tuple = (0.0, 0.0, 0.0),
            steps: int = 500
        ) -> np.ndarray:
        """
        Run a single instance of the system kernel to generate a trajectory.

        Public API unchanged. The kernel is compiled once per device and cached.

        Args:
            - origin: Starting position (state) as (x, y, z) coordinates.
            - steps: Number of iterations to run the system for.

        Returns:
            ndarray: Array of shape (steps, 3) containing [x, y, z] coordinates at each step.

        Raises:
            RuntimeError: If the system kernel has not been defined or no CUDA device is available.
            ValueError: If inputs are malformed.
        """

        # Basic validation and normalisation
        self._steps = steps
        config = self.config if self.config else []
        if self.system_kernel is None:
            raise RuntimeError("system_kernel not defined.")
        if not self._devices:
            raise RuntimeError("No compatible CUDA devices found.")

        if len(origin) != 3:
            raise ValueError(f"'origin' must be a 3-tuple, got length {len(origin)}.")
        if steps < 0:
            raise ValueError("'steps' must be non-negative.")
        if steps == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Optional arity check against subclass-declared variables list
        expected = getattr(self, "_variables", None)
        if expected is not None and len(config) != len(expected):
            raise ValueError(f"Expected {len(expected)} config variables {tuple(expected)}, got {len(config)}.")

        # Choose a device to run on. By convention use the first compatible device.
        dev_index = self._devices[0]

        # Create a short-lived context for the launch
        ctx = drv.Device(dev_index).make_context()
        try:
            # Load a Function for this context from cached PTX
            func = self._ensure_kernel_compiled_for_device("system_kernel", dev_index)

            # Allocate output buffer
            trajectory = np.empty((steps, 3), dtype=np.float32)

            # Prepare arguments
            x0, y0, z0 = map(np.float32, origin)
            vars_f32 = [np.float32(v) for v in config]
            steps_i = np.int32(steps)

            # Launch kernel
            func(
                drv.Out(trajectory),
                x0, y0, z0,
                *vars_f32,
                steps_i,
                block=(1, 1, 1),
                grid=(1, 1)
            )

            return trajectory
        finally:
            ctx.pop()
            ctx.detach()

    def _axis_from_bounds(self, start: float, stop: float, step: float) -> tuple[float, int]:
        """
        Build one lattice axis as (effective_step, count).
        Singleton if start == stop. Otherwise step must be non-zero.
        We do not force-include the upper bound.
        """
        if np.isclose(stop, start, rtol=0.0, atol=1e-12):
            return 0.0, 1
        if step == 0.0:
            raise ValueError("Non-singleton dimension requires non-zero stride.")
        span = stop - start
        if span * step < 0.0:
            step = -step
        n = int(np.floor(abs(span) / abs(step))) + 1
        return float(step), int(n)

    def get_heatmap(
            self,
            base: Tuple = None,
            limit: Tuple = None,
            stride: Tuple = None,
            config: Tuple = None,
            perturbation: tuple = None,
            steps: int = 500
        ) -> np.ndarray:
        """
        Compute a chaos heatmap over a grid of *starting coordinates* using all compatible GPUs.

        Conventions used by the distance kernel:
            - Each grid point is an *origin* (initial state) (x0, y0, z0) from which two trajectories
              are launched; the second is offset by `perturbation`.
            - `steps` is the number of integration steps for the distance estimate.

        Args:
            - perturbation: Offset (vector) applied to the second trajectory’s initial state.
            - steps: Integration steps used by the distance kernel.

        Returns:
            ndarray: Chaos metric array reshaped to the grid defined by (base, limit, stride).
                     Singleton dimensions are preserved in the output shape.

        Notes:
            - Supports 1D, 2D, or 3D grids of starting coordinates.
            - Any dimension with base == limit is treated as a single slice, so 1D or 2D heatmaps
              can be produced without artificial extra points.
            - For non-singleton dimensions, the upper bound is included if it lies within half a stride.
            - Work is sharded across available CUDA devices; results are reassembled in the original
              meshgrid ordering.
        """

        base, limit, stride, config = self._sanitise_volume(base, limit, stride, config)
        self._steps = steps

        if config is None:
            raise RuntimeError("Configuration has not been set.")
        if base is None:
            raise RuntimeError("Base coordinates have not been set.")
        if limit is None:
            raise RuntimeError("Limit coordinates have not been set.")
        if stride is None:
            raise RuntimeError("Stride pattern has not been set.")
        if not self._devices:
            raise RuntimeError("No compatible CUDA devices found.")
        if self.distance_kernel is None:
            raise RuntimeError("distance_kernel not defined.")

        # If we don't have a perturbation vector, create it now
        if perturbation is None:
            perturbation = (stride[0] * 1e-9, 0, 0)

        dims = len(base)
        if not (len(limit) == len(stride) == dims):
            raise ValueError("base, limit, and stride must have the same dimensionality.")
        if dims not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, or 3D grids are supported.")

        # Extend to 3 axes for the kernel. Extra axes become singletons.
        b = list(map(float, base))   + [0.0] * (3 - dims)
        L = list(map(float, limit))  + [0.0] * (3 - dims)
        s = list(map(float, stride)) + [0.0] * (3 - dims)

        # Per-axis step and count
        sx, nx = self._axis_from_bounds(b[0], L[0], s[0])
        sy, ny = self._axis_from_bounds(b[1], L[1], s[1])
        sz, nz = self._axis_from_bounds(b[2], L[2], s[2])

        # Convert kernel pParameters into correct format: base coordinate, stride size, count of strides
        x0, y0, z0 = float(b[0]), float(b[1]), float(b[2])
        sx, sy, sz = float(sx), float(sy), float(sz)
        nx, ny, nz = int(nx), int(ny), int(nz)

        total = nx * ny * nz
        if total <= 0:
            empty3 = np.empty((nx, ny, nz), dtype=np.float32)
            return empty3[:, 0, 0] if dims == 1 else (empty3[:, :, 0] if dims == 2 else empty3)

        # Partition [0, total) into near-equal contiguous slices across devices
        num_devices = len(self._devices)
        base_count = total // num_devices
        remainder  = total % num_devices
        counts  = [base_count + (1 if i < remainder else 0) for i in range(num_devices)]
        offsets = np.cumsum([0] + counts[:-1]).tolist()

        # Storage for per-device results. Index aligns with self._devices order.
        results: list[Optional[np.ndarray]] = [None] * num_devices
        threads: list[Thread] = []

        # now = time.time()

        for slot, dev_index, offset, count in zip(range(num_devices), self._devices, offsets, counts):
            t = Thread(
                target=self._heatmap_worker,
                args=(slot, dev_index, config, perturbation, steps,
                      x0, y0, z0, sx, sy, sz, nx, ny, nz,
                      offset, count,
                      results)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # print(f"heatmap GPU took time {time.time() - now}")

        # If any worker failed, its slot will still be None
        if any(r is None for r in results):
            raise RuntimeError("A heatmap worker failed to launch or execute. Check previous CUDA errors.")

        merged = np.concatenate([r if r is not None else np.empty((0,), np.float32) for r in results])
        out3 = merged.reshape((nx, ny, nz)).astype(np.float32, copy=False)

        # Trim back to requested dimensionality
        if dims == 1:
            return out3[:, 0, 0]
        if dims == 2:
            return out3[:, :, 0]
        return out3

    def _heatmap_worker(
            self,
            slot: int,
            dev_index: int,
            variables: List[float],
            perturbation: Tuple[float, float, float],
            steps: int,
            x0: float, y0: float, z0: float,
            sx: float, sy: float, sz: float,
            nx: int, ny: int, nz: int,
            offset: int, count: int,
            results: List[Optional[np.ndarray]]
        ) -> None:
        """
        Compute a contiguous linear window [offset, offset+count) of the lattice on one GPU.

        Implementation notes:
          - Compiles and caches 'distance_kernel' to PTX for this device on first use.
          - Loads the function into the active context immediately before launch.
          - Creates a short-lived context per call. (May later move to a persistent context.)
          - Uses a grid-stride loop in the kernel, so the grid size can stay moderate.
        """

        # Nothing to do for empty slice
        if count <= 0:
            results[slot] = np.empty((0,), dtype=np.float32)
            return

        # Create a short-lived context for the device
        ctx = drv.Device(dev_index).make_context()
        try:
            # Load kernel function into *this* context from cached PTX
            func = self._ensure_kernel_compiled_for_device("distance_kernel", dev_index)

            # Optional arity check
            expected = getattr(self, "_variables", None)
            if expected is not None and len(variables) != len(expected):
                raise ValueError(f"Expected {len(expected)} variables {tuple(expected)}, got {len(variables)}")

            # Prepare scalar arguments
            vars_f32 = [np.float32(v) for v in variables]
            px, py, pz = map(np.float32, perturbation)
            steps_i = np.int32(steps)

            # Output buffer for this slice
            out_buf = np.empty(count, dtype=np.float32)

            # Choose a starting block size and ensure plenty of blocks
            threads_per_block = 256
            sm_count = drv.Device(dev_index).get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
            min_blocks = sm_count * 4
            blocks_needed = (count + threads_per_block - 1) // threads_per_block
            blocks = int(max(min_blocks, blocks_needed))

            if blocks <= 0:
                results[slot] = np.empty((0,), dtype=np.float32)
                return

            # Allocate device output once and use a stream for accurate timing and fewer syncs
            d_out = drv.mem_alloc(out_buf.nbytes)
            stream = drv.Stream()

            # Assemble args in the order the kernel expects
            args = [
                *vars_f32,
                px, py, pz,
                steps_i,
                np.float32(x0), np.float32(y0), np.float32(z0),
                np.float32(sx), np.float32(sy), np.float32(sz),
                np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(offset), np.int32(count),
                d_out,
            ]

            # # Optional timing with events
            # start, end = drv.Event(), drv.Event()
            # start.record(stream)

            func(*args, block=(threads_per_block, 1, 1), grid=(blocks, 1, 1), stream=stream)

            # end.record(stream)
            # end.synchronize()   # kernel complete here

            # Async copy back, then sync once
            drv.memcpy_dtoh_async(out_buf, d_out, stream)
            stream.synchronize()

            results[slot] = out_buf

        finally:
            ctx.pop()
            ctx.detach()

    def _title_parts(
            self,
            base: Tuple,
            limit: Tuple = None,
            config: Tuple = None
        ):
        """
        Build common title components for plot labelling.

        This helper formats:
          - Control parameters (from `config` and `self._variables`) as a parenthesised string, e.g.
            " (rho: 10 sigma: 28 beta: 2.67)".
          - Volume information:
            * If `limit` is None: " at [x, y, z]".
            * Otherwise: " from [x, y, z] to [x', y', z']".

        Args:
            base: Origin coordinates for the plotted data.
            limit: Opposite corner coordinates. If omitted, the output assumes a single point (`base`).
            config: Control variable values corresponding to `self._variables`.

        Returns:
            control: Formatted control parameter string (may be empty).
            volume: Formatted coordinate (range) string.
        """

        control = ""
        if config is not None:
            control = " ".join(f"{name}: {value:.03f}" for name, value in zip(self._variables, config))
            control = f" ({control})"

        if limit is None:
            volume = " at [{}]".format(
                ", ".join(f"{value:.03f}" for value in base)
            )
        else:
            volume = " from [{}] to [{}]".format(
                ", ".join(f"{value:.03f}" for value in base),
                ", ".join(f"{value:.03f}" for value in limit)
            )

        return control, volume

    def _plot_leafname(self, origin: Tuple = None, type: str = "heatmap", path: str = None, ext: str = "png", sig: int = 3, maxlen: int = 180) -> str:
        """
        Build a timestamped heatmap filename from the instance's variables, config,
        base, and limit values, formatted to `sig` significant figures and truncated
        to `maxlen`.

        Args:
            - origin: Starting coordinates if this is a trajectory plot.
            - type: Type of plot (e.g. "heatmap", "trajectory").
            - path: Optional directory for the file; returns only the filename if None.
            - ext: File extension without dot. Defaults to "png".
            - sig: Significant figures for numeric formatting. Defaults to 3.
            - maxlen: Max length of the metadata suffix. Defaults to 180.
        """

        fmt = lambda x: "0" if float(x) == 0 else f"{float(x):.{sig}g}"
        if origin:
            suffix = (
                f"cfg[{','.join(f'{k}={fmt(v)}' for k, v in zip(self.variables, self.config))}]"
                f"at[{','.join(fmt(v) for v in origin)}]"
            )
        else:
            suffix = (
                f"cfg[{','.join(f'{k}={fmt(v)}' for k, v in zip(self.variables, self.config))}]"
                f"base[{','.join(fmt(v) for v in self.base)}]"
                f"limit[{','.join(fmt(v) for v in self.limit)}]"
            )
        suffix = suffix[:maxlen]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        leaf = f"{ts}-{self.__class__.__name__}-{type}-{suffix}.{ext}"

        return leaf if path is None else os.path.join(path, leaf)

    def plot_trajectory(
            self,
            trajectory: np.ndarray,
            origin: Tuple,
            title=None,
            block=False,
            min_span=0.1,
            max_span=None,
            plot: Optional[DivergentPlotHandle] = None,
        ) -> DivergentPlotHandle:
        """
        Plot a 3D trajectory from `trajectory` data, centring and scaling axes
        within `min_span` and `max_span` limits, with optional custom title.

        Args:
            - trajectory: Nx3 array of (x, y, z) coordinates.
            - origin: Reference point used in the title if none is given.
            - title: Plot title. Defaults to a class-based string from config and origin.
            - block: Whether to block execution until the plot window is closed.
            - min_span: Minimum axis span. Defaults to 0.1.
            - max_span: Maximum axis span. If None, no limit is applied.

        Returns:
            - plot handle for the trajectory plot
        """

        def _traj_on_key(event):
            """
            Handle key presses save the plot.

            Args:
                event: Matplotlib key event.

            Notes:
                - Pressing Shift+S saves the plot.
            """

            if event.key in ("shift+s"):
                out = self._plot_leafname(origin=origin, path=self._script_dir, type="trajectory", ext="png")
                fig = event.canvas.figure
                fig.canvas.draw()
                fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
                print(f"Saved {out}")

        # Apply min/max span rules
        def _traj_limit(lo, hi, span):
            if span < min_span:
                c = 0.5 * (lo + hi)
                lo, hi = c - min_span / 2, c + min_span / 2
            elif max_span is not None and span > max_span:
                c = 0.5 * (lo + hi)
                lo, hi = c - max_span / 2, c + max_span / 2
            return lo, hi

        if trajectory is None or len(trajectory) < 2:
            raise RuntimeError("No trajectory data to plot.")

        config = self.config
        if config is None:
            raise RuntimeError("Configuration not initialised")

        if title is None:
            control, volume = self._title_parts(base=origin, config=config)
            title = f"{self.__class__.__name__} Trajectory{control}{volume}"

        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        spans = [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
        max_extent = max(spans)
        buffer = 0.1 * max_extent

        # Can we put this plot into the one with the supplied handle?
        reuse = (
            plot is not None
            and plot.fig is not None
            and plt.fignum_exists(plot.fig.number)
        )

        if reuse:
            # Yes. Reuse...
            print("Reuse plot")
            fig, ax = plot.fig, plot.ax
            line = plot.line if (plot.line is not None and plot.line in ax.lines) else None

            # fig.patch.set_facecolor(DivergentSystem._bg_col)   # figure background (margins)
            # ax.set_facecolor(DivergentSystem._bg_col)          # axes background

            if line is not None:
                # Replace data on the existing line
                line.set_data_3d(x, y, z)
            else:
                # Axes was cleared; recreate the artist on the same axes
                ax.cla()
                (line,) = ax.plot(x, y, z, linewidth=0.5, alpha=0.8, color="magenta")
                ax.set_xlabel("X", color=DivergentSystem._fg_col)
                ax.set_ylabel("Y", color=DivergentSystem._fg_col)
                ax.set_zlabel("Z", color=DivergentSystem._fg_col)
        else:
            # No. New window...
            print("New plot")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            fig.patch.set_facecolor(DivergentSystem._bg_col)   # figure background (margins)
            ax.set_facecolor(DivergentSystem._bg_col)          # axes background

            (line,) = ax.plot(x, y, z, linewidth=0.5, alpha=0.8, color="magenta")
            ax.set_xlabel("X", color=DivergentSystem._fg_col)
            ax.set_ylabel("Y", color=DivergentSystem._fg_col)
            ax.set_zlabel("Z", color=DivergentSystem._fg_col)

            # Create a plot handle object for this plot
            plot = DivergentPlotHandle(fig=fig, ax=ax, line=line)

            fig.canvas.mpl_connect("key_press_event", _traj_on_key)

        # Ensure we're using the right colour
        line.set_color(DivergentSystem._line_col)

        # Update title and limits every call
        ax.set_title(title, color=DivergentSystem._fg_col)
        ax.set_xlim(_traj_limit(x.min() - buffer, x.max() + buffer, spans[0]))
        ax.set_ylim(_traj_limit(y.min() - buffer, y.max() + buffer, spans[1]))
        ax.set_zlim(_traj_limit(z.min() - buffer, z.max() + buffer, spans[2]))
        ax.tick_params(colors=DivergentSystem._fg_col)

        if not reuse:
            # Pane (the filled rectangles behind the data)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color(DivergentSystem._fill_col)
                axis.line.set_color(DivergentSystem._grid_col)

            # Grid line colour and width (per axis)
            ax.grid(True)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["color"] = DivergentSystem._grid_col
                axis._axinfo["grid"]["linewidth"] = 0.2

        plt.tight_layout()
        plt.rcParams['savefig.facecolor'] = DivergentSystem._bg_col

        if block:
            plt.show(block=True)
        else:
            fig.canvas.draw_idle()
            try:
                plt.pause(0.001)  # safe cross-backend nudge
            except Exception:
                pass

        # Return the plot handle for this plot
        return plot

    def _plot_preprocess(self, data: np.ndarray, equal: bool = False):
        """
        Prepare `data` for plotting by identifying finite values and optionally
        histogram-equalising it.

        Args:
            data: Input array.
            equal: If True, normalises value distribution using histogram equalisation.

        Returns:
            tuple: (finite_mask, finite_count, processed_data)
                - finite_mask: Boolean mask of finite values.
                - finite_count: Number of finite entries.
                - processed_data: Equalised data if `equal` is True, else original data.
        """

        def histogram_equalise(data: np.ndarray) -> np.ndarray:
            """
            Histogram-equalise `data` to [0, 1], mapping constant fields to 0.5 and
            non-finite entries to 0.

            Args:
                data: Input array.

            Returns:
                np.ndarray: Equalised array with same shape as input.

            Raises:
                ValueError: If no finite values are present.
            """

            flat = data.ravel()
            finite_mask = np.isfinite(flat)

            if not np.any(finite_mask):
                raise ValueError("Data contains no finite values to equalise.")

            finite_vals = flat[finite_mask]
            lo = float(finite_vals.min())
            hi = float(finite_vals.max())

            # Degenerate case: constant finite field
            if hi == lo:
                out = np.zeros_like(flat, dtype=float)
                out[finite_mask] = 0.5  # map constant field to mid-grey
                # Non-finite entries map to 0
                out[~finite_mask] = 0.0
                return out.reshape(data.shape)

            # Build CDF on finite range only
            hist, bin_edges = np.histogram(finite_vals, bins=256, range=[lo, hi])
            cdf = hist.cumsum()
            denom = float(cdf.max() - cdf.min())
            cdf = (cdf - cdf.min()) / (denom if denom != 0.0 else 1.0)

            # Map finite values via interpolation; set non-finite to 0
            out = np.empty_like(flat, dtype=float)
            out[finite_mask] = np.interp(finite_vals, bin_edges[:-1], cdf)
            out[~finite_mask] = 0.0

            return out.reshape(data.shape)

        finite_mask = np.isfinite(data)
        finite_count = int(finite_mask.sum())
        std = float(np.nanstd(data[finite_mask])) if finite_count > 0 else 0.0
        if equal and std > 1e-12:
            # Equalise the distribution of values (in case they are lopsided) and normalise
            # before equalisation
            data = histogram_equalise(data)

        return finite_mask, finite_count, data

    def plot_heatmap_2D(
            self,
            data: np.ndarray,
            base: Tuple = None,
            limit: Tuple = None,
            stride: Tuple = None,
            config: Tuple = None,
            equal: bool = True,
            theme: str = "inferno",
            block: bool = False,
            save: bool = False
        ) -> DivergentPlotHandle:
        """
        Plot a 2D heatmap view from `data` between `base` and `limit` coordinates.

        This accepts:
          - a true 2D array, or
          - a 3D array with exactly one singleton axis

        The function renders a 2D image for interaction, but keeps a reversible
        “slice map” so that mouse clicks and recomputations (zoom/pan/keys) are
        translated back to the correct 3D world coordinates. No squeezing or
        axis reordering is performed on the caller’s array outside of the local
        2D view extraction used for display.

        Args:
            - data: 2D field (nx, ny) or 3D field (n0, n1, n2) with one axis == 1.
            - base: Lower bounds per axis. Length 2 or 3. If 2D input and only two
                    values are provided, z defaults to 0.
            - limit: Upper bounds per axis. Length 2 or 3.
            - stride: Preferred sample spacings for recompute paths. Optional.
            - config: Variable configuration; may be used by interactive events.
            - equal: If True, histogram-equalises data before plotting.
            - theme: Matplotlib colormap name.
            - block: Whether to block execution until the plot window is closed.
            - save: If True, saves a PNG of the current figure.

        Returns:
            - plot handle for the heatmap plot
        """

        # Remember if we have opened a trajectory plot via clicking on the heatmap
        self._plot = None

        # ---------- helpers: coordinate mapping and slice selection ----------

        SliceMap = namedtuple("SliceMap", "plot_axes fixed_axis fixed_index")

        def _axis_coords(n: int, lo: float, hi: float) -> np.ndarray:
            """Centres per cell along an axis from lo->hi inclusive, length n."""
            n = int(n)
            lo = float(lo)
            hi = float(hi)
            return np.linspace(lo, hi, n)

        def _edges_from_centres(c: np.ndarray) -> np.ndarray:
            """Bin edges for nearest-cell mapping given centre coordinates."""
            if c.size == 1:
                # Degenerate single cell. Give it a tiny width so clicks still map.
                w = 1e-12 if np.isfinite(c[0]) else 1.0
                return np.array([c[0] - w, c[0] + w])
            d = np.diff(c)
            left = c[0] - d[0] / 2.0
            right = c[-1] + d[-1] / 2.0
            mids = c[:-1] + d / 2.0
            return np.concatenate([[left], mids, [right]])

        def _build_slice_map(arr: np.ndarray) -> Tuple[np.ndarray, SliceMap]:
            """
            Returns a 2D view for display and a mapping back to the source axes.

            For 2D input, the view is `arr` and we fabricate a fixed z later.
            For 3D input with one singleton axis, we take that axis at index 0,
            but record which two source axes are shown on X and Y.
            """
            if arr.ndim == 2:
                return arr, SliceMap(plot_axes=(0, 1), fixed_axis=None, fixed_index=0)

            if arr.ndim != 3:
                raise ValueError("Data must be 2D or 3D.")

            singletons = [i for i, s in enumerate(arr.shape) if s == 1]
            if len(singletons) != 1:
                raise ValueError("3D input must have exactly one singleton axis.")

            fixed_axis = singletons[0]
            # View is the two non-singleton axes in their natural (increasing) order
            view2d = np.take(arr, indices=0, axis=fixed_axis)
            plot_axes = tuple(i for i in range(3) if i != fixed_axis)
            return view2d, SliceMap(plot_axes=plot_axes, fixed_axis=fixed_axis, fixed_index=0)

        def _default_stride() -> Tuple[float, float, float]:
            """
            A defensive stride for recomputation if none was provided.
            For a length-1 axis we return 0 spacing.
            """
            if stride is not None:
                # Pad to length 3 if needed
                s = tuple(stride) + (0.0,) * max(0, 3 - len(stride))
                return s[:3]
            dims = 3 if (base is not None and len(base) >= 3) else 2
            s = []
            for i in range(dims):
                n = data.shape[i] if i < data.ndim else 1
                if n > 1:
                    s.append((float(limit[i]) - float(base[i])) / (n - 1))
                else:
                    s.append(0.0)
            if dims == 2:
                s.append(0.0)
            return tuple(s[:3])

        # Build the 2D display view and the mapping back to the source axes
        view2d, smap = _build_slice_map(data)

        # Pre-process the 2D view (equalisation etc.) for display only
        finite_mask, finite_count, view2d = self._plot_preprocess(view2d, equal)

        # Per-axis world coordinates from base->limit. We only build those we need.
        # For 2D input we still allow optional base/limit[2] but do not require it.
        base = tuple(base) if base is not None else (0.0, 0.0) + ((0.0,) if data.ndim == 3 else ())
        limit = tuple(limit) if limit is not None else tuple(float(n - 1) for n in data.shape) + \
            ((0.0,) if data.ndim == 2 else ())
        stride3 = _default_stride()

        # Defensive padding of base/limit to length 3 for recompute paths
        if len(base) < 3:
            base = base + (0.0,) * (3 - len(base))
        if len(limit) < 3:
            limit = limit + (0.0,) * (3 - len(limit))

        # World coordinates on the two plotted axes
        ax_x, ax_y = smap.plot_axes
        nx, ny = view2d.shape
        xs = _axis_coords(n=data.shape[ax_x] if data.ndim == 3 else nx, lo=base[ax_x], hi=limit[ax_x])
        ys = _axis_coords(n=data.shape[ax_y] if data.ndim == 3 else ny, lo=base[ax_y], hi=limit[ax_y])

        # Ensure display increases left-to-right and bottom-to-top by flipping the view if needed
        if xs[-1] < xs[0]:
            xs = xs[::-1]
            view2d = view2d[::-1, :]
        if ys[-1] < ys[0]:
            ys = ys[::-1]
            view2d = view2d[:, ::-1]

        # Bin edges for stable mapping from world coords to nearest cell index
        xedges = _edges_from_centres(xs)
        yedges = _edges_from_centres(ys)

        # ---------- figure and image ----------

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(DivergentSystem._bg_col)   # figure background (margins)
        ax.set_facecolor(DivergentSystem._bg_col)          # axes background

        im = ax.imshow(
            view2d.T,                  # imshow expects (ny, nx)
            cmap=theme,
            origin="lower",
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
        )

        # Labels reflect the true source axes we are plotting
        ax.set_xlabel(f"v{ax_x}", color=DivergentSystem._fg_col)
        ax.set_ylabel(f"v{ax_y}", color=DivergentSystem._fg_col)

        # Title shows the fixed slice when applicable
        control, volume = self._title_parts(base=base, limit=limit, config=config)
        title = f"{self.__class__.__name__} Heatmap{control}{volume}"
        if smap.fixed_axis is not None:
            fixed_val = float(base[smap.fixed_axis])  # singleton axis value
            title += f"  [slice axis{smap.fixed_axis}={fixed_val:.6g}]"
        ax.set_title(title, color=DivergentSystem._fg_col)

        # Light foreground for readability
        ax.tick_params(colors=DivergentSystem._fg_col)
        for spine in ax.spines.values():
            spine.set_color(DivergentSystem._fg_col)

        # Colour bar styled to match
        cbar = plt.colorbar(im, ax=ax, label="Normalised difference")
        cbar.ax.set_facecolor(DivergentSystem._bg_col)
        cbar.outline.set_edgecolor(DivergentSystem._fg_col)
        cbar.set_label("Normalised difference", color=DivergentSystem._fg_col)
        cbar.ax.tick_params(colors=DivergentSystem._fg_col)

        plt.tight_layout()
        plt.rcParams['savefig.facecolor'] = DivergentSystem._bg_col

        # Mutable copy of config so changes persist between presses
        ctrl_vars = list(config) if config is not None else []

        # ---------- interaction helpers ----------

        _press = {"e": None, "xy": None}

        def _toolbar_active(event) -> bool:
            tb = getattr(event.canvas, "toolbar", None)
            return bool(tb and getattr(tb, "mode", ""))  # "pan/zoom" or "zoom rect"

        def _inside_axes_and_extent(event) -> bool:
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return False
            x0, x1, y0, y1 = im.get_extent()
            return (min(x0, x1) <= event.xdata <= max(x0, x1)
                    and min(y0, y1) <= event.ydata <= max(y0, y1))

        def _to_indices(x: float, y: float) -> Tuple[int, int]:
            """Map world coordinates to nearest cell indices on the current view."""
            ix = int(np.clip(np.searchsorted(xedges, x) - 1, 0, nx - 1))
            iy = int(np.clip(np.searchsorted(yedges, y) - 1, 0, ny - 1))
            return ix, iy

        def _indices_to_origin(ix: int, iy: int) -> Tuple[float, float, float]:
            """Rebuild the full 3D world coordinate from 2D indices and the slice map."""
            coords = [0.0, 0.0, 0.0]
            coords[ax_x] = float(xs[ix])
            coords[ax_y] = float(ys[iy])
            if smap.fixed_axis is not None:
                coords[smap.fixed_axis] = float(base[smap.fixed_axis])
            else:
                # 2D input: prefer caller-provided z, otherwise 0
                coords[2] = float(base[2]) if len(base) >= 3 else 0.0
            return tuple(coords)

        def _extract_view_from_full(arr3: np.ndarray) -> Optional[np.ndarray]:
            """Take the slice along the fixed axis so that shape matches (nx, ny)."""
            if arr3.ndim == 3:
                try:
                    view = np.take(arr3, indices=0 if smap.fixed_index is None else smap.fixed_index,
                                   axis=smap.fixed_axis if smap.fixed_axis is not None else 2)
                except Exception:
                    return None
                return view
            if arr3.ndim == 2:
                return arr3
            return None

        def _recompute_for_view(
                x0_view: float,
                x1_view: float,
                y0_view: float,
                y1_view: float
            ) -> Optional[np.ndarray]:
            """
            Recompute the heatmap over the requested X/Y world bounds on the plotted axes,
            keeping the fixed axis at its slice value. Resolution matches the current nx×ny.
            """
            # Monotonic bounds for kernel
            bx0, bx1 = (x0_view, x1_view) if x0_view <= x1_view else (x1_view, x0_view)
            by0, by1 = (y0_view, y1_view) if y0_view <= y0_view else (y1_view, y0_view)  # corrected below
            # Bug fix: proper monotonic for Y
            by0, by1 = (y0_view, y1_view) if y0_view <= y1_view else (y1_view, y0_view)

            # Build 3D base/limit with the fixed axis collapsed
            new_base = list(base[:3])
            new_limit = list(limit[:3])
            new_stride = [0.0, 0.0, 0.0]

            # Assign along plotted axes
            new_base[ax_x], new_limit[ax_x] = bx0, bx1
            new_base[ax_y], new_limit[ax_y] = by0, by1
            # Keep fixed axis collapsed to a slice
            if smap.fixed_axis is not None:
                fixed_val = float(base[smap.fixed_axis])
                new_base[smap.fixed_axis] = fixed_val
                new_limit[smap.fixed_axis] = fixed_val

            # Match on-screen pixel density
            new_stride[ax_x] = (bx1 - bx0) / max(nx, 1)
            new_stride[ax_y] = (by1 - by0) / max(ny, 1)
            # Fixed axis stride stays 0

            # Title shows the fixed slice when applicable
            control, volume = self._title_parts(base=tuple(new_base), limit=tuple(new_limit), config=tuple(ctrl_vars))
            title = f"{self.__class__.__name__} Heatmap{control}{volume}"
            if smap.fixed_axis is not None:
                fixed_val = float(base[smap.fixed_axis])  # singleton axis value
                title += f"  [slice axis{smap.fixed_axis}={fixed_val:.6g}]"
            ax.set_title(title, color=DivergentSystem._fg_col)

            # Recompute the heatmap at the new location (pick a moderate steps number)
            full = self.get_heatmap(
                base=tuple(new_base),
                limit=tuple(new_limit),
                stride=tuple(new_stride),
                config=tuple(ctrl_vars) if ctrl_vars else None,
                steps=self._steps,
            )

            # Extract the 2D view and flip to match the current on-screen direction
            view = _extract_view_from_full(full)
            if view is None or view.ndim != 2:
                return None

            if x1_view < x0_view:
                view = view[::-1, :]
            if y1_view < y0_view:
                view = view[:, ::-1]

            # Equalise etc. for display
            _mask, _count, view = self._plot_preprocess(view, equal)

            # Update global axes arrays so picking stays correct after zoom
            nonlocal xs, ys, xedges, yedges
            xs = np.linspace(bx0, bx1, nx)
            ys = np.linspace(by0, by1, ny)
            xedges = _edges_from_centres(xs)
            yedges = _edges_from_centres(ys)

            return view

        # ---------- event handlers ----------

        def _on_zoom_or_pan(event):
            """
            Handle scroll or toolbar pan/zoom events by recomputing the heatmap for
            the visible region at matching pixel density, preserving axis orientation.
            """
            if event.inaxes is None or event.inaxes is not ax:
                return
            x0_view, x1_view = ax.get_xlim()
            y0_view, y1_view = ax.get_ylim()

            new_view = _recompute_for_view(x0_view, x1_view, y0_view, y1_view)
            if new_view is None:
                return

            im.set_data(new_view.T)  # transpose: image expects (ny, nx)
            im.set_extent([x0_view, x1_view, y0_view, y1_view])

            fig.canvas.draw_idle()

        def _on_key(event, delta=0.1):
            """
            Handle key presses to adjust control variables or save the plot.

            - Number keys 1–9 adjust the corresponding control variable by `delta`
              (hold Shift to subtract).
            - Regenerates and updates the heatmap after changes.
            - Pressing Shift+S saves the plot.
            """
            if not ctrl_vars:
                if event.key in ("shift+s",):
                    _save_plot()
                return

            key = event.key or ""
            shift = key.startswith("shift+")
            keynum = key.replace("shift+", "")

            if keynum.isdigit():
                idx = int(keynum) - 1
                if 0 <= idx < len(ctrl_vars):
                    step = float(delta)
                    ctrl_vars[idx] = ctrl_vars[idx] - step if shift else ctrl_vars[idx] + step
                    print(f"Control var {idx} now {ctrl_vars[idx]:.6f}")

                    # Recompute across the current view limits with updated config
                    x0_view, x1_view = ax.get_xlim()
                    y0_view, y1_view = ax.get_ylim()
                    new_view = _recompute_for_view(x0_view, x1_view, y0_view, y1_view)
                    if new_view is not None:
                        im.set_data(new_view.T)
                        fig.canvas.draw_idle()
            elif event.key in ("shift+s",):
                _save_plot()

        def _on_press(event):
            """Store mouse press event if inside plot axes and not using the toolbar."""
            _press["e"] = event if _inside_axes_and_extent(event) and not _toolbar_active(event) else None
            _press["xy"] = (event.x, event.y) if _press["e"] is not None else None

        def _on_release(event):
            """
            Handle mouse release to trigger zoom/pan or plot a trajectory.

            Notes:
                - If toolbar is active, treats release as zoom/pan end.
                - If movement is within click threshold, prints coordinates and
                  plots a trajectory from that point using full 3D world coords.
                - Otherwise, performs zoom or pan update.
            """
            if _toolbar_active(event):
                _on_zoom_or_pan(event)
                return

            click_thresh = 8  # <= movement that counts as a click
            pevent = _press.pop("e", None)
            if pevent is None or event.inaxes is not ax or not _inside_axes_and_extent(event):
                return

            dx = event.x - pevent.x
            dy = event.y - pevent.y  # pixel coords
            if dx*dx + dy*dy <= click_thresh:
                # Click: map to grid cell, then to full 3D world coordinates
                ix, iy = _to_indices(event.xdata, event.ydata)
                origin = _indices_to_origin(ix, iy)
                print(f"click at v{ax_x}={origin[ax_x]:.6g}, v{ax_y}={origin[ax_y]:.6g}"
                      + (f", v{smap.fixed_axis}={origin[smap.fixed_axis]:.6g}" if smap.fixed_axis is not None else ""))
                print(f"steps={self._steps}")
                # Fudge: magnify the number of steps for the trajectory - we usually want more than
                # we used in the heatmap (because heatmaps are so computationally expensive!)
                steps = min(10 * self._steps, 10000)
                trajectory = self.get_trajectory(origin=origin, steps=steps)
                self._plot = self.plot_trajectory(trajectory, origin=origin, block=False, plot=self._plot)
            else:
                # Drag (pan or zoom via bbox selection without toolbar)
                _on_zoom_or_pan(event)

        def _save_plot():
            """
            Save the current figure as a PNG in the script directory using a timestamped filename.

            Notes:
                - Uses `_plot_leafname()` to generate the output path.
                - Renders the latest view before saving.
                - Prints the saved path to stdout.
            """
            out = self._plot_leafname(path=self._script_dir, ext="png")
            fig.canvas.draw()  # ensure the latest view is rendered
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved {out}")

        # ---------- connect events ----------

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("scroll_event", _on_zoom_or_pan)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        if save:
            _save_plot()
        else:
            plt.show(block=block)

        # Return the plot handle for this plot
        return DivergentPlotHandle(fig=fig, ax=ax, line=None)

    def plot_heatmap(
            self,
            data: np.ndarray,
            base: Tuple = None,
            limit: Tuple = None,
            stride: Tuple = None,
            config: Tuple = None,
            equal: bool = True,
            theme: str = "inferno",
            title: str = None,
            block: bool = False,
            save: bool = False
        ) -> DivergentPlotHandle:
        """
        Render a 1D, 2D or 3D FTLE field over a state space (a finite subset of the phase space) for
        the specified system.

        Args:
            data (np.ndarray): The array of chaos metrics.
            base: Lower bound of search space in each dimension (defaults to self.base).
            limit: Upper bound of search space in each dimension.
            stride: Step size in each dimension.
            config: The control variable states passed into get_heatmap()
            equal: Equalise and normalise the chaos values to get a more even distribution.
            theme (str): Colour scheme to use (e.g. 'viridis', 'plasma', 'inferno', 'magma', 'cividis').
            title (str): Title of the plot.
            block: Should this window block program progress until closed? (default: False).

        Returns:
            - plot handle for the heatmap plot
        """

        base, limit, stride, config = self._sanitise_volume(base, limit, stride, config)

        if title is None:
            control, volume = self._title_parts(base=base, limit=limit, config=config)
            title = f"{self.__class__.__name__} Heatmap{control}{volume}"

        dims = data.ndim
        if dims not in (2, 3):
            raise ValueError("Data must be 2D or 3D for plotting.")

        elif dims == 2 or (dims == 3 and sum(s == 1 for s in data.shape) == 1):
            # Pass the untouched data (2D or 3D-with-singleton) straight through
            return self.plot_heatmap_2D(
                data,
                base=base,
                limit=limit,
                stride=stride,
                config=config,
                equal=equal,
                theme=theme,
                block=block,
                save=save
            )

        # 3D

        # Prepare data for volume rendering. Equalisation may adjust the distribution but must
        # return an array of the same shape. The mask/count can be useful for diagnostics.
        finite_mask, finite_count, data = self._plot_preprocess(data, equal)

        # Voxel counts and world-space bounds for each axis.
        nx, ny, nz = data.shape
        bx, by, bz = map(float, base)
        lx, ly, lz = map(float, limit)

        # Determine the scalar range present in the data and guard against edge cases.
        # If the range collapses to a point, expand it slightly to avoid zero-width TFs.
        vmin = float(np.nanmin(data)); vmax = float(np.nanmax(data))
        if not np.isfinite([vmin, vmax]).all():
            raise ValueError("Data contains only NaN/Inf values.")
        if vmax == vmin:
            vmax = vmin + 1.0

        # Ensure world axes increase with position. If a limit is less than its base,
        # reverse the corresponding data axis and swap the world bounds so that
        # coordinates remain monotonically increasing.
        vol = data
        ox, ox_end = bx, lx
        if lx < bx:
            vol = vol[::-1, :, :]
            ox, ox_end = lx, bx

        oy, oy_end = by, ly
        if ly < by:
            vol = vol[:, ::-1, :]
            oy, oy_end = ly, by

        oz, oz_end = bz, lz
        if lz < bz:
            vol = vol[:, :, ::-1]
            oz, oz_end = lz, bz

        # Replace NaN and infinities with boundary values to keep samples within the
        # transfer-function domain and avoid rendering artefacts.
        vol = np.nan_to_num(vol, nan=vmin, posinf=vmax, neginf=vmin)

        # Compute grid spacing in world units. Protect against single-voxel axes by
        # treating the denominator as at least 1.
        dx = (ox_end - ox) / max(nx - 1, 1)
        dy = (oy_end - oy) / max(ny - 1, 1)
        dz = (oz_end - oz) / max(nz - 1, 1)

        # Allocate a uniform rectilinear grid and populate its geometry. VTK expects
        # dimensions as point counts, not cell counts, hence the +1 on each axis.
        grid = pv.UniformGrid() if hasattr(pv, "UniformGrid") else pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)   # POINT counts
        grid.origin = (ox, oy, oz)
        grid.spacing = (dx, dy, dz)

        # Provide the scalar field on cells, then interpolate to points for smoother
        # volume rendering. Use Fortran order to match VTK's column-major expectations.
        grid.cell_data.clear()
        grid.cell_data["chaos"] = np.ascontiguousarray(vol.ravel(order="F"))
        grid = grid.cell_data_to_point_data()
        grid.set_active_scalars("chaos")

        # Sample a Matplotlib colourmap uniformly into 256 RGBA entries, stored as 8-bit.
        m_cmap = plt.get_cmap(theme, 256)
        rgba = (m_cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

        # Build a VTK lookup table from the sampled RGBA palette. VTK stores channels
        # as floats in [0, 1], so normalise each row before insertion.
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()
        for i in range(256):
            r, g, b, a = rgba[i] / 255.0
            lut.SetTableValue(i, r, g, b, a)

        # Map each palette row to a scalar position across the data range. This is the
        # domain for the transfer functions.
        xs = vmin + (vmax - vmin) * (np.linspace(0.0, 1.0, rgba.shape[0]))

        # Construct a colour transfer function by placing one RGB point per palette row
        # across [vmin, vmax]. Alpha is handled separately by the opacity function.
        ctf = vtk.vtkColorTransferFunction()
        xs = np.linspace(vmin, vmax, 256)
        for x, (r8, g8, b8, _) in zip(xs, rgba):
            ctf.AddRGBPoint(float(x), r8/255.0, g8/255.0, b8/255.0)

        # Construct an opacity transfer function. Opacity follows a power-law falloff
        # from vmin to vmax. Larger 'mag' values concentrate opacity near vmin.
        otf = vtk.vtkPiecewiseFunction()
        # mag = 2.75
        mag = 3.0
        scale = 0.75
        a = np.power(np.linspace(0.8, 0.1, 256), mag)
        for x, ai in zip(xs, a):
            otf.AddPoint(float(x), float(scale * ai))

        # Create the plotting context and add the volume. A manual colour/opacity setup
        # is used, so no colormap is requested from PyVista. Use a dark theme and light
        # text.
        dark = pv.themes.DarkTheme()
        dark.font.color = "#eaecef"
        p = pv.Plotter(window_size=(1280, 1024), off_screen=save, theme=dark)
        p.set_background("#15203b", top="#0a0f1c")
        actor = p.add_volume(
            grid,
            scalars="chaos",
            cmap=None,
            clim=(vmin, vmax),
            shade=False,
            scalar_bar_args=dict(title="Equalised and normalised chaos"),
        )
        prop = actor.prop
        m = actor.mapper

        # Apply the transfer functions to the volume property.
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetIndependentComponents(True)

        # Keep the mapper and LUT domains consistent with the data range. This avoids
        # out-of-range annotations and ensures the scalar bar represents the same range.
        m.scalar_range = (float(vmin), float(vmax))
        try:
            lut.SetRange(float(vmin), float(vmax))      # VTK 9+
        except AttributeError:
            lut.SetTableRange(float(vmin), float(vmax)) # Older VTK

        # Disable special colours for samples below/above range so the scalar bar does
        # not display "below"/"above" swatches.
        lut.SetUseBelowRangeColor(False)
        lut.SetUseAboveRangeColor(False)

        # Add a scalar bar linked to this mapper/LUT so colours in the legend match the
        # rendered volume.
        m.lookup_table = lut
        p.add_scalar_bar(
            title="Equalised and normalised chaos",
            n_colors=256,
            mapper=m
        )

        # Choose a world-space sampling step based on an approximate voxel diagonal.
        # This controls ray-marching step size and the opacity unit distance below.
        xmin, xmax, ymin, ymax, zmin, zmax = actor.GetBounds()
        sx, sy, sz = (xmax - xmin) / 256, (ymax - ymin) / 256, (zmax - zmin) / 256
        voxel_diag = (sx*sx + sy*sy + sz*sz) ** 0.5
        sample = 0.5 * voxel_diag

        # Prefer a fixed sampling distance to keep results predictable across runs.
        try:
            m.SetAutoAdjustSampleDistances(False)
        except Exception as e:
            print(f"skipped exception {e}")
            pass

        try:
            m.SetSampleDistance(float(sample))
        except Exception as e:
            print(f"skipped exception {e}")
            pass

        # Scale accumulated opacity per unit distance in world coordinates so that it
        # is consistent with the chosen sampling step.
        prop.SetScalarOpacityUnitDistance(float(sample))   # try 1*diag; increase to 2*diag if still too opaque

        # Enable stochastic jittering of sample positions to reduce banding from regular
        # sampling, especially at lower sample rates.
        try:
            m.SetUseJittering(True)  # Reduces banding at lower sample rates
        except Exception as e:
            print(f"skipped exception {e}")
            pass

        # Add scene aids for orientation and context.
        p.add_axes(line_width=2, color="#eaecef")
        p.add_bounding_box(color="gray", line_width=1)
        p.add_text(
            title or f"{self.__class__.__name__} Heatmap ({self.variables})",
            font_size=16,
            color="#eaecef",
            shadow=False
        )

        if save:
            # Create a short video of the volume rotating
            fps = 30
            n_frames = fps * 30
            outfile = self._plot_leafname(path=self._script_dir, ext="mp4")

            # Keep window alive for rendering
            p.show(auto_close=False)

            p.open_movie(
                outfile,
                framerate=fps,
                format='FFMPEG',    # Force ffmpeg plugin
                quality=9,
                codec='libx264',
                pixelformat='yuv420p',
                # Prefer CRF/preset over `quality` to avoid conflicting rate controls
                ffmpeg_params=[
                    '-movflags', '+faststart',   # progressive download
                    '-profile:v', 'high',        # broad compatibility
                    '-level:v', '4.1',
                    '-crf', '18',                # quality target (lower = higher quality)
                    '-preset', 'slow',           # encode speed vs size
                    '-an'                        # no audio track
                ]
            )

            # Give the actor a user transform that we can update each frame
            tf = vtk.vtkTransform()
            actor.SetUserTransform(tf)

            # Compute the centre (world coords) once
            xmin, xmax, ymin, ymax, zmin, zmax = actor.GetBounds()
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)

            # Ensure the camera looks at the centre and has a sensible distance
            cam = p.camera
            cam.SetFocalPoint(cx, cy, cz)
            if np.linalg.norm(np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())) < 1e-9:
                # Start a little away from the centre if currently at the focal point
                diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
                cam.SetPosition(cx + 1.5 * diag, cy, cz)
            cam.SetViewUp(0.0, 0.0, 1.0)  # z-up

            # Degrees per frame (tune to taste)
            rx, ry, rz = 0.1, 0.43, 0.0   # azimuth, elevation, roll

            for f in range(n_frames):
                print(f"write frame {f} of {n_frames}")
                cam.Azimuth(rx)            # orbit around view-up axis at the focal point
                cam.Elevation(ry)          # tilt up/down around focal point
                if rz:
                    cam.Roll(rz)           # optional: rotate about view axis (horizon)
                cam.OrthogonalizeViewUp()  # keeps view-up stable after elevation
                p.render()
                p.write_frame()
                rx += 0.001

            p.close()  # Closes the movie writer and the plotter
        else:
            # Display the plot window and optionally block
            if block:
                p.show()
            else:
                p.show(auto_close=False)

        return DivergentPlotHandle(None, None, None)
