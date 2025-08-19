# DivergentSystem (CUDA): Windows install and quick start

## Overview
This repository provides a base class, `DivergentSystem`, for compiling and running CUDA kernels from Python via PyCUDA, plus utilities for computing trajectories and chaos heatmaps and for plotting results. Several example implementations are included. These instructions are written for Windows users. Other OS users should be able to adapt the steps fairly straightforwardly.

Technical note: the CUDA kernels are passed to `pycuda.compiler.SourceModule`, compiled by NVCC, then launched through the CUDA driver API. It is assumed that you have a suitable NVIDA graphics card (GPUs) availble (not sure? check here <https://developer.nvidia.com/cuda-gpus>).

---

## 1) Prerequisites (Windows 10 or 11, 64‑bit)

1. **NVIDIA GPU and driver**  
   - An NVIDIA CUDA‑capable GPU with a recent driver.
   - <https://www.nvidia.com/Download/index.aspx/>
2. **Microsoft Visual C++ Build Tools**  
   - Install the “Desktop development with C++” workload (community edition is fine). NVCC uses the MSVC host compiler on Windows.
   - <https://visualstudio.microsoft.com/downloads/>
3. **NVIDIA CUDA Toolkit**  
   - Install a CUDA Toolkit that matches the driver and MSVC. Keep the default options so `nvcc.exe` and CUDA DLLs are on `PATH`.
   - <https://developer.nvidia.com/cuda-downloads>
4. **Python (64‑bit)**  
   - Install Python 3.11 for Windows. During setup, tick “Add python.exe to PATH”. Other versions may work, but 3.11 usually avoids wheel issues on Windows.
   - <https://www.python.org/downloads/windows/>

> Tip: On Windows it is often convenient to use PowerShell or the “Developer PowerShell for VS 2022”.

---

## 2) Get the sources
The sources are hosted on GitHub: <https://github.com/sarev/DivergentSystem>

### Option A: clone with Git
```powershell
cd C:\dev
git clone https://github.com/sarev/DivergentSystem.git
cd DivergentSystem

---

## 3) Create and activate a virtual environment

Create a dedicated python vitrual environment and upgrade packaging tools immediately.

```powershell
cd C:\dev\divergent
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
```

To deactivate later: `deactivate`

> Tip: these environments only affect the shell instance where they are activated. You can activate the environment in multiple shells at the same time. You don't *have* to deactivate before closing a terminal window (it will effectively tidy itself up).

---

## 4) Install Python dependencies

Install the required packages.

```powershell
pip install numpy matplotlib pycuda pyvista vtk typing_extensions
```

What these are used for:
- `pycuda` compiles and loads the kernels, then launches them through the driver.
- `numpy` provides array operations.
- `matplotlib` handles plotting of trajectories and heatmaps.
- `pyvista` and `vtk` are used by the plotting utilities.
- `typing_extensions` used for typing hints in function/method parameters.

If you prefer `requirements.txt`, the minimal set is:

```
numpy
matplotlib
pycuda
pyvista
vtk
typing_extensions
```

Install with:

```powershell
pip install -r requirements.txt
```

---

## 5) Verify the CUDA toolchain

First confirm that NVCC is visible:
```powershell
nvcc --version
```

Then confirm that PyCUDA can see at least one device. Create a file named `check_cuda.py` with the following content, save it next to the repository files, then run `python check_cuda.py` from the activated environment.

```python
# check_cuda.py
import pycuda.driver as drv

def main() -> None:
    drv.init()
    n = drv.Device.count()
    print(f"CUDA devices: {n}")
    for i in range(n):
        d = drv.Device(i)
        name = d.name()
        cc = d.compute_capability()
        print(f"{i}: {name} CC {cc[0]}.{cc[1]}")

if __name__ == "__main__":
    main()
```

If this prints one or more devices, the setup is ready. The runtime selects compatible devices and manages per‑device module loading.

---

## 6) Quick start with the Lorenz system

### Option A: run the provided `LorenzSystem.py`
`LorenzSystem.py` includes a small CLI. The following command runs the example with classic Lorenz parameters and a modest trajectory length for a quick test:

```powershell
python LorenzSystem.py --sigma 10 --rho 28 --beta 2.6666667 --origin 1 1 1 --steps 2500
```

The script will compute results using the CUDA kernels in `LorenzSystem` and save or show figures depending on the configured sections in `__main__`.

### Option B: minimal example (`example.py`)
If you prefer a minimal demonstration, create `example.py` in the repository root with the following content:

```python
# example.py
from LorenzSystem import LorenzSystem

def main() -> None:
    system = LorenzSystem()
    system.config = (10.0, 28.0, 8.0/3.0)  # sigma, rho, beta
    origin = (1.0, 1.0, 1.0)
    traj = system.get_trajectory(origin, steps=5000)
    system.plot_trajectory(traj, origin, block=True)

if __name__ == "__main__":
    main()
```

Run it with:
```powershell
python example.py
```

This compiles the kernels on first use, caches them as PTX files for reuse, generates a Lorenz trajectory from the given starting point, and displays it as a 3D plot.

---

## 7) How kernel compilation and caching works

- Build artefacts and cache files are written under `CUDA_cache\` next to the Python files.
- NVCC is invoked by PyCUDA with `--keep` and a unique `--keep-dir`. The generated `.ptx` is moved into a stable location based on the device compute capability and a hash of the kernel source.
- Temporary build folders are cleaned automatically on completion.

No manual steps are required.

> Note: this is very hacky! It has to work-around a general lack of transparency around what pycuda is doing behind the scenes.

---

## 8) Troubleshooting

- **`No compatible CUDA devices found.`**  
  Check that the NVIDIA driver is correctly installed and that the user account can access the GPU. Ensure the device’s compute capability is supported by the installed CUDA Toolkit.

- **`nvcc` or `cl.exe` not found**  
  Install the CUDA Toolkit and MSVC Build Tools. If needed, run from the Developer Command Prompt, or add CUDA `bin` and the MSVC toolchain to `PATH`.

- **`pycuda._driver.LogicError: cuInit failed`**  
  This usually indicates a missing or incompatible NVIDIA driver. Reinstall or update the driver.

- **Matplotlib windows do not appear**  
  Avoid running in headless contexts. If using remote desktops, check the backend or save figures to files instead of showing them.

---

## 9) Uninstall or clean up

To remove the virtual environment:
```powershell
deactivate
rmdir /s /q .venv
```
To clear cached kernels, delete the `CUDA_cache\` directory.

---

## 10) Notes for Linux users
Linux users can follow the same outline: install a matching NVIDIA driver and CUDA Toolkit, create a virtual environment with Python 3.11 or newer, then install the same Python dependencies. Replace the activation command with `source .venv/bin/activate` and use the appropriate package manager for system prerequisites.

---

That should be all that is required to get the example systems running on Windows with a CUDA‑capable GPU. Enjoy!
