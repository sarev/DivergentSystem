#!/usr/bin/env python3
#
# HalvorsenSystem.py
#
# Copyright 2025 7th software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from DivergentSystem import DivergentSystem


class ShimizuMoriokaSystem(DivergentSystem):
    def __init__(self):
        # Common helpers shared by both kernels, split out to a separate docstring.
        _helpers = r"""

        // Compute the instantaneous time derivatives of the Shimizu-Morioka system
        // using double precision operations.
        //
        //   dX/dt = Y
        //   dY/dt = X - a*Y - X*Z
        //   dZ/dt = -b*Z + X^2
        //
        // Inputs:
        //   X, Y, Z : current state
        //   a, b    : system parameters
        // Outputs (by reference):
        //   dX, dY, dZ : derivatives at (X, Y, Z)
        //
        // Notes:
        //   - Assumes finite inputs; performs no clamping or normalisation.
        //
        __forceinline__ __device__ inline void shimizu_morioka_derivs_d(
            double X, double Y, double Z,
            double a, double b,
            double &dX, double &dY, double &dZ)
        {
            dX = Y;
            dY = X - a*Y - X*Z;
            dZ = -b*Z + X*X;
        }

        // Map a linear lattice index to 3D coordinates in single precision.
        // Indexing order is z-fastest, then y, then x (row-major with z stride 1).
        //
        // Given a 1D index idx in [0, nx*ny*nz), compute integer lattice indices:
        //   iz = idx % nz
        //   iy = (idx / nz) % ny
        //   ix =  idx / (ny * nz)
        // and convert them to world-space coordinates:
        //   x = x0 + ix * sx
        //   y = y0 + iy * sy
        //   z = z0 + iz * sz
        //
        // Inputs:
        //   idx          : linear index into an nx×ny×nz lattice
        //   x0,y0,z0     : base (origin) of the lattice
        //   sx,sy,sz     : strides between adjacent samples along x,y,z
        //   nx,ny,nz     : lattice extents (unused here except to document layout)
        // Outputs (by reference):
        //   x,y,z        : single-precision coordinates for this lattice point
        //
        // Notes:
        //   - No bounds checking is performed; caller must ensure 0 <= idx < nx*ny*nz.
        //   - Designed for use inside windowed kernels where idx already includes any offset.
        //
        __forceinline__ __device__ inline void lattice_coords_f32(
            int idx, float x0, float y0, float z0,
            float sx, float sy, float sz,
            int nx, int ny, int nz,
            float &x, float &y, float &z)
        {
            int tmp = idx / nz;
            int iz  = idx - tmp * nz;
            int ix  = tmp / ny;
            int iy  = tmp - ix * ny;

            x = x0 + ix * sx;
            y = y0 + iy * sy;
            z = z0 + iz * sz;
        }
        """

        system_kernel = (
            r"""
            extern "C" {

            """
            + _helpers +
            r"""

            // Integrate one Shimizu–Morioka trajectory with fixed-step RK4 and
            // write the path as float3 samples.
            //
            // Launch:
            //   Single-trajectory kernel. Use <<<1,1>>> or ensure only one thread runs.
            //   Thread and block indices are not used.
            //
            // Parameters:
            //   trajectory : output buffer of length steps*3, layout [x, y, z] per step
            //   x, y, z    : initial state at t = 0 (promoted to double internally)
            //   a, b       : system parameters
            //   steps      : number of integration steps
            //
            // Behaviour:
            //   - Uses constant timestep dt = 0.01 in double precision for all maths.
            //   - At each i in [0, steps): emits the current state, then advances one RK4 step
            //     using shimizu_morioka_derivs_d to compute k1..k4.
            //   - Converts state to float before writing to reduce bandwidth.
            //   - Terminates early if any state component becomes non-finite.
            //
            // Notes:
            //   - No bounds checking of the trajectory pointer.
            //   - Stability and fidelity depend on dt and steps; tune per use case.
            //
            __global__ void system_kernel(
                float *trajectory,
                float x, float y, float z,
                const float a, const float b,
                const int steps)
            {
                const double dt = 0.01; // integrator timestep

                // Promote to double for integration accuracy
                double xd = (double)x;
                double yd = (double)y;
                double zd = (double)z;

                for (int i = 0; i < steps; ++i) {
                    // Emit current state as float3
                    trajectory[i*3 + 0] = (float)xd;
                    trajectory[i*3 + 1] = (float)yd;
                    trajectory[i*3 + 2] = (float)zd;

                    // RK4 integration step in double
                    double k1x, k1y, k1z;
                    shimizu_morioka_derivs_d(xd, yd, zd, (double)a, (double)b, k1x, k1y, k1z);

                    double x2 = xd + 0.5*dt*k1x;
                    double y2 = yd + 0.5*dt*k1y;
                    double z2 = zd + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    shimizu_morioka_derivs_d(x2, y2, z2, (double)a, (double)b, k2x, k2y, k2z);

                    double x3 = xd + 0.5*dt*k2x;
                    double y3 = yd + 0.5*dt*k2y;
                    double z3 = zd + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    shimizu_morioka_derivs_d(x3, y3, z3, (double)a, (double)b, k3x, k3y, k3z);

                    double x4 = xd + dt*k3x;
                    double y4 = yd + dt*k3y;
                    double z4 = zd + dt*k3z;
                    double k4x, k4y, k4z;
                    shimizu_morioka_derivs_d(x4, y4, z4, (double)a, (double)b, k4x, k4y, k4z);

                    xd += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    yd += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    zd += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    if (!isfinite(xd) || !isfinite(yd) || !isfinite(zd)) {
                        break;
                    }
                }
            }

            } // extern "C"
            """
        )

        distance_kernel = (
            r"""
            extern "C" {

            """
            + _helpers +
            r"""

            // Compute a local chaos metric (FTLE-like) for a window of lattice points by
            // integrating two nearby Shimizu–Morioka trajectories per point and tracking
            // their separation growth.
            //
            // Launch and indexing:
            //   - Use a 1D kernel configuration. Choose blocks of 256 threads to match
            //     __launch_bounds__(256, 1). Grid size should satisfy blocks*threads >= n_items.
            //   - Each thread i_local processes exactly one lattice item with linear index
            //     idx = offset + i_local. Threads with i_local >= n_items return immediately.
            //
            // Inputs:
            //   a, b            : system parameters (single precision)
            //   px, py, pz      : perturbation added to the starting (x, y, z) of the second trajectory
            //   steps           : number of RK4 steps to integrate
            //   x0,y0,z0        : lattice base coordinates
            //   sx,sy,sz        : lattice strides along x, y, z
            //   nx,ny,nz        : lattice extents (documentary; mapping is done by lattice_coords_f32)
            //   offset,n_items  : linear window [offset, offset + n_items)
            //   out             : output buffer of length at least n_items; one float per lattice item
            //
            // Behaviour:
            //   - Map idx to starting coordinates (xf, yf, zf) using lattice_coords_f32.
            //   - Create two double-precision states: (x1,y1,z1) = (xf,yf,zf) and
            //     (x2,y2,z2) = (xf+px, yf+py, zf+pz).
            //   - Integrate both with fixed-step RK4, dt = 0.01, using shimizu_morioka_derivs_d.
            //   - Accumulate log(delta / delta0) each step, where delta is the current Euclidean
            //     separation and delta0 is the initial separation, guarded to avoid underflow.
            //   - Periodically renormalise the separation to ~delta0 every REORTHO steps to keep
            //     the two trajectories in the linear regime.
            //   - Early exit on any non-finite state or if the log accumulator exceeds a safe bound.
            //
            // Output:
            //   - Write a single float to out[i_local]: accum_log / (steps_done * dt).
            //     This approximates the finite-time Lyapunov exponent at the lattice point.
            //
            // Notes and constraints:
            //   - All heavy maths is in double precision; only I/O and parameters are float.
            //   - No bounds checking on out beyond the i_local guard.
            //   - Numerical guards: DELTA0_MIN, DELTA_MIN and ACCUM_MAX_ABS prevent divide-by-zero,
            //     log of zero, and runaway accumulation.
            //   - The z dimension participates in the lattice mapping even if the system is effectively 3D
            //     with a single state triple.
            //   - Ensure out is __restrict__ and properly aligned for best memory throughput.
            //
            __global__ void distance_kernel(
                const float a, const float b,
                const float px, const float py, const float pz,
                const int   steps,
                const float x0, const float y0, const float z0,
                const float sx, const float sy, const float sz,
                const int   nx, const int ny, const int nz,
                const int   offset, const int n_items,
                float* __restrict__ out)
            {
                const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
                if (i_local >= n_items) return;

                // Map to lattice coordinates and use as starting state
                const int idx = offset + i_local;
                float xf, yf, zf;
                lattice_coords_f32(idx, x0, y0, z0, sx, sy, sz, nx, ny, nz, xf, yf, zf);

                // Two trajectories: base and perturbed by (px, py, pz)
                double x1 = (double)xf;
                double y1 = (double)yf;
                double z1 = (double)zf;

                double x2 = (double)xf + (double)px;
                double y2 = (double)yf + (double)py;
                double z2 = (double)zf + (double)pz;

                // Initial separation magnitude
                double dx0 = x2 - x1;
                double dy0 = y2 - y1;
                double dz0 = z2 - z1;
                double delta0 = sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0);

                // Benettin FTLE settings
                const double dt = 0.01;
                const int    REORTHO       = 25;
                const double DELTA0_MIN    = 1e-20;
                const double DELTA_MIN     = 1e-30;
                const double ACCUM_MAX_ABS = 1e12;

                // Clamp tiny or zero initial separation
                if (!(delta0 > 0.0)) {
                    x2 = x1 + DELTA0_MIN; y2 = y1; z2 = z1;
                    delta0 = DELTA0_MIN;
                } else if (delta0 < DELTA0_MIN) {
                    const double s = DELTA0_MIN / delta0;
                    x2 = x1 + dx0 * s; y2 = y1 + dy0 * s; z2 = z1 + dz0 * s;
                    delta0 = DELTA0_MIN;
                }

                double accum_log = 0.0;
                int i = 0;

                for (i = 0; i < steps; ++i) {
                    // ===== Integrate first system =====
                    double k1x, k1y, k1z;
                    shimizu_morioka_derivs_d(x1, y1, z1, (double)a, (double)b, k1x, k1y, k1z);
                    double x1_2 = x1 + 0.5*dt*k1x;
                    double y1_2 = y1 + 0.5*dt*k1y;
                    double z1_2 = z1 + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    shimizu_morioka_derivs_d(x1_2, y1_2, z1_2, (double)a, (double)b, k2x, k2y, k2z);
                    double x1_3 = x1 + 0.5*dt*k2x;
                    double y1_3 = y1 + 0.5*dt*k2y;
                    double z1_3 = z1 + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    shimizu_morioka_derivs_d(x1_3, y1_3, z1_3, (double)a, (double)b, k3x, k3y, k3z);
                    double x1_4 = x1 + dt*k3x;
                    double y1_4 = y1 + dt*k3y;
                    double z1_4 = z1 + dt*k3z;
                    double k4x, k4y, k4z;
                    shimizu_morioka_derivs_d(x1_4, y1_4, z1_4, (double)a, (double)b, k4x, k4y, k4z);
                    x1 += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    y1 += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    z1 += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    // ===== Integrate second system =====
                    shimizu_morioka_derivs_d(x2, y2, z2, (double)a, (double)b, k1x, k1y, k1z);
                    double x2_2 = x2 + 0.5*dt*k1x;
                    double y2_2 = y2 + 0.5*dt*k1y;
                    double z2_2 = z2 + 0.5*dt*k1z;
                    shimizu_morioka_derivs_d(x2_2, y2_2, z2_2, (double)a, (double)b, k2x, k2y, k2z);
                    double x2_3 = x2 + 0.5*dt*k2x;
                    double y2_3 = y2 + 0.5*dt*k2y;
                    double z2_3 = z2 + 0.5*dt*k2z;
                    shimizu_morioka_derivs_d(x2_3, y2_3, z2_3, (double)a, (double)b, k3x, k3y, k3z);
                    double x2_4 = x2 + dt*k3x;
                    double y2_4 = y2 + dt*k3y;
                    double z2_4 = z2 + dt*k3z;
                    shimizu_morioka_derivs_d(x2_4, y2_4, z2_4, (double)a, (double)b, k4x, k4y, k4z);
                    x2 += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    y2 += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    z2 += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    if (!isfinite(x1) || !isfinite(y1) || !isfinite(z1) ||
                        !isfinite(x2) || !isfinite(y2) || !isfinite(z2)) {
                        break;
                    }

                    // Current separation and FTLE accumulation
                    double dx = x2 - x1;
                    double dy = y2 - y1;
                    double dz = z2 - z1;
                    double delta = fmax(sqrt(dx*dx + dy*dy + dz*dz), DELTA_MIN);
                    accum_log += log(delta / delta0);

                    // Periodic renormalisation to keep separation ~ delta0
                    if (((i + 1) % REORTHO) == 0) {
                        const double s = delta0 / delta;
                        x2 = x1 + dx * s;
                        y2 = y1 + dy * s;
                        z2 = z1 + dz * s;
                    }

                    if (!isfinite(accum_log) || fabs(accum_log) > ACCUM_MAX_ABS) {
                        break;
                    }
                }

                const double t_total = dt * (double)(i + 1);
                out[i_local] = (t_total > 0.0 && isfinite(t_total)) ? (float)(accum_log / t_total) : 0.0f;
            }

            } // extern "C"
            """
        )

        super().__init__(system_kernel, distance_kernel)
        self._variables = ["a", "b"]


if __name__ == "__main__":
    import argparse
    import pickle

    # Test harness aligned
    parser = argparse.ArgumentParser(
        description="Shimizu-Morioka test harness: set control variables and start point via CLI or use defaults.")
    parser.add_argument("--a", type=float, default=0.81, help="Parameter a. Default: 0.81")
    parser.add_argument("--b", type=float, default=0.375, help="Parameter b. Default: 0.375")

    parser.add_argument("--origin", "-o", nargs=3, type=float,
                        default=(0.1, 0.2, 0.1), metavar=("X0", "Y0", "Z0"),
                        help="Initial position (x0, y0, z0). Default: 0.1, 0.2, 0.1")
    parser.add_argument("--steps", "-s", type=int, default=20000, help="Integration steps. Default: 20000")

    args = parser.parse_args()

    instance = ShimizuMoriokaSystem()
    instance.config = (args.a, args.b)

    # origin = tuple(args.origin)
    # steps = args.steps

    # print(f"{instance.__class__.__name__} variables {instance.variables} are {instance.config}")
    # print(f"origin {origin}")
    # print(f"iterate for {steps} steps")

    # Single trajectory demo
    # traj = instance.get_trajectory(origin, steps)
    # print(f"trajectory length {len(traj)}")
    # instance.plot_trajectory(traj, origin, block=False)

    # Heatmap demo over a typical Shimizu-Morioka region
    # steps = 4000
    # instance.base = (-2.5, -2.5, -2.5)
    # instance.limit = (2.5, 2.5, 2.5)

    # divs = 256
    # instance.calc_stride(divs)
    # print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

    # import pickle

    # pkl = False
    # if pkl:
    #     heatmap = instance.get_heatmap(perturbation = (1e-8, 0, 0), steps=steps)
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl", 'wb') as f:
    #         pickle.dump(heatmap, f)
    # if not pkl:
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl",'rb') as f:
    #         heatmap = pickle.load(f)

    # p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=pkl, save=not pkl)
    # p.close()
    # exit()

    # instance.base = (-2.5, 2.0, -1.5)
    # instance.limit = (-1.0, 2.0, 0.0)
    # instance.config = (0.54, 0.46)

    # divs = 1024
    # instance.calc_stride(divs)
    # print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

    # pkl = True
    # if pkl:
    #     heatmap = instance.get_heatmap(perturbation = (1e-8, 0, 0), steps=steps)
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl", 'wb') as f:
    #         pickle.dump(heatmap, f)
    # if not pkl:
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl",'rb') as f:
    #         heatmap = pickle.load(f)

    # p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=pkl, save=not pkl)
    # p.close()
    # exit()

    # -- Content
    #
    # 3D heatmap
    #
    steps = 5000
    divs = 255
    instance.base = (-12.5, -12.5, -12.5)
    instance.limit = (12.5, 12.5, 12.5)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    # p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=True, save=False)
    p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=False, save=True)
    p.close()
    exit()

    # -- Content
    #
    # 3D heatmap
    #
    steps = 200
    divs = 255
    instance.base = (-2.5, -2.5, -2.5)
    instance.limit = (2.5, 2.5, 2.5)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap animation frames
    #
    y = 2.0
    instance.base = (-2.5, y, -2.5)
    instance.limit = (2.5, y, 2.5)
    divs = 1023
    instance.calc_stride(divs)
    steps = 4000
    print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")
    for var in range(0, 102, 3):
        a = var / 100.0
        b = 1.0 - a
        instance.config = (a, b)
        heatmap = instance.get_heatmap(steps=steps)
        p = instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=False, save=True)
        p.close()
