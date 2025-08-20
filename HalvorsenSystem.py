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


class HalvorsenSystem(DivergentSystem):
    def __init__(self):
        # Common helpers shared by both kernels
        _helpers = r"""
        // ---- helpers: double precision dynamics + lattice mapping ----
        __forceinline__ __device__ inline void halvorsen_derivs_d(
            double X, double Y, double Z,
            double a,
            double &dX, double &dY, double &dZ)
        {
            // Halvorsen system
            //   dX/dt = -a*X - 4*Y - 4*Z - Y^2
            //   dY/dt = -a*Y - 4*Z - 4*X - Z^2
            //   dZ/dt = -a*Z - 4*X - 4*Y - X^2
            dX = -a*X - 4.0*Y - 4.0*Z - Y*Y;
            dY = -a*Y - 4.0*Z - 4.0*X - Z*Z;
            dZ = -a*Z - 4.0*X - 4.0*Y - X*X;
        }

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

        # ---- CUDA kernels matching DivergentSystem contract ----
        system_kernel = (
            r"""
            extern "C" {

            """
            + _helpers +
            r"""

            __global__ void system_kernel(
                float *trajectory,
                float x, float y, float z,
                const float a,
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
                    halvorsen_derivs_d(xd, yd, zd, (double)a, k1x, k1y, k1z);

                    double x2 = xd + 0.5*dt*k1x;
                    double y2 = yd + 0.5*dt*k1y;
                    double z2 = zd + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    halvorsen_derivs_d(x2, y2, z2, (double)a, k2x, k2y, k2z);

                    double x3 = xd + 0.5*dt*k2x;
                    double y3 = yd + 0.5*dt*k2y;
                    double z3 = zd + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    halvorsen_derivs_d(x3, y3, z3, (double)a, k3x, k3y, k3z);

                    double x4 = xd + dt*k3x;
                    double y4 = yd + dt*k3y;
                    double z4 = zd + dt*k3z;
                    double k4x, k4y, k4z;
                    halvorsen_derivs_d(x4, y4, z4, (double)a, k4x, k4y, k4z);

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

            __global__ void distance_kernel(
                const float a,
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
                double x1 = (double)xf, y1 = (double)yf, z1 = (double)zf;
                double x2 = (double)xf + (double)px;
                double y2 = (double)yf + (double)py;
                double z2 = (double)zf + (double)pz;

                // Initial separation magnitude
                double dx0 = x2 - x1, dy0 = y2 - y1, dz0 = z2 - z1;
                double delta0 = sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0);

                // Benettin FTLE settings
                const double dt            = 0.01;
                const int    REORTHO       = 25;
                const double DELTA0_MIN    = 1e-20;
                const double DELTA_MIN     = 1e-30;
                const double LOGARG_MIN    = 1e-300; // avoid log(0)
                const double LOGARG_MAX    = 1e300;  // avoid log(inf)

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
                int    n_accum   = 0;      // how many times we added to accum_log
                bool   bad       = false;  // set if we detect non-finite state

                int i = 0;
                for (i = 0; i < steps; ++i) {
                    // ===== Integrate first system (RK4) =====
                    double k1x, k1y, k1z;
                    halvorsen_derivs_d(x1, y1, z1, (double)a, k1x, k1y, k1z);
                    double x1_2 = x1 + 0.5*dt*k1x, y1_2 = y1 + 0.5*dt*k1y, z1_2 = z1 + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    halvorsen_derivs_d(x1_2, y1_2, z1_2, (double)a, k2x, k2y, k2z);
                    double x1_3 = x1 + 0.5*dt*k2x, y1_3 = y1 + 0.5*dt*k2y, z1_3 = z1 + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    halvorsen_derivs_d(x1_3, y1_3, z1_3, (double)a, k3x, k3y, k3z);
                    double x1_4 = x1 + dt*k3x, y1_4 = y1 + dt*k3y, z1_4 = z1 + dt*k3z;
                    double k4x, k4y, k4z;
                    halvorsen_derivs_d(x1_4, y1_4, z1_4, (double)a, k4x, k4y, k4z);
                    x1 += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    y1 += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    z1 += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    // ===== Integrate second system (RK4) =====
                    halvorsen_derivs_d(x2, y2, z2, (double)a, k1x, k1y, k1z);
                    double x2_2 = x2 + 0.5*dt*k1x, y2_2 = y2 + 0.5*dt*k1y, z2_2 = z2 + 0.5*dt*k1z;
                    halvorsen_derivs_d(x2_2, y2_2, z2_2, (double)a, k2x, k2y, k2z);
                    double x2_3 = x2 + 0.5*dt*k2x, y2_3 = y2 + 0.5*dt*k2y, z2_3 = z2 + 0.5*dt*k2z;
                    halvorsen_derivs_d(x2_3, y2_3, z2_3, (double)a, k3x, k3y, k3z);
                    double x2_4 = x2 + dt*k3x, y2_4 = y2 + dt*k3y, z2_4 = z2 + dt*k3z;
                    halvorsen_derivs_d(x2_4, y2_4, z2_4, (double)a, k4x, k4y, k4z);
                    x2 += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    y2 += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    z2 += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    // Divergence/NaN guard for states
                    if (!isfinite(x1) || !isfinite(y1) || !isfinite(z1) ||
                        !isfinite(x2) || !isfinite(y2) || !isfinite(z2)) {
                        bad = true;
                        break;
                    }

                    // Current separation
                    double dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
                    double delta = fmax(sqrt(dx*dx + dy*dy + dz*dz), DELTA_MIN);

                    // Accumulate only at re-orthonormalisation points
                    if (((i + 1) % REORTHO) == 0) {
                        double ratio = delta / delta0;                 // > 0
                        ratio = fmin(fmax(ratio, LOGARG_MIN), LOGARG_MAX);
                        accum_log += log(ratio);
                        ++n_accum;

                        // Renormalise to keep separation ~ delta0
                        const double s = delta0 / fmax(delta, DELTA_MIN);
                        x2 = x1 + dx * s;
                        y2 = y1 + dy * s;
                        z2 = z1 + dz * s;
                    }
                }

                // Time corresponding to the contributions actually summed
                const double t_total = dt * (double)(n_accum * REORTHO);

                // Final write with robust guard
                float val;
                if (bad || !(t_total > 0.0) || !isfinite(accum_log)) {
                    val = NAN;  // or 0.0f if you prefer a sentinel
                } else {
                    val = (float)(accum_log / t_total);
                }
                out[i_local] = val;
            }

            } // extern "C"
            """
        )

        super().__init__(system_kernel, distance_kernel)
        self._variables = ["a"]


if __name__ == "__main__":
    # Test harness
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="HalvorsenSystem test harness: set control variable and start point via CLI or use defaults.")
    parser.add_argument("--a", type=float, default=1.4, help="Halvorsen parameter a. Default: 1.4")

    parser.add_argument("--origin", "-o", nargs=3, type=float,
                        default=(0.1, 0.0, 0.0), metavar=("X0", "Y0", "Z0"),
                        help="Initial position (x0, y0, z0). Default: 0.1, 0.0, 0.0")
    parser.add_argument("--steps", "-s", type=int, default=20000, help="Integration steps. Default: 20000")

    args = parser.parse_args()

    instance = HalvorsenSystem()
    instance.config = (args.a,)

    # origin = tuple(args.origin)
    # steps = args.steps

    # print(f"{instance.__class__.__name__} variables {instance.variables} are {instance.config}")
    # print(f"origin {origin}")
    # print(f"iterate for {steps} steps")

    # # Single trajectory demo
    # traj = instance.get_trajectory(origin, steps)
    # print(f"trajectory length {len(traj)}")
    # instance.plot_trajectory(traj, origin, block=False)

    # steps = 500

    # # divs = 255
    # # instance.base = (0, -2.5, -2.5)
    # # instance.limit = (5.0, 2.5, 2.5)

    # divs = 1023
    # instance.base = (0, -2.5, 1)
    # instance.limit = (5.0, 2.5, 1)

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

    # p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=pkl, save=not pkl)
    # p.close()
    # exit()

    # steps = 500
    # divs = 1024
    # print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

    # for y in range(-350, 350, 4):
    #     instance.base = (0, y / 100.0, -3.5)
    #     instance.limit = (5, y / 100.0, 3.5)
    #     instance.calc_stride(divs)
    #     print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")
    #     heatmap = instance.get_heatmap(perturbation = (1e-8, 0, 0), steps=steps)
    #     p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
    #     p.close()

    # -- Content
    #
    # 3D heatmap
    #
    steps = 500
    divs = 255
    instance.base = (-2.5, -2.5, -2.5)
    instance.limit = (2.5, 2.5, 2.5)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap
    #
    steps = 1000
    divs = 1023
    z = 1.0
    instance.base = (-2.5, -3.0, z)
    instance.limit = (2.5, 3.0, z)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap
    #
    steps = 800
    divs = 1023
    y = 1.5
    instance.base = (-2.5, y, -2.5)
    instance.limit = (2.5, y, 2.5)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap animation frames
    #
    instance.config = (1.9)
    steps = 500
    divs = 1279
    for y in range(-250, 250, 5):
        instance.base = (-2.5, y / 100.0, -2.5)
        instance.limit = (2.5, y / 100.0, 2.5)
        instance.calc_stride(divs)
        print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

        heatmap = instance.get_heatmap(steps=steps)
        # p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=True, save=False)
        p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
        p.close()

    # -- Content
    #
    # 3D heatmap
    #
    steps = 600
    divs = 255
    instance.base = (-2.5, -2.5, -2.5)
    instance.limit = (2.5, 2.5, 2.5)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap animation frames
    #
    steps = 2000
    divs = 1279
    for c in range(100, 251, 2):
        instance.config = (c / 100.0)
        instance.base = (-2.5, -2.5, 0.5)
        instance.limit = (2.5, 2.5, 0.5)
        instance.calc_stride(divs)
        print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

        heatmap = instance.get_heatmap(steps=steps)
        # p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=True, save=False)
        p = instance.plot_heatmap(heatmap, theme="BuPu_r", equal=True, block=False, save=True)
        p.close()
