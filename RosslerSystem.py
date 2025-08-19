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


class RosslerSystem(DivergentSystem):
    def __init__(self):
        # Common helpers shared by both kernels
        _helpers = r"""
        // ---- helpers: double precision dynamics + lattice mapping ----
        __forceinline__ __device__ inline void rossler_derivs_d(
            double X, double Y, double Z,
            double a, double b, double c,
            double &dX, double &dY, double &dZ)
        {
            // Rossler system
            //   dX/dt = -Y - Z
            //   dY/dt =  X + a*Y
            //   dZ/dt =  b + Z*(X - c)
            dX = -Y - Z;
            dY =  X + a*Y;
            dZ =  b + Z*(X - c);
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
                const float a, const float b, const float c,
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
                    rossler_derivs_d(xd, yd, zd, (double)a, (double)b, (double)c, k1x, k1y, k1z);

                    double x2 = xd + 0.5*dt*k1x;
                    double y2 = yd + 0.5*dt*k1y;
                    double z2 = zd + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    rossler_derivs_d(x2, y2, z2, (double)a, (double)b, (double)c, k2x, k2y, k2z);

                    double x3 = xd + 0.5*dt*k2x;
                    double y3 = yd + 0.5*dt*k2y;
                    double z3 = zd + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    rossler_derivs_d(x3, y3, z3, (double)a, (double)b, (double)c, k3x, k3y, k3z);

                    double x4 = xd + dt*k3x;
                    double y4 = yd + dt*k3y;
                    double z4 = zd + dt*k3z;
                    double k4x, k4y, k4z;
                    rossler_derivs_d(x4, y4, z4, (double)a, (double)b, (double)c, k4x, k4y, k4z);

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
                const float a, const float b, const float c,
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
                const double dt = 0.005;
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
                    rossler_derivs_d(x1, y1, z1, (double)a, (double)b, (double)c, k1x, k1y, k1z);
                    double x1_2 = x1 + 0.5*dt*k1x;
                    double y1_2 = y1 + 0.5*dt*k1y;
                    double z1_2 = z1 + 0.5*dt*k1z;
                    double k2x, k2y, k2z;
                    rossler_derivs_d(x1_2, y1_2, z1_2, (double)a, (double)b, (double)c, k2x, k2y, k2z);
                    double x1_3 = x1 + 0.5*dt*k2x;
                    double y1_3 = y1 + 0.5*dt*k2y;
                    double z1_3 = z1 + 0.5*dt*k2z;
                    double k3x, k3y, k3z;
                    rossler_derivs_d(x1_3, y1_3, z1_3, (double)a, (double)b, (double)c, k3x, k3y, k3z);
                    double x1_4 = x1 + dt*k3x;
                    double y1_4 = y1 + dt*k3y;
                    double z1_4 = z1 + dt*k3z;
                    double k4x, k4y, k4z;
                    rossler_derivs_d(x1_4, y1_4, z1_4, (double)a, (double)b, (double)c, k4x, k4y, k4z);
                    x1 += (dt/6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                    y1 += (dt/6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                    z1 += (dt/6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);

                    // ===== Integrate second system =====
                    rossler_derivs_d(x2, y2, z2, (double)a, (double)b, (double)c, k1x, k1y, k1z);
                    double x2_2 = x2 + 0.5*dt*k1x;
                    double y2_2 = y2 + 0.5*dt*k1y;
                    double z2_2 = z2 + 0.5*dt*k1z;
                    rossler_derivs_d(x2_2, y2_2, z2_2, (double)a, (double)b, (double)c, k2x, k2y, k2z);
                    double x2_3 = x2 + 0.5*dt*k2x;
                    double y2_3 = y2 + 0.5*dt*k2y;
                    double z2_3 = z2 + 0.5*dt*k2z;
                    rossler_derivs_d(x2_3, y2_3, z2_3, (double)a, (double)b, (double)c, k3x, k3y, k3z);
                    double x2_4 = x2 + dt*k3x;
                    double y2_4 = y2 + dt*k3y;
                    double z2_4 = z2 + dt*k3z;
                    rossler_derivs_d(x2_4, y2_4, z2_4, (double)a, (double)b, (double)c, k4x, k4y, k4z);
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
        self._variables = ["a", "b", "c"]


if __name__ == "__main__":
    # Test harness
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="RosslerSystem test harness: set control variables and start point via CLI or use defaults.")
    parser.add_argument("--a", type=float, default=0.2, help="Rossler parameter a. Default: 0.2")
    parser.add_argument("--b", type=float, default=0.2, help="Rossler parameter b. Default: 0.2")
    parser.add_argument("--c", type=float, default=5.7, help="Rossler parameter c. Default: 5.7")
    parser.add_argument("--origin", "-o", nargs=3, type=float,
                        default=(1.0, 0.0, 0.0), metavar=("X0", "Y0", "Z0"),
                        help="Initial position (x0, y0, z0). Default: 1, 0, 0")
    parser.add_argument("--steps", "-s", type=int, default=20000, help="Integration steps. Default: 20000")
    args = parser.parse_args()

    # Create instance and set control variables
    instance = RosslerSystem()
    instance.config = (args.a, args.b, args.c)

    # print(f"{instance.__class__.__name__} variables {instance.variables} are {instance.config}")

    # # Single trajectory demo
    # origin = tuple(args.origin)
    # steps = args.steps
    # print(f"origin {origin}")
    # print(f"iterate for {steps} steps")
    # traj = instance.get_trajectory(origin, steps)
    # print(f"trajectory length {len(traj)}")
    # instance.plot_trajectory(traj, origin, block=True)
    # exit()

    # # Heatmap demo over a typical Rossler region
    # steps = 1500

    # # divs = 255

    # divs = 1023
    # instance.base = (-15.0, -10.0, 7.0)
    # instance.limit = (-5.0, 0.0, 7.0)

    # instance.calc_stride(divs)

    # pkl = True
    # if pkl:
    #     # Calculate and save heatmap...
    #     heatmap = instance.get_heatmap(perturbation=(1e-8, 0, 0), steps=steps)
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl", 'wb') as f:
    #         pickle.dump(heatmap, f)
    # if not pkl:
    #     # Load heatmap we saved earlier...
    #     with open(f"{instance.__class__.__name__}-volume{divs}.pkl",'rb') as f:
    #         heatmap = pickle.load(f)

    # p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=pkl, save=not pkl)
    # p.close()
    # exit(0)

    # divs = 1024
    # for x in range(-100, 201, 5):
    #     instance.base = (x / 10.0, -15.0, -5.0)
    #     instance.limit = (x / 10.0, 5.0, 0.0)
    #     instance.calc_stride(divs)
    #     print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")
    #     heatmap = instance.get_heatmap(perturbation = (1e-8, 0, 0), steps=steps)
    #     # instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=True, save=False)
    #     p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=False, save=True)
    #     p.close()

    # -- Content
    #
    # 3D heatmap
    #
    steps = 1500
    divs = 255
    instance.base = (-15.0, -10.0, 0.0)
    instance.limit = (-5.0, 0.0, 10.0)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap
    #
    steps = 2500
    divs = 1023
    z = 5.0
    instance.base = (-15.0, -10.0, z)
    instance.limit = (-5.0, 0.0, z)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap
    #
    steps = 2500
    divs = 1023
    y = -5.0
    instance.base = (-15.0, y, 0.0)
    instance.limit = (-5.0, y, 10.0)
    instance.calc_stride(divs)
    heatmap = instance.get_heatmap(steps=steps)
    p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=False, save=True)
    p.close()
    # exit()

    # -- Content
    #
    # 2D heatmap animation frames
    #
    instance.config = (0.2, 0.15, 5.7)
    steps = 3000
    divs = 1279
    # for x in range(-100, 201, 5):
    for x in range(-500, 20001, 100):
        instance.base = (0.0, x / 100.0, -50.0)
        instance.limit = (30.0, x / 100.0, 0.0)
        # instance.base = (x / 10.0, -15.0, -5.0)
        # instance.limit = (x / 10.0, 5.0, 0.0)
        instance.calc_stride(divs)
        print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

        heatmap = instance.get_heatmap(steps=steps)
        # p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=True, save=False)
        p = instance.plot_heatmap(heatmap, theme="PuRd_r", equal=True, block=False, save=True)
        p.close()
