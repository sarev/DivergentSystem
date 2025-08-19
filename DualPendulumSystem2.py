#!/usr/bin/env python3
#
# DualPendulumSystem2.py
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
#
# Implements an ideal Dual Pendulum and tracks the motion of the second bob.

from DivergentSystem import DivergentSystem
import numpy as np
import argparse


class DualPendulumSystem2(DivergentSystem):
    def __init__(self):

        _helpers = r"""
        // ---- helpers: double precision, no lambdas ----
        __forceinline__ __device__ inline double wrap_pi_d(double a)
        {
            const double PI     = 3.1415926535897932384626433832795;
            const double TWO_PI = 6.283185307179586476925286766559;
            a = fmod(a + PI, TWO_PI);
            if (a < 0.0) a += TWO_PI;
            return a - PI;
        }

        // Map a linear lattice index to (x,y,z) using base coords, stride, and sizes
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

        // Equal-mass (m1=m2=1) frictionless double pendulum derivatives in double.
        // Trig is evaluated on wrapped angles ONLY. State remains cumulative.
        __forceinline__ __device__ inline void dp_derivs_d(
            double T1, double T2, double W1, double W2,
            double L1, double L2, double g,
            double &dT1, double &dT2, double &dW1, double &dW2)
        {
            const double t1r = wrap_pi_d(T1);
            const double t2r = wrap_pi_d(T2);
            const double dlt = t1r - t2r;

            double s1, c1, s2, c2, sD, cD;
            sincos(t1r, &s1, &c1);
            sincos(t2r, &s2, &c2);
            sincos(dlt,  &sD, &cD);

            const double c2D  = cos(2.0 * dlt);
            double denom = 3.0 - c2D;            // equal masses
            if (denom < 1e-12) denom = 1e-12;    // guard near singular configs

            dT1 = W1;
            dT2 = W2;

            const double num1 = -3.0 * g * s1
                                - g * sin(t1r - 2.0 * t2r)
                                - 2.0 * sD * (W2*W2 * L2 + W1*W1 * L1 * cD);
            dW1 = num1 / (L1 * denom);

            const double num2 = 2.0 * sD * (2.0 * W1*W1 * L1 + 2.0 * g * c1 + W2*W2 * L2 * cD);
            dW2 = num2 / (L2 * denom);
        }

        // Cartesian position of the second bob for absolute-angle convention.
        __forceinline__ __device__ inline void pos2_d(double th1, double th2, double L1, double L2,
                                      double &x2, double &y2)
        {
            const double t1r = wrap_pi_d(th1);
            const double t2r = wrap_pi_d(th2);

            double s1, c1, s2, c2;
            sincos(t1r, &s1, &c1);
            sincos(t2r, &s2, &c2);

            const double x1 =  L1 * s1;
            const double y1 = -L1 * c1;

            x2 = x1 + L2 * s2;
            y2 = y1 - L2 * c2;
        }
        """

        system_kernel = rf"""
        extern "C" {{

        {_helpers}

        // system_kernel(
        //     float *trajectory,                   // array to store the system 'trajectory'
        //     float x, y, z,                       // starting coordinate (theta1_0, theta2_0, z ignored)
        //     const float omega1, omega2_0, r, g,  // control variables: initial angular velocities, length ratio r=L2/L1, gravity g
        //     const int steps                      // number of integration steps
        // )
        //
        // Integrates the ideal, frictionless, equal-mass double pendulum and emits the Cartesian
        // path of the second bob (x2, y2, 0) for each step.
        //
        // Notes:
        // - Control variables are in physical units (rad/s, unitless, m/s²).
        // - z is ignored but preserved for API shape.
        // - Uses double precision for RK4 integration, float for outputs.
        // - Register usage is reduced by reusing stage variables and immediately applying contributions.
        //
        __global__ void system_kernel(
            float *trajectory,
            float x, float y, float z,
            const float omega1, const float omega2_0, const float r, const float g,
            const int steps)
        {{
            (void)z; // unused

            // Pendulum geometry constants
            const double L1 = 1.0;
            const double L2 = fmax((double)r * L1, 1e-12);
            const double dt = 0.01;

            // Initial state
            double t1 = (double)x;
            double t2 = (double)y;
            double w1 = (double)omega1;
            double w2 = (double)omega2_0;

            // RK4 slope accumulators (only two sets of vars live at a time)
            double at1, at2, aw1, aw2; // temporary derivatives

            for (int i = 0; i < steps; ++i) {{

                // Output Cartesian path of the second bob
                double xb, yb;
                pos2_d(t1, t2, L1, L2, xb, yb);
                trajectory[i*3 + 0] = (float)xb;
                trajectory[i*3 + 1] = (float)yb;
                trajectory[i*3 + 2] = 0.0f;

                // RK4 integration in-place, reusing variables

                // --- Stage 1 ---
                double k_t1, k_t2, k_w1, k_w2;
                dp_derivs_d(t1, t2, w1, w2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);

                double acc_t1 = k_t1;
                double acc_t2 = k_t2;
                double acc_w1 = k_w1;
                double acc_w2 = k_w2;

                // --- Stage 2 ---
                dp_derivs_d(
                    t1 + 0.5*dt*k_t1,
                    t2 + 0.5*dt*k_t2,
                    w1 + 0.5*dt*k_w1,
                    w2 + 0.5*dt*k_w2,
                    L1, L2, (double)g, at1, at2, aw1, aw2
                );
                acc_t1 += 2.0 * at1;
                acc_t2 += 2.0 * at2;
                acc_w1 += 2.0 * aw1;
                acc_w2 += 2.0 * aw2;

                // --- Stage 3 ---
                dp_derivs_d(
                    t1 + 0.5*dt*at1,
                    t2 + 0.5*dt*at2,
                    w1 + 0.5*dt*aw1,
                    w2 + 0.5*dt*aw2,
                    L1, L2, (double)g, at1, at2, aw1, aw2
                );
                acc_t1 += 2.0 * at1;
                acc_t2 += 2.0 * at2;
                acc_w1 += 2.0 * aw1;
                acc_w2 += 2.0 * aw2;

                // --- Stage 4 ---
                dp_derivs_d(
                    t1 + dt*at1,
                    t2 + dt*at2,
                    w1 + dt*aw1,
                    w2 + dt*aw2,
                    L1, L2, (double)g, at1, at2, aw1, aw2
                );
                acc_t1 += at1;
                acc_t2 += at2;
                acc_w1 += aw1;
                acc_w2 += aw2;

                // Update state
                t1 += (dt / 6.0) * acc_t1;
                t2 += (dt / 6.0) * acc_t2;
                w1 += (dt / 6.0) * acc_w1;
                w2 += (dt / 6.0) * acc_w2;

                if (!isfinite(t1) || !isfinite(t2) || !isfinite(w1) || !isfinite(w2)) {{
                    break;
                }}
            }}
        }}

        }} // extern "C"
        """

        distance_kernel = rf"""
        // distance_kernel(
        //   const float omega1, omega2_0, r, g,   // control variables
        //   const float px, py, pz,                 // perturbation (pz ignored here)
        //   const int   steps,                      // integration steps
        //   const float x0, y0, z0,                 // lattice base
        //   const float sx, sy, sz,                 // lattice strides
        //   const int   nx, ny, nz,                 // lattice sizes
        //   const int   offset, n_items,            // linear window [offset, offset+n_items)
        //   float* out                               // FTLE-like growth rate per lattice point
        // )
        //
        // Computes a Benettin-style FTLE estimate by integrating two nearby trajectories of the
        // equal-mass double pendulum from each lattice coordinate (xf, yf, zf). Angles remain
        // cumulative; trig is evaluated on wrapped angles inside dp_derivs_d.
        //
        // Notes:
        // - Uses double precision for integration state, float for I/O.
        // - Reuses RK4 temporaries to reduce live registers.
        // - __launch_bounds__(128,1) is declared; launch with <=128 threads per block.
        //
        extern "C" {{

        {_helpers}

        __global__ void distance_kernel(
            const float omega1, const float omega2_0, const float r, const float g,
            const float px, const float py, const float pz,
            const int   steps,
            const float x0, const float y0, const float z0,
            const float sx, const float sy, const float sz,
            const int   nx, const int ny, const int nz,
            const int   offset, const int n_items,
            float* __restrict__ out)
        {{
            (void)pz; // unused in this system

            // Linear index within this worker's window
            const int i_local = blockIdx.x * blockDim.x + threadIdx.x;
            if (i_local >= n_items) return;

            // Global lattice index and coordinates
            const int idx = offset + i_local;
            float xf, yf, zf;
            lattice_coords_f32(idx, x0, y0, z0, sx, sy, sz, nx, ny, nz, xf, yf, zf);
            (void)zf; // z not used; API preserved

            // Pendulum constants (double for stability)
            const double L1 = 1.0;
            const double L2 = fmax((double)r * L1, 1e-12);
            const double dt = 0.01;

            // Benettin settings
            const int    REORTHO       = 37;     // prime cadence helps avoid resonance
            const double DELTA0_MIN    = 1e-20;
            const double DELTA_MIN     = 1e-30;
            const double ACCUM_MAX_ABS = 1e12;

            // Two trajectories (cumulative angles)
            double t1_a = (double)xf;
            double t2_a = (double)yf;
            double w1_a = (double)omega1;
            double w2_a = (double)omega2_0;

            double t1_b = t1_a + (double)px;
            double t2_b = t2_a + (double)py;
            double w1_b = (double)omega1;
            double w2_b = (double)omega2_0;

            // Baseline angular offset for renormalisation
            double dxa0 = t1_b - t1_a;
            double dya0 = t2_b - t2_a;
            double delta0_ang = sqrt(dxa0*dxa0 + dya0*dya0);
            if (!(delta0_ang > 0.0)) {{
                t1_b = t1_a + DELTA0_MIN; t2_b = t2_a;
                delta0_ang = DELTA0_MIN;
                dxa0 = DELTA0_MIN; dya0 = 0.0;
            }} else if (delta0_ang < DELTA0_MIN) {{
                const double s = DELTA0_MIN / delta0_ang;
                dxa0 *= s; dya0 *= s;
                t1_b = t1_a + dxa0; t2_b = t2_a + dya0;
                delta0_ang = DELTA0_MIN;
            }}

            // Baseline Cartesian separation using second-bob positions
            double xa, ya, xb, yb;
            pos2_d(t1_a, t2_a, L1, L2, xa, ya);
            pos2_d(t1_b, t2_b, L1, L2, xb, yb);
            double dxp0 = xb - xa, dyp0 = yb - ya;
            double delta0_pos = sqrt(dxp0*dxp0 + dyp0*dyp0);
            if (!(delta0_pos > 0.0)) delta0_pos = DELTA0_MIN;

            double accum_log = 0.0;
            int since_reorth = 0;
            int i = 0;

            // Reused RK4 temporaries to keep register footprint low
            double k_t1, k_t2, k_w1, k_w2;
            double tt1,  tt2,  ww1,  ww2;
            double sum_t1, sum_t2, sum_w1, sum_w2;

            for (i = 0; i < steps; ++i) {{

                // ===== Trajectory A =====
                // Stage 1
                dp_derivs_d(t1_a, t2_a, w1_a, w2_a, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_a + 0.5*dt*k_t1;
                tt2 = t2_a + 0.5*dt*k_t2;
                ww1 = w1_a + 0.5*dt*k_w1;
                ww2 = w2_a + 0.5*dt*k_w2;
                sum_t1 = k_t1; sum_t2 = k_t2; sum_w1 = k_w1; sum_w2 = k_w2;

                // Stage 2
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_a + 0.5*dt*k_t1;
                tt2 = t2_a + 0.5*dt*k_t2;
                ww1 = w1_a + 0.5*dt*k_w1;
                ww2 = w2_a + 0.5*dt*k_w2;
                sum_t1 += 2.0*k_t1; sum_t2 += 2.0*k_t2; sum_w1 += 2.0*k_w1; sum_w2 += 2.0*k_w2;

                // Stage 3
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_a + dt*k_t1;
                tt2 = t2_a + dt*k_t2;
                ww1 = w1_a + dt*k_w1;
                ww2 = w2_a + dt*k_w2;
                sum_t1 += 2.0*k_t1; sum_t2 += 2.0*k_t2; sum_w1 += 2.0*k_w1; sum_w2 += 2.0*k_w2;

                // Stage 4
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                sum_t1 += k_t1; sum_t2 += k_t2; sum_w1 += k_w1; sum_w2 += k_w2;

                // Update A
                t1_a += (dt/6.0) * sum_t1;
                t2_a += (dt/6.0) * sum_t2;
                w1_a += (dt/6.0) * sum_w1;
                w2_a += (dt/6.0) * sum_w2;

                // ===== Trajectory B =====
                // Stage 1
                dp_derivs_d(t1_b, t2_b, w1_b, w2_b, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_b + 0.5*dt*k_t1;
                tt2 = t2_b + 0.5*dt*k_t2;
                ww1 = w1_b + 0.5*dt*k_w1;
                ww2 = w2_b + 0.5*dt*k_w2;
                sum_t1 = k_t1; sum_t2 = k_t2; sum_w1 = k_w1; sum_w2 = k_w2;

                // Stage 2
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_b + 0.5*dt*k_t1;
                tt2 = t2_b + 0.5*dt*k_t2;
                ww1 = w1_b + 0.5*dt*k_w1;
                ww2 = w2_b + 0.5*dt*k_w2;
                sum_t1 += 2.0*k_t1; sum_t2 += 2.0*k_t2; sum_w1 += 2.0*k_w1; sum_w2 += 2.0*k_w2;

                // Stage 3
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                tt1 = t1_b + dt*k_t1;
                tt2 = t2_b + dt*k_t2;
                ww1 = w1_b + dt*k_w1;
                ww2 = w2_b + dt*k_w2;
                sum_t1 += 2.0*k_t1; sum_t2 += 2.0*k_t2; sum_w1 += 2.0*k_w1; sum_w2 += 2.0*k_w2;

                // Stage 4
                dp_derivs_d(tt1, tt2, ww1, ww2, L1, L2, (double)g, k_t1, k_t2, k_w1, k_w2);
                sum_t1 += k_t1; sum_t2 += k_t2; sum_w1 += k_w1; sum_w2 += k_w2;

                // Update B
                t1_b += (dt/6.0) * sum_t1;
                t2_b += (dt/6.0) * sum_t2;
                w1_b += (dt/6.0) * sum_w1;
                w2_b += (dt/6.0) * sum_w2;

                // Divergence checks
                if (!isfinite(t1_a) || !isfinite(t2_a) || !isfinite(w1_a) || !isfinite(w2_a) ||
                    !isfinite(t1_b) || !isfinite(t2_b) || !isfinite(w1_b) || !isfinite(w2_b)) {{
                    break;
                }}

                ++since_reorth;

                // Benettin accumulation and renormalisation
                if (since_reorth == REORTHO) {{
                    double xa1, ya1, xb1, yb1;
                    pos2_d(t1_a, t2_a, L1, L2, xa1, ya1);
                    pos2_d(t1_b, t2_b, L1, L2, xb1, yb1);
                    const double dxp = xb1 - xa1, dyp = yb1 - ya1;
                    const double delta_pos = fmax(sqrt(dxp*dxp + dyp*dyp), DELTA_MIN);

                    accum_log += log(delta_pos / delta0_pos);

                    // Renormalise in angle space back to baseline angular offset
                    const double dxa = t1_b - t1_a;
                    const double dya = t2_b - t2_a;
                    const double norm_ang = fmax(sqrt(dxa*dxa + dya*dya), DELTA_MIN);
                    const double s = delta0_ang / norm_ang;

                    t1_b = t1_a + dxa * s;
                    t2_b = t2_a + dya * s;

                    // Refresh Cartesian baseline for next segment
                    pos2_d(t1_a, t2_a, L1, L2, xa1, ya1);
                    pos2_d(t1_b, t2_b, L1, L2, xb1, yb1);
                    const double dxp2 = xb1 - xa1, dyp2 = yb1 - ya1;
                    delta0_pos = fmax(sqrt(dxp2*dxp2 + dyp2*dyp2), DELTA_MIN);

                    since_reorth = 0;
                }}

                if (!isfinite(accum_log) || fabs(accum_log) > ACCUM_MAX_ABS) {{
                    break;
                }}
            }}

            // Final partial contribution if we ended mid-segment
            if (since_reorth > 0) {{
                double xa1, ya1, xb1, yb1;
                pos2_d(t1_a, t2_a, L1, L2, xa1, ya1);
                pos2_d(t1_b, t2_b, L1, L2, xb1, yb1);
                const double dxp = xb1 - xa1, dyp = yb1 - ya1;
                const double delta_pos = fmax(sqrt(dxp*dxp + dyp*dyp), DELTA_MIN);
                accum_log += log(delta_pos / delta0_pos);
            }}

            const double t_total = dt * (double)(i + 1);
            out[i_local] = (t_total > 0.0 && isfinite(t_total)) ? (float)(accum_log / t_total) : 0.0f;
        }}

        }} // extern "C"
        """

        super().__init__(system_kernel, distance_kernel)
        # Variables are: initial joint angular velocities, L2/L1 ratio, and gravity
        self._variables = ["omega1", "omega2", "length_ratio", "gravity"]


if __name__ == "__main__":
    # Test harness...

    parser = argparse.ArgumentParser(
        description="DualPendulumSystem2 test harness: set control variables and start point via CLI or use defaults."
    )
    parser.add_argument("--omega1", "-w1", type=float, default=0.0,
                        help="Initial angular velocity of joint 1 (rad/s). Default: 0.0")
    parser.add_argument("--omega2", "-w2", type=float, default=0.0,
                        help="Initial angular velocity of joint 2 (rad/s). Default: 0.0")
    parser.add_argument("--length_ratio", "--r", "-r", dest="r", type=float, default=1.0,
                        help="Length ratio L2/L1. Default: 1.0")
    parser.add_argument("--gravity", "-g", type=float, default=9.80665,
                        help="Gravity (m/s^2). Default: 9.80665")
    parser.add_argument("--origin", "-o", nargs=3, type=float,
                        default=(1.7627825445142729, 2.007128639793479, 0.0),
                        metavar=("THETA1_0", "THETA2_0", "Z"),
                        help="Initial positions (theta1, theta2, z). Default: 1.76278, 2.00713, 0.0")
    parser.add_argument("--steps", "-s", type=int, default=20000,
                        help="Number of integration steps. Default: 20000")

    args = parser.parse_args()


    # Angles are in radians. We default to an interesting chaotic setup.
    instance = DualPendulumSystem2()

    # Control variables (omega1, omega2, r, g)
    instance.config = (args.omega1, args.omega2, args.r, args.gravity)

    # Example starting angles (theta1, theta2, z ignored) and steps
    origin = tuple(args.origin)
    steps = args.steps

    print(f"{instance.__class__.__name__} variables {instance.variables} are {instance.config}")
    print(f"origin {origin} (theta1, theta2, z unused)")
    print(f"iterate for {steps} steps")

    # Generate and show the single trajectory (theta1, theta2) as (x, y). Third channel is 0.
    trajectory = instance.get_trajectory(origin, steps)
    print(f"trajectory length {len(trajectory)}")
    instance.plot_trajectory(trajectory, origin, block=False)

    # Generate a chaos heatmap over a 3D parameter grid in angle-angle-z space.
    # We scan theta1_0 and theta2_0; the third axis is a dummy to preserve the API.
    steps = 1000
    instance.base = (-1.1*np.pi, -1.1*np.pi, 0.0)
    instance.limit = (1.1*np.pi, 1.1*np.pi, 0.0)
    instance.calc_stride(256)
    print(f"steps {steps} base {instance.base} limit {instance.limit} stride {instance.stride} config {instance.config}")

    heatmap = instance.get_heatmap(steps=steps)
    instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=True)
    exit()

    # for steps in [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
    #     print(f"Creating plot for {steps} steps...")

    #     # Compute the heatmap for this system using the specified configuration and state space
    #     heatmap = instance.get_heatmap(steps=steps)

    #     # Display a plot of the heatmap
    #     instance.plot_heatmap(heatmap, theme="inferno", equal=True, block=False, save=True)

    # steps = 750
    # for res in [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
    #     instance.calc_stride(res)

    #     print(f"Creating plot for {res}x{res}...")

    #     # Compute the heatmap for this system using the specified configuration and state space
    #     heatmap = instance.get_heatmap(steps=steps)

    #     # Display a plot of the heatmap
    #     instance.plot_heatmap(heatmap, theme="viridis", equal=True, block=False, save=True)

    steps = 550

    for ratio in np.arange(0.5, 10.0 + 0.0001, np.pi / 20):
        instance.config = (ratio, args.omega2, 1, args.gravity)

        print(f"Creating plot for omega1 {ratio}...")

        # Compute the heatmap for this system using the specified configuration and state space
        heatmap = instance.get_heatmap(steps=steps)

        # Display a plot of the heatmap
        instance.plot_heatmap(heatmap, theme="YlGnBu", equal=True, block=False, save=True)



    # for ratio in np.arange(0.5, 10.0 + 0.0001, 0.3141593):
    #     instance.config = (args.omega1, ratio, 1, args.gravity)

    #     print(f"Creating plot for omega2 {ratio}...")

    #     # Compute the heatmap for this system using the specified configuration and state space
    #     heatmap = instance.get_heatmap(steps=steps)

    #     # Display a plot of the heatmap
    #     instance.plot_heatmap(heatmap, theme="gist_heat", equal=True, block=False, save=True)

    # for ratio in np.arange(0.5, 10.0 + 0.0001, 0.3):
    #     instance.config = (args.omega1, args.omega2, ratio, args.gravity)

    #     print(f"Creating plot for length ratio {ratio}...")

    #     # Compute the heatmap for this system using the specified configuration and state space
    #     heatmap = instance.get_heatmap(steps=steps)

    #     # Display a plot of the heatmap
    #     instance.plot_heatmap(heatmap, theme="gist_earth", equal=True, block=False, save=True)

    print("Finished!")
