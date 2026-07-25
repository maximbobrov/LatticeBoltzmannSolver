# LBM — a D2Q9 Lattice Boltzmann solver (CPU + CUDA)

Interactive 2D incompressible-flow solver based on the Lattice Boltzmann
Method: five classic scenarios, each with
a quantitative reference (exact solution or published benchmark), live
solver-vs-reference plots in the GUI, and a set of command-line gates that
validate every ingredient of the discretization. The CPU and GPU cores execute
**literally the same code**, so they can be cross-checked to machine
precision.

Kármán vortex street:
<img width="1919" height="1026" alt="image" src="https://github.com/user-attachments/assets/438407ae-461f-4057-890b-e052b423c00b" />

Porous medium:
<img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/9615eecd-dd9e-4a14-9979-3564fd6ff024" />

## 1. Physics and mathematics

### 1.1 The idea of LBM

Instead of discretizing the Navier–Stokes equations directly, LBM evolves a
set of particle distribution functions `f_i(x, t)` on a regular lattice. Each
`f_i` is the density of particles moving with one of a small set of discrete
velocities `e_i`. One time step consists of two operations:

1. **Collision** — at every node the distributions relax toward a local
   equilibrium (BGK, single relaxation time τ):

   ```
   f_i*(x,t) = f_i(x,t) − (1/τ) [ f_i(x,t) − f_i^eq(ρ,u) ]
   ```

2. **Streaming** — post-collision values hop to the neighbouring node along
   their velocity:

   ```
   f_i(x + e_i, t+1) = f_i*(x, t)
   ```

A Chapman–Enskog expansion shows that in the low-Mach limit this system
solves the weakly compressible Navier–Stokes equations with kinematic
viscosity

```
ν = c_s² (τ − 1/2),      c_s² = 1/3   (lattice speed of sound)
```

so `τ = 3ν + 1/2`. Everything is in *lattice units*: Δx = Δt = 1, density
ρ ≈ 1. The physical regime is set purely by the Reynolds number
`Re = U·L/ν`, where `U = u_lat` (the velocity scale, kept ≤ 0.1 so that
compressibility errors ~O(Ma²) stay small; Ma = u_lat·√3) and `L` is the
scenario's characteristic length in cells.

### 1.2 The D2Q9 lattice

Nine velocities: rest, four axis directions, four diagonals.

```
   6  2  5        w_0     = 4/9
   3  0  1        w_1..4  = 1/9    (axis)
   7  4  8        w_5..8  = 1/36   (diagonal)
```

Equilibrium (second-order truncated Maxwellian):

```
f_i^eq = w_i ρ [ 1 + 3(e_i·u) + 4.5(e_i·u)² − 1.5 u² ]
```

Macroscopic fields are moments of `f`:

```
ρ = Σ_i f_i,        ρu = Σ_i f_i e_i  ( + F/2 with forcing, see below )
```

The moments of `f^eq` reproduce mass, momentum and the Euler stress exactly
(`Σ f^eq e e = ρ(c_s² I + uu)`) — this is verified numerically in the first
self-test.

### 1.3 Body force — Guo forcing

For the force-driven channel (Poiseuille) a constant body force `F = (g_x,0)`
is applied with the scheme of Guo et al. (2002), which is second-order
accurate and free of spurious `τ`-dependent terms:

```
u        = ( Σ f_i e_i + F/2 ) / ρ                       (velocity shift)
F_i      = w_i (1 − 1/(2τ)) [ 3(e_i − u) + 9(e_i·u) e_i ] · F
f_i^new  = f_i − (1/τ)(f_i − f_i^eq) + F_i
```

The channel force is chosen as `g_x = 8νU/H²` so the analytic peak velocity
equals exactly `u_lat`.

### 1.4 Boundary conditions

| Cell type   | Rule |
|-------------|------|
| `CT_FLUID`  | full stream + collide |
| `CT_SOLID`  | **halfway bounce-back**: a population about to leave a fluid cell toward a wall cell returns reversed in the same step: `f_ī(x_f, t+1) = f_i*(x_f, t)`. The no-slip plane sits **half a cell** beyond the last fluid node — second-order accurate wall placement. Hence an N-cell box has an effective size of N−2 cells between wall planes. |
| `CT_MOVING` | bounce-back plus the wall-momentum correction `+ 6 w_i (e_i · u_w)` (i.e. `2 w_i ρ₀ (e_i·u_w)/c_s²` with ρ₀ = 1). Drives the cavity/Couette lid. |
| `CT_INLET`  | equilibrium inlet: `f_i = f_i^eq(ρ=1, u_in)` imposed every step. |
| `CT_OUTLET` | first-order outflow: the outlet column copies the full population set of its upstream neighbour (zero-gradient in f). |

Periodic directions (Taylor–Green both axes, channels in x) are handled by
index wrapping during streaming; non-periodic edges are always fenced by a
one-cell layer of boundary cells.

### 1.5 Forces on immersed bodies — momentum exchange

The drag/lift on the cylinder is accumulated over all fluid→solid links: each
bounced population transfers momentum `2 f_i* e_i` per step (static wall).
Coefficients use the standard normalization `Cd = F_x / (½ ρ₀ U² D)`,
`Cl = F_y / (½ ρ₀ U² D)`. The Strouhal number is measured from the mean
period between upward zero crossings of the `Cl(t)` history:
`St = D / (U·T)`.

### 1.6 Algorithmic form: fused pull step

The implementation uses the *pull* formulation, fused into a single pass per
cell (better for GPUs — one kernel, coalesced reads/writes, no separate
streaming sweep):

```
for every cell x:
    gather  f_i  from  x − e_i          (bounce-back resolved during gather)
    ρ, u    from the gathered set        (+ F/2 correction)
    collide (BGK + Guo term)
    write the post-collision set to the second buffer
swap buffers (ping-pong)
```

Storage is SoA: `f[i·NX·NY + y·NX + x]`, two buffers of 9 double-precision
fields each.

---

## 2. Code architecture — what runs where

```
lattice.h        THE physics. lbmCellUpdate() / lbmFeq() / lbmMacroAt(),
                 written once, marked __host__ __device__.
lbm_core.cpp     CPU core: state arrays, OpenMP stepping loop, macro fields,
                 diagnostics (mass, energy, nu_eff, Cd/Cl/St ring buffer).
scenarios.cpp    applyScenario() (derived parameters) + buildFlags() (cell-
                 type maps: walls, lid, inlet/outlet, cylinder mask).
validation.cpp   reference solutions, Ghia tables, profile panels, self-test
                 gates, CPU-vs-GPU check, cavity/cylinder benchmark gates,
                 MLUPS benchmark.
viz.cpp          freeglut/OpenGL: field as one texture (98th-percentile
                 robust color scaling), velocity arrows, color bar, HUD,
                 key legend, solver-vs-reference panels, Cl(t) strip.
main_app.cpp     GLUT loop, keyboard, CPU/GPU switching, CLI dispatch,
                 CUDA stubs for non-CUDA builds.
cuda_kernels.h   plain-C++ GPU interface (no CUDA types leak out).
cuda_kernels.cu  CUDA core: thin per-cell kernel wrappers around lattice.h.
```

### The single-source CPU/GPU design

`lbmCellUpdate()` in [lattice.h](lattice.h) is the entire numerical method.
It is compiled twice:

- **CPU**: an OpenMP `parallel for` over rows calls it per cell
  (`lbm_core.cpp`).
- **GPU**: a CUDA kernel launches one thread per cell and calls *the same
  function* (`cuda_kernels.cu`); `LBM_HD` expands to
  `__host__ __device__ __forceinline__` under nvcc and to `inline` otherwise.

Consequences:

- there is exactly one implementation of the physics to debug;
- `--gpucheck` demands the two cores agree: after 200 steps of the cavity the
  measured difference is `max|f_gpu − f_cpu| = 8.3·10⁻¹⁶` — pure FMA
  roundoff, i.e. bit-level equivalence of the algorithm.

### Division of labour in GPU mode

Everything hot stays resident on the device: the two ping-pong `f` buffers
and the flag field are uploaded once (`gpuLbmInit`), then `gpuLbmSteps(n)`
runs n fused kernels back to back. Per display frame only the three macro
fields (ρ, u_x, u_y — computed by a device kernel) are downloaded, 3× less
traffic than pulling the 9 distributions. For the cylinder, a small
momentum-exchange kernel with atomic adds runs per step over a box around the
body and the per-step force history is downloaded once per batch. The full
`f` state is downloaded only when switching GPU→CPU (the toggle is seamless
mid-run) or for `--gpucheck`.

The CPU path exists on its own merit (runs without any CUDA toolkit — the
`.pro` file detects nvcc and falls back to stubs) and as the reference for
the GPU. Measured on a 128² cavity: **88 MLUPS** CPU (OpenMP) vs
**599 MLUPS** GPU (RTX 5070 Laptop; kernel-launch bound at this tiny size —
the gap widens on larger grids).

---

## 3. Scenarios and validation

Validation is the point of this project. Every scenario isolates one
ingredient of the method and has a quantitative reference; the GUI overlays
the live solver profile (yellow) on the reference (cyan curve / white “+”
scatter) with the relative L2 error and a GOOD/FAIR/POOR verdict.

| # | Scenario (key `O`) | What it isolates | Reference | Result |
|---|--------------------|------------------|-----------|--------|
| 1 | **Taylor–Green vortex** — periodic decaying vortex array | collision + streaming, *no walls at all* | exact transient `u = −U cos kx sin ky · e^{−2νk²t}`; effective viscosity measured from kinetic-energy decay `E ~ e^{−4νk²t}` | ν_eff error **0.07 %** |
| 2 | **Poiseuille** — force-driven channel | bounce-back walls + Guo forcing | exact parabola `u = 4U s(1−s)`, wall-to-wall coordinate `s` | L2 **0.10 %** |
| 3 | **Couette** — lid-dragged channel | moving-wall bounce-back | exact linear profile `u = U s` | L2 **0.03 %** |
| 4 | **Lid-driven cavity** | everything together, steady nonlinear flow | Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982): tabulated center-line profiles, Re = 100 and Re = 1000 | Re=100: L2 **0.5 % / 2.4 %** (u/v); Re=1000: **1.5 % / 2.1 %** |
| 5 | **Cylinder in channel** — Kármán vortex street | inlet/outlet, momentum exchange, unsteady flow | St ≈ 0.164 at Re = 100 (unbounded reference; blockage D/H = 1/8), Cd ≈ 1.3–1.4 | **St = 0.1637**, Cd = 1.33, Cl_amp = 0.28 |
| 6 | **Backward-facing step** — 1:2 sudden expansion, parabolic (developed) inflow | separation and reattachment; the parabolic-inlet machinery | reattachment length, Armaly et al. (1983): x_r/S ≈ 3 (Re=100), ≈ 5 (Re=200), ≈ 8 (Re=400); Re on the inlet hydraulic diameter | **x_r/S = 4.18** at Re = 200 |
| 7 | **Porous medium** — random overlapping grains (fixed seed), fully periodic, force-driven | flow in complex geometry — the signature LBM application | Darcy's law: `⟨u_sup⟩ = K·g/ν`, so the permeability K must be a property of the geometry alone | K ≈ 8 cells² at ε = 0.71; K drifts **6.8 %** between τ = 0.74 and 0.62 — the known BGK+bounce-back artifact (wall position depends on τ; Pan, Luo & Miller 2006), the ready-made gate for a TRT upgrade |

<img width="1919" height="1031" alt="image" src="https://github.com/user-attachments/assets/75c98d54-fe35-414c-89f4-3bda9b5435ed" />


Additional structural gates (all in `--selftest`):

- **f^eq moments** — mass / momentum / Euler stress of the discrete
  equilibrium are exact: error 1.4·10⁻¹⁶;
- **mass conservation** — bounce-back is conservative; drift over 300 cavity
  steps: 4.5·10⁻¹³ (relative);
- **CPU ≡ GPU** — 8.3·10⁻¹⁶ after 200 steps (see above).

Wall-coordinate subtlety used throughout the validation: with halfway
bounce-back the wall planes sit at Y = 0.5 and Y = NY−1.5, so profiles are
compared in the normalized coordinate `s = (j − 0.5)/(N−2)`, and the macro
fields of wall cells are *ghost-filled* (`u_ghost = 2u_wall − u_fluid`) so
that bilinear sampling reads exactly `u_wall` on the wall plane.

---

## 4. Building and running

Requirements: MSVC 2022, Qt's qmake (build system only — no Qt libraries),
freeglut via vcpkg, optionally the CUDA toolkit (auto-detected; without it
the app builds CPU-only).

```bash
./build.bat        # vcvars64 + qmake + nmake (+ copies freeglut.dll)
```

### Command-line gates

```bash
./LBM.exe --selftest        # all structural gates; exit code = #failed
./LBM.exe --gpucheck        # CPU vs GPU, 200 steps, max|df|
./LBM.exe --cavity 100      # cavity to steady state vs Ghia (or 1000)
./LBM.exe --cylinder 100    # Karman street, measures St / Cd / Cl
./LBM.exe --step 200        # backward-facing step, x_r/S vs Armaly
./LBM.exe --porous          # Darcy gate: K at two viscosities must agree
./LBM.exe --bench 2000      # MLUPS, CPU and GPU
./LBM.exe --xtest           # headless replay of the GUI GPU-toggle paths
./LBM.exe --auto            # hands-off GUI soak test (timers press the keys)
```

### GUI controls

| Key | Action |
|-----|--------|
| `Space` / `B` / `N` | run–pause / single step / reset |
| `O` | next scenario |
| `X` | toggle CPU ↔ GPU core (state carried over, works mid-run) |
| `1 2 3` | field: vorticity / \|u\| / ρ |
| `8 9` | grid N ÷2 / ×2 |
| `-` `=` | Re ÷2 / ×2 (τ recomputed on the fly) |
| `S` | steps per frame 1/10/50/200/1000 |
| `V`, `,` `.` | velocity arrows on/off, arrow length |
| `G` | lattice grid lines overlay |
| `P` | passive tracer particles (massless markers advected by the macro velocity — LBM itself has no particles, this is flow visualization) |
| left click | **cell probe**: highlights the cell and draws its 9 populations as arrows of the non-equilibrium part `f_i − f_i^eq` (orange = surplus, blue = deficit; the neq part carries the viscous stress), plus a table of `f_i`, `f_i^eq`, ρ, u. Right click hides it. Works in GPU mode too (the probed state is pulled from the device). |
| `[` `]` | color brightness |
| `K` / `T` | validation table / self-tests (console) |
| `ESC` | quit |

---

## 5. Known pitfalls (hard-won)

- **CUDA context must be created before the OpenGL window.** Creating it
  lazily (first `cudaMalloc` when the user pressed `X`) with a live GL
  context crashed inside `nvcuda64.dll` on an Optimus laptop (RTX 5070,
  driver 580.88) — while the identical headless code path ran flawlessly.
  Diagnosed from the Windows event log (faulting module) and a deterministic
  `--auto` repro; fixed by `gpuWarmup()` (`cudaFree(0)`) in `main()` before
  `glutInit()`. Applies to any CUDA+GL app on such systems.
- **`nmake clean` after editing `lbm.h`** — the incremental build has been
  seen linking stale objects after header changes, producing spurious access
  violations.
- **τ → 0.5 instability**: BGK becomes unstable as `τ` approaches 1/2
  (high Re on a coarse grid). The app warns when τ < 0.51; the cures are a
  finer grid (`9`), lower Re (`-`), or (as an extension) a TRT/MRT collision
  operator.
- **Compressibility**: errors grow as O(Ma²); keep `u_lat ≤ 0.1`. This is
  visible experimentally: raising `u_lat` degrades the measured ν_eff in the
  Taylor–Green gate.

## 6. Pointers for extensions

TRT/MRT collision (stability at high Re), Smagorinsky LES closure, thermal
lattice (second distribution set for temperature → Rayleigh–Bénard, with the
critical Rayleigh number 1708 as the next gate), interpolated bounce-back for
curved walls (removes the staircase cylinder surface), Zou–He velocity/
pressure boundaries, D3Q19 in 3D.

## 7. References — where each piece of the physics comes from

The code is written from scratch for this project; every formula follows the
standard formulations below (conventions match the Krüger et al. textbook).

- **D2Q9 BGK model** (velocity set, weights, `f^eq`, `ν = c_s²(τ−½)`):
  Y. Qian, D. d'Humières, P. Lallemand, *Lattice BGK models for the
  Navier–Stokes equation*, Europhys. Lett. **17** (1992) 479.
- **Theory / Chapman–Enskog / method overview**: T. Krüger, H. Kusumaatmaja,
  A. Kuzmin, O. Shardt, G. Silva, E. Viggen, *The Lattice Boltzmann Method:
  Principles and Practice*, Springer (2017) — the de-facto standard textbook;
  companion code: github.com/lbm-principles-practice. Also the classic
  review: S. Chen, G. Doolen, Annu. Rev. Fluid Mech. **30** (1998) 329.
- **Guo forcing scheme** (§1.3): Z. Guo, C. Zheng, B. Shi, *Discrete lattice
  effects on the forcing term in the lattice Boltzmann method*, Phys. Rev. E
  **65** (2002) 046308.
- **Halfway bounce-back (2nd-order wall), moving-wall momentum term,
  momentum-exchange force** (§1.4–1.5): A. J. C. Ladd, *Numerical
  simulations of particulate suspensions via a discretized Boltzmann
  equation. Part 1*, J. Fluid Mech. **271** (1994) 285; momentum exchange
  refined in R. Mei, D. Yu, W. Shyy, L.-S. Luo, Phys. Rev. E **65** (2002)
  041203.
- **Cavity benchmark tables**: U. Ghia, K. N. Ghia, C. T. Shin,
  *High-Re solutions for incompressible flow using the Navier–Stokes
  equations and a multigrid method*, J. Comput. Phys. **48** (1982) 387.
- **Taylor–Green vortex**: exact 2D Navier–Stokes solution (G. I. Taylor,
  A. E. Green, Proc. R. Soc. A **158** (1937); the 2D decaying form used
  here is the standard LBM accuracy test).
- **Cylinder references**: St(Re=100) ≈ 0.164 from the circular-cylinder
  literature (C. H. K. Williamson, Annu. Rev. Fluid Mech. **28** (1996));
  the confined-channel benchmark variant is M. Schäfer, S. Turek,
  *Benchmark computations of laminar flow around a cylinder* (1996).
- **Backward-facing step references**: B. F. Armaly, F. Durst, J. C. F.
  Pereira, B. Schönung, *Experimental and theoretical investigation of
  backward-facing step flow*, J. Fluid Mech. **127** (1983) 473; the
  sudden-expansion-at-inlet variant follows D. K. Gartling, Int. J. Numer.
  Methods Fluids **11** (1990) 953.
- **Permeability τ-dependence of BGK bounce-back** (the porous-gate
  artifact): C. Pan, L.-S. Luo, C. T. Miller, *An evaluation of lattice
  Boltzmann schemes for porous medium flow simulation*, Comput. Fluids
  **35** (2006) 898.
- **Well-known open-source LBM codes** (not used as a base, useful for
  cross-reading): Palabos, OpenLB, and the minimal MATLAB/Python examples by
  J. Latt (lbmethod.org lineage).
