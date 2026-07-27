# A D2Q9 Lattice Boltzmann Solver with Single-Source CPU/GPU Execution and a Benchmark-Driven Verification Suite

An interactive two-dimensional incompressible-flow solver built on the
lattice Boltzmann method (LBM). The numerical core is written **once** and
compiled unchanged for both the CPU (OpenMP) and the GPU (CUDA), so the two
back-ends are provably identical to floating-point round-off. Seven canonical
flows are provided, each paired with an analytic solution or a published
benchmark, together with a command-line verification suite: unit checks on the
discrete operator, an observed-order-of-accuracy study, established benchmark
comparisons, and a bit-exact CPU/GPU cross-check.

Kármán vortex street:
<img width="1919" height="1026" alt="image" src="https://github.com/user-attachments/assets/438407ae-461f-4057-890b-e052b423c00b" />

Porous medium:
<img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/9615eecd-dd9e-4a14-9979-3564fd6ff024" />

---

## 1. Governing method

### 1.1 The lattice Boltzmann equation

Rather than discretising the Navier–Stokes equations directly, LBM evolves a
set of discrete-velocity distribution functions $f_i(\mathbf{x},t)$ on a
regular lattice, where $f_i$ is the population of particles moving with the
lattice velocity $\mathbf{e}_i$. Each time step is the composition of a local
**collision** and a **streaming** shift. With the single-relaxation-time
Bhatnagar–Gross–Krook (BGK) operator,

```math
f_i^{\star}(\mathbf{x},t) = f_i(\mathbf{x},t) - \frac{1}{\tau}\Big[\,f_i(\mathbf{x},t) - f_i^{\mathrm{eq}}(\rho,\mathbf{u})\,\Big],
```

```math
f_i(\mathbf{x}+\mathbf{e}_i,\,t+1) = f_i^{\star}(\mathbf{x},t),
```

where $\tau$ is the relaxation time and $f_i^{\star}$ the post-collision state.
A Chapman–Enskog expansion shows that, in the low-Mach limit, this system
recovers the weakly compressible Navier–Stokes equations with kinematic
viscosity

```math
\nu = c_s^{2}\Big(\tau-\tfrac{1}{2}\Big),\qquad c_s^{2}=\tfrac{1}{3},\qquad \Longrightarrow\qquad \tau = 3\nu + \tfrac{1}{2} ,
```

$c_s$ being the lattice speed of sound. All quantities are expressed in lattice
units ($\Delta x=\Delta t=1$, reference density $\rho_0=1$). The physical
regime is set solely by the Reynolds number, with the lattice Mach number kept
small to control the $\mathcal{O}(\mathrm{Ma}^2)$ compressibility error:

```math
\mathrm{Re}=\frac{U L}{\nu},\qquad \mathrm{Ma}=\frac{U}{c_s}=\sqrt{3}\,u_{\text{lat}},\qquad u_{\text{lat}}\lesssim 0.1 .
```

Here $U=u_{\text{lat}}$ is the velocity scale and $L$ the characteristic length
of the flow in lattice cells.

### 1.2 The D2Q9 velocity set

The model uses nine velocities — one rest, four axial, four diagonal:

```math
\mathbf{e}_0=(0,0),\quad
\mathbf{e}_{1\text{–}4}=(\pm1,0),(0,\pm1),\quad
\mathbf{e}_{5\text{–}8}=(\pm1,\pm1),
```

with lattice weights

```math
w_0=\tfrac{4}{9},\qquad w_{1\text{–}4}=\tfrac{1}{9},\qquad w_{5\text{–}8}=\tfrac{1}{36}.
```

The second-order (truncated Maxwellian) equilibrium is

```math
f_i^{\mathrm{eq}} = w_i\,\rho\left[\,1 + 3(\mathbf{e}_i\!\cdot\!\mathbf{u}) + \tfrac{9}{2}(\mathbf{e}_i\!\cdot\!\mathbf{u})^2 - \tfrac{3}{2}\,\mathbf{u}^2\,\right],
```

in which the coefficients are $c_s^{-2}=3$, $\tfrac{1}{2} c_s^{-4}=\tfrac{9}{2}$ and
$\tfrac{1}{2} c_s^{-2}=\tfrac{3}{2}$. The hydrodynamic moments are

```math
\rho=\sum_i f_i,\qquad \rho\,\mathbf{u}=\sum_i f_i\,\mathbf{e}_i + \tfrac{1}{2}\mathbf{F},
```

the $\tfrac{1}{2}\mathbf{F}$ term being the forcing correction of §1.3. By
construction the equilibrium reproduces the mass, momentum and Euler stress
moments exactly,

```math
\sum_i f_i^{\mathrm{eq}}=\rho,\qquad
\sum_i f_i^{\mathrm{eq}}\mathbf{e}_i=\rho\mathbf{u},\qquad
\sum_i f_i^{\mathrm{eq}}\mathbf{e}_i\mathbf{e}_i=\rho\big(c_s^{2}\mathbf{I}+\mathbf{u}\mathbf{u}\big),
```

a set of identities verified to machine precision by the first unit test (§3).

### 1.3 Body force (Guo scheme)

A constant body force $\mathbf{F}=(g_x,0)$ drives the Poiseuille channel. The
scheme of Guo, Zheng & Shi (2002) is used, which is second-order accurate and
free of spurious $\tau$-dependent terms. The velocity carries a half-force
shift, and a forcing population is added after collision:

```math
\mathbf{u} = \frac{1}{\rho}\left(\sum_i f_i\,\mathbf{e}_i + \tfrac{1}{2}\mathbf{F}\right),
```

```math
F_i = w_i\left(1-\frac{1}{2\tau}\right)\Big[\,3(\mathbf{e}_i-\mathbf{u}) + 9(\mathbf{e}_i\!\cdot\!\mathbf{u})\,\mathbf{e}_i\,\Big]\cdot\mathbf{F},
```

```math
f_i^{\star} = f_i - \frac{1}{\tau}\big(f_i - f_i^{\mathrm{eq}}\big) + F_i .
```

For a channel of half-plane-to-half-plane height $H$, the force is chosen as
$g_x = 8\nu U/H^2$ so that the analytic centre-line velocity equals exactly
$u_{\text{lat}}$.

### 1.4 Boundary conditions

| Cell type | Rule |
|-----------|------|
| `CT_FLUID`  | full stream + collide |
| `CT_SOLID`  | **halfway bounce-back** — a population directed into a wall cell is reflected within the same step (see below). The no-slip plane lies half a cell beyond the last fluid node, giving second-order wall placement; an $N$-cell box therefore has an effective width of $N-2$ cells between wall planes. |
| `CT_MOVING` | halfway bounce-back with a wall-momentum term (moving lid). |
| `CT_INLET`  | equilibrium inlet, $f_i=f_i^{\mathrm{eq}}(\rho{=}1,\mathbf{u}_{\text{in}})$, imposed every step (uniform, or a prescribed parabolic profile for the step case). |
| `CT_OUTLET` | first-order outflow — the outlet column copies the populations of its upstream neighbour (zero-gradient in $f$). |

Halfway bounce-back reflects the population into the fluid cell along the
opposite direction $\bar\imath$ (with $\mathbf{e}_{\bar\imath}=-\mathbf{e}_i$):

```math
f_i(\mathbf{x}_f,\,t+1) = f_{\bar\imath}^{\star}(\mathbf{x}_f,\,t)\ +\ 2\,w_i\,\rho_0\,\frac{\mathbf{e}_i\!\cdot\!\mathbf{u}_w}{c_s^{2}},
```

where the last term (with $2\rho_0/c_s^{2}=6$ for $\rho_0=1$) vanishes for a
static wall and injects the wall momentum for a moving one. Periodic
directions are handled by index wrapping during streaming; every non-periodic
edge is fenced by a one-cell layer of boundary cells.

### 1.5 Force on immersed bodies (momentum exchange)

The hydrodynamic force on the cylinder is accumulated over every fluid→solid
link by the momentum-exchange method: each reflected population transfers
$2 f_i^{\star}\mathbf{e}_i$ per step. The drag and lift coefficients and the
Strouhal number follow the standard definitions

```math
C_d=\frac{F_x}{\tfrac{1}{2}\rho_0 U^{2} D},\qquad
C_l=\frac{F_y}{\tfrac{1}{2}\rho_0 U^{2} D},\qquad
\mathrm{St}=\frac{D}{U\,T},
```

$D$ being the cylinder diameter and $T$ the shedding period, measured from the
mean interval between upward zero-crossings of $C_l(t)$.

### 1.6 Algorithmic form

The update is cast in the *pull* formulation and fused into a single pass per
cell — one kernel, coalesced reads and writes, no separate streaming sweep:

```
for every cell x:
    gather   f_i  from  x − e_i        (bounce-back resolved during the gather)
    ρ, u     from the gathered set      (+ ½F correction)
    collide  (BGK + Guo forcing term)
    write    the post-collision set to the second buffer
swap buffers   (ping-pong)
```

Storage is structure-of-arrays, $f[i\,N_xN_y + y\,N_x + x]$, two
double-precision buffers of nine fields.

---

## 2. Implementation: single-source CPU/GPU

```
lattice.h        the physics: lbmCellUpdate() / lbmFeq() / lbmMacroAt(),
                 written once and marked __host__ __device__
lbm_core.cpp     CPU core: state, OpenMP stepping, macroscopic fields,
                 diagnostics (mass, energy, ν_eff, C_d/C_l/St history)
scenarios.cpp    applyScenario() derived parameters + buildFlags() cell maps
validation.cpp   analytic references, Ghia tables, order study, benchmark gates
viz.cpp          freeglut/OpenGL rendering, dimensional axes, comparison panels
main_app.cpp     GLUT loop, keyboard, CPU/GPU switching, CLI dispatch
cuda_kernels.*   CUDA core: thin per-cell kernel wrappers around lattice.h
```

The routine `lbmCellUpdate()` in [`lattice.h`](lattice.h) *is* the entire
numerical method — the fused stream-and-collide update of one cell (§1.6). It is
written once and compiled for **two independent back-ends**:

- **CPU back-end (OpenMP).** In [`lbm_core.cpp`](lbm_core.cpp) an OpenMP
  `#pragma omp parallel for` sweeps the lattice and calls `lbmCellUpdate()` once
  per cell, so the work is spread across all CPU cores (multi-threaded, not
  vectorised by hand). This path needs no CUDA toolkit at all — the build
  detects `nvcc` and, when it is absent, links stub GPU functions so the program
  still compiles and runs CPU-only.
- **GPU back-end (CUDA).** In [`cuda_kernels.cu`](cuda_kernels.cu) a CUDA kernel
  launches **one thread per cell**, each thread calling **the same**
  `lbmCellUpdate()`. The two distribution buffers and the flag field are uploaded
  once; `gpuLbmSteps(n)` then runs $n$ fused kernels back-to-back on the device,
  and only the three macroscopic fields are copied back per rendered frame.

The macro `LBM_HD` expands to `__host__ __device__ __forceinline__` under `nvcc`
and to plain `inline` otherwise, which is what lets a single source function
serve both compilers. OpenMP is therefore *not* an alternative to CUDA — it is
simply how the **CPU** version is parallelised across cores; CUDA is how the
**GPU** version is parallelised across threads. Both execute identical
arithmetic.

Two consequences follow. First, there is a single implementation of the physics
to reason about and maintain. Second, the back-ends must agree bitwise: after
200 cavity steps the measured discrepancy is
$\max_i|f_i^{\text{GPU}}-f_i^{\text{CPU}}| = 8.3\times10^{-16}$, i.e. pure
fused-multiply-add round-off (§3.1, `--gpucheck`). The CPU path thus doubles as
the reference implementation against which the GPU is verified. Throughput is
quantified in §3.4.

---

## 3. Verification

Verification follows the standard hierarchy for lattice Boltzmann codes
(Krüger et al. 2017, ch. 4): (i) unit checks on the discrete operator;
(ii) analytic-solution flows with a grid-convergence study establishing the
design order of accuracy; (iii) comparison against established benchmarks;
and (iv) a code-to-code cross-check between the two back-ends. All checks are
reproducible from the command line (§4) and return an exit code equal to the
number of failed gates.

### 3.1 Unit tests on the discrete operator

- **Equilibrium moments** — the mass, momentum and Euler-stress moments of
  $f_i^{\mathrm{eq}}$ (§1.2) are reproduced to
  $1.4\times10^{-16}$.
- **Mass conservation** — halfway bounce-back is exactly conservative; the
  relative drift over 300 cavity steps is $4.5\times10^{-13}$.
- **CPU $\equiv$ GPU** — bit-level agreement,
  $8.3\times10^{-16}$ after 200 steps.

### 3.2 Order of accuracy

The design accuracy is confirmed on the Taylor–Green vortex, which has the
exact solution

```math
\mathbf{u}(\mathbf{x},t) = U\!\begin{pmatrix}-\cos k_x x\,\sin k_y y\\[2pt]\ \ \sin k_x x\,\cos k_y y\end{pmatrix}e^{-\nu(k_x^2+k_y^2)\,t},\qquad k_x=k_y=\frac{2\pi}{N},
```

with kinetic energy decaying as $E(t)=E_0\,e^{-2\nu(k_x^2+k_y^2)t}$. Because the
solution is known in closed form at every instant, the **computed** field
$\mathbf{u}_h$ can be compared directly against the **exact** field
$\mathbf{u}_{\text{exact}}$ evaluated at the same physical time. The study runs
the same flow at increasing resolution $N$ and measures, at each $N$, the
relative $L_2$ difference between the two fields over all $N^2$ lattice nodes:

```math
\varepsilon_2(N)=\frac{\lVert \mathbf{u}_h-\mathbf{u}_{\text{exact}}\rVert_2}{\lVert \mathbf{u}_{\text{exact}}\rVert_2}
=\sqrt{\frac{\sum_{\mathbf{x}}\lVert \mathbf{u}_h(\mathbf{x})-\mathbf{u}_{\text{exact}}(\mathbf{x})\rVert^2}{\sum_{\mathbf{x}}\lVert \mathbf{u}_{\text{exact}}(\mathbf{x})\rVert^2}} .
```

To make the comparison a clean grid-refinement study, **diffusive scaling** is
used: the relaxation time $\tau$ (hence the lattice viscosity) is held fixed,
the Mach number is shrunk as $\mathrm{Ma}\sim 1/N$, and each run is integrated
to the *same* physical time (so the step count grows as $N^2$). Both the
spatial truncation error and the $\mathcal{O}(\mathrm{Ma}^2)$ compressibility
error are then $\mathcal{O}(\Delta x^2)$, and the observed order between two
successive resolutions is

```math
p=\frac{\ln\!\big[\varepsilon_2(N_1)/\varepsilon_2(N_2)\big]}{\ln(N_2/N_1)} .
```

Results (`--order`, $\tau=0.8$; each row is one full simulation compared to the
analytic solution):

| $N$ | $u_{\text{lat}}$ | steps | $\varepsilon_2$ (solver vs exact) | order $p$ |
|----:|------:|------:|:-----------:|:---------:|
| 32  | 0.0400 | 130  | $4.85\times10^{-3}$ | — |
| 64  | 0.0200 | 520  | $1.22\times10^{-3}$ | 1.99 |
| 128 | 0.0100 | 2080 | $3.00\times10^{-4}$ | 2.03 |
| 256 | 0.0050 | 8320 | $7.67\times10^{-5}$ | 1.96 |

Each halving of the grid spacing (doubling of $N$) cuts the error by
$\approx4\times$, i.e. the observed order is $p\approx 2$ — the second-order
convergence expected of the BGK scheme with halfway bounce-back. (The last
column is the order between consecutive rows; the dash marks the coarsest grid,
which has no predecessor to compare against.)

### 3.3 Benchmark flows

Each scenario isolates one ingredient of the method and is compared against an
analytic solution or a tabulated benchmark; the GUI overlays the live solver
profile on the reference with the relative $L_2$ error and a verdict.

| # | Scenario (key `O`) | Ingredient exercised | Reference | Result |
|---|--------------------|----------------------|-----------|--------|
| 1 | **Taylor–Green vortex** — periodic decaying array | bulk collision + streaming, no walls | exact decay; $\nu_{\text{eff}}$ from the energy history | $\nu_{\text{eff}}$ error **0.07 %** |
| 2 | **Poiseuille** — force-driven channel | bounce-back walls + Guo forcing | exact parabola $u=4Us(1-s)$ | $L_2$ **0.10 %** |
| 3 | **Couette** — lid-dragged channel | moving-wall bounce-back | exact linear $u=Us$ | $L_2$ **0.03 %** |
| 4 | **Lid-driven cavity** | full nonlinear steady flow | Ghia, Ghia & Shin (1982), $\mathrm{Re}=100,\,1000$ | Re 100: $L_2$ **0.5 % / 2.4 %**; Re 1000: **1.5 % / 2.1 %** |
| 5 | **Cylinder in channel** — Kármán street | inlet/outlet, momentum exchange, unsteady wake | $\mathrm{St}\approx0.164$ at $\mathrm{Re}=100$ (Williamson 1996) | **St = 0.1637**, $C_d=1.33$ |
| 6 | **Backward-facing step** — 1:2 sudden expansion | separation / reattachment, parabolic inflow | reattachment $x_r/S$, Armaly et al. (1983) | **$x_r/S=4.18$** at $\mathrm{Re}=200$ |
| 7 | **Porous medium** — random grains, periodic, force-driven | flow in complex geometry (signature LBM use) | Darcy's law (below) | $K\approx8$ cells², $\varepsilon=0.71$ |

<img width="1919" height="1031" alt="image" src="https://github.com/user-attachments/assets/75c98d54-fe35-414c-89f4-3bda9b5435ed" />

The problem statement of each case follows. Throughout, $N$ is the base
resolution (key `8`/`9`), $U=u_{\text{lat}}$ the velocity scale, and the
viscosity is set from the target Reynolds number as $\nu=U L_{\text{ch}}/\mathrm{Re}$
with $L_{\text{ch}}$ the characteristic length named below.

#### 1. Taylor–Green vortex

- **Geometry / BC.** Square domain $N\times N$, **fully periodic** on both axes
  — no walls. $L_{\text{ch}}=N$.
- **Setup.** The lattice is initialised with the exact Taylor–Green field
  (§3.2) at $t=0$ and left to decay freely.
- **Computed.** The kinetic energy $E(t)=\tfrac{1}{2}\sum \mathbf{u}^2$ is
  monitored and the effective viscosity is recovered from its decay rate,
  $\nu_{\text{eff}}=-\ln\!\big(E(t)/E_0\big)/[\,2(k_x^2+k_y^2)\,t\,]$.
- **Compared to.** The analytically imposed $\nu$. Because there are no walls,
  this isolates the collision operator alone.
- **Result.** $\nu_{\text{eff}}$ matches $\nu$ to **0.07 %** ($64^2$, 3000
  steps).

#### 2. Poiseuille channel

- **Geometry / BC.** Strip $N\times N$, **periodic in $x$**; the top and bottom
  rows are static no-slip walls (`CT_SOLID`, halfway bounce-back). The channel
  height between wall planes is $H=N-2=L_{\text{ch}}$.
- **Setup.** A uniform body force $g_x=8\nu U/H^2$ (Guo forcing, §1.3) drives
  the flow, which is integrated to steady state.
- **Computed.** The streamwise velocity profile $u(y)$ across the channel at
  mid-length, in the normalised coordinate $s=(y-\tfrac{1}{2})/H\in[0,1]$.
- **Compared to.** The exact parabola $u(s)=4Us(1-s)$; error is the relative
  $L_2$ norm along the profile.
- **Result.** $L_2=$ **0.10 %** ($H=32$).

#### 3. Couette channel

- **Geometry / BC.** Strip $N\times N$, **periodic in $x$**; bottom row a static
  wall, top row a wall **moving** at $U$ (`CT_MOVING`, bounce-back with the
  wall-momentum term of §1.4). $H=N-2=L_{\text{ch}}$.
- **Setup.** The moving lid drags the fluid; integrated to steady state.
- **Computed.** The profile $u(s)$ across the channel.
- **Compared to.** The exact linear profile $u(s)=Us$ (relative $L_2$).
- **Result.** $L_2=$ **0.03 %** ($H=32$) — the cleanest case, since a linear
  profile is reproduced almost exactly.

For the channel and cavity cases the wall planes lie at $y=1/2$ and $y=N-3/2$,
and the macroscopic fields of wall cells are ghost-filled,
$\mathbf{u}_{\text{ghost}}=2\,\mathbf{u}_{\text{wall}}-\mathbf{u}_{\text{fluid}}$,
so that bilinear sampling reads exactly $\mathbf{u}_{\text{wall}}$ on the wall
plane.

#### 4. Lid-driven cavity

- **Geometry / BC.** Square cavity $N\times N$ enclosed on all four sides; the
  left, right and bottom are static walls, the **top lid moves** at $U$. Side
  $L=N-2=L_{\text{ch}}$.
- **Setup.** The classic nonlinear steady recirculation; run to steady state at
  $\mathrm{Re}=100$ and $1000$ ($128^2$).
- **Computed.** The two centre-line profiles: $u(y)$ along the vertical centre
  line $x=L/2$, and $v(x)$ along the horizontal centre line $y=L/2$.
- **Compared to.** The tabulated benchmark of **Ghia, Ghia & Shin (1982)** —
  the de-facto standard for incompressible cavity flow — at the matching
  Reynolds number.
- **Result.** $\mathrm{Re}=100$: $L_2=$ **0.5 %** and **2.4 %** (for $u$ and
  $v$); $\mathrm{Re}=1000$: **1.5 %** and **2.1 %**.

#### 5. Cylinder in a channel (Kármán vortex street)

- **Geometry / BC.** Channel $3N\times N$ (e.g. $384\times128$) with static
  top/bottom walls, a **uniform equilibrium inlet** ($u=U$) on the left, a
  first-order **outflow** on the right, and a circular cylinder of diameter
  $D=N/8=L_{\text{ch}}$ (blockage $D/H\approx1/8$) placed $\sim6.4\,D$
  downstream of the inlet. Its centre is offset laterally by $0.31$ cells to
  break the symmetry and trigger shedding.
- **Setup.** Uniform inflow at $\mathrm{Re}=UD/\nu=100$; the wake develops an
  unsteady periodic vortex street. The force on the cylinder is obtained by
  **momentum exchange** (§1.5) every step, giving the time series $C_d(t)$,
  $C_l(t)$.
- **Computed.** The Strouhal number $\mathrm{St}=D/(UT)$ from the mean period
  $T$ between upward zero-crossings of $C_l(t)$; the mean drag $C_d$ and the
  lift amplitude.
- **Compared to.** The accepted circular-cylinder value
  $\mathrm{St}\approx0.164$–$0.166$ at $\mathrm{Re}=100$ (Williamson 1996); the
  small confinement shifts it slightly.
- **Result.** **$\mathrm{St}=0.1637$**, $C_d=1.33$, $C_l$ amplitude $0.28$.

#### 6. Backward-facing step

- **Geometry / BC.** Channel $4N\times N$ (e.g. $512\times128$) with a **sudden
  1:2 expansion at the inlet plane**: the lower half of the left boundary is a
  solid step face of height $S=(N-2)/2$, while the upper half carries a
  **parabolic (fully developed) inlet** of mean velocity $U$; top and bottom
  are static walls and the right boundary is an outflow. The Reynolds number is
  built on the inlet hydraulic diameter, $L_{\text{ch}}=N-2$.
- **Setup.** $\mathrm{Re}=200$; run to steady state. The flow separates at the
  step and forms a recirculation bubble along the lower wall.
- **Computed.** The reattachment length $x_r$ — the distance from the step to
  the point where the near-floor streamwise velocity changes sign — expressed
  in step heights $x_r/S$.
- **Compared to.** The experimental/computational data of **Armaly et al.
  (1983)** for the 2-D laminar regime.
- **Result.** **$x_r/S=4.18$** at $\mathrm{Re}=200$, within the reported range.

#### 7. Porous medium (Darcy flow)

- **Geometry / BC.** Square domain $N\times N$, **fully periodic**, filled with
  randomly placed overlapping circular grains (diameter $N/8$, fixed random
  seed for reproducibility) until the solid fraction reaches $\approx28\%$;
  the resulting porosity is $\varepsilon\approx0.71$. $L_{\text{ch}}=D$ (grain
  diameter).
- **Setup.** A uniform body force $g_x$ drives creeping flow through the pore
  network to steady state — the signature LBM application, since the geometry
  enters only through the solid/fluid mask.
- **Computed.** Darcy's law relates the superficial (whole-volume-averaged)
  velocity to the force through a geometry-only permeability $K$, namely
  $\langle u\rangle_{\text{sup}}=K\,g_x/\nu$, hence
  $K=\nu\,\langle u\rangle_{\text{sup}}/g_x$, with the superficial velocity
  $\langle u\rangle_{\text{sup}}=\frac{1}{N_xN_y}\sum_{\text{fluid}}u_x$
  averaged over the whole box.
- **Compared to.** $K$ must be a property of the geometry **alone**, so the gate
  measures it at two viscosities and requires agreement. This exposes a *known*
  limitation of plain BGK/bounce-back: the effective wall position depends
  weakly on $\tau$, so $K$ drifts by **6.8 %** between $\tau=0.74$ and $0.62$
  (Pan, Luo & Miller 2006).
- **Result.** $K\approx8$ cells² at $\varepsilon=0.71$; the reported drift
  doubles as a ready-made acceptance target for a two-relaxation-time (TRT)
  upgrade, which pins the wall for all $\tau$ via the magic parameter
  $\Lambda=\tfrac{3}{16}$.

### 3.4 Throughput: CPU vs GPU

Performance is reported for the Taylor–Green vortex (fully periodic, so the
figure is the pure per-cell kernel cost with no wall or boundary handling), in
double precision, for 4000 time steps. The metric is **MLUPS** — millions of
lattice-cell updates per second, $\mathrm{MLUPS}=N_xN_y\cdot\text{steps}/(t\cdot10^6)$.
Hardware: Intel laptop CPU (OpenMP over all cores) and an NVIDIA RTX 5070 Laptop
GPU (one CUDA thread per cell). Run it with `--bench [steps] [N]`.

| Grid | CPU time | CPU MLUPS | GPU time | GPU MLUPS | GPU speed-up |
|-----:|---------:|----------:|---------:|----------:|:------------:|
| $64^2$   | 0.17 s | 98  | 0.06 s | 275 | 2.8× |
| $128^2$  | 0.53 s | 125 | 0.10 s | 642 | 5.1× |
| $256^2$  | 1.94 s | 136 | 0.38 s | 697 | 5.1× |
| $512^2$  | 8.89 s | 118 | 1.40 s | 748 | 6.3× |

The GPU saturates at roughly 700 MLUPS from $128^2$ upward, while the small
$64^2$ grid under-fills the device and is launch-bound; the CPU back-end holds
about 120–135 MLUPS across all sizes. The absolute values are modest because the
scheme runs in double precision — a single-precision or structure-of-arrays
optimisation pass would raise both back-ends substantially — but the CPU/GPU
ratio and the identical results (`--gpucheck`) are the point here.

---

## 4. Building and running

Requirements: MSVC 2022, Qt's `qmake` (build system only — no Qt libraries),
freeglut via vcpkg, and optionally the CUDA toolkit (auto-detected; without it
the application builds CPU-only). Paths are taken from the `QTDIR`,
`VCPKG_ROOT` and `CUDA_PATH` environment variables.

```bash
./build.bat        # vcvars64 + qmake + nmake (+ copies freeglut.dll)
```

### Command-line verification gates

```bash
./LBM.exe --selftest        # unit tests + channel gates; exit code = #failed
./LBM.exe --order           # Taylor–Green grid-convergence (observed order p)
./LBM.exe --gpucheck        # CPU vs GPU, 200 steps, reports max|Δf|
./LBM.exe --cavity 100      # cavity to steady state vs Ghia (or 1000)
./LBM.exe --cylinder 100    # Kármán street; measures St, C_d, C_l
./LBM.exe --step 200        # backward-facing step; reattachment x_r/S
./LBM.exe --porous          # Darcy gate: K at two viscosities
./LBM.exe --bench 4000 256  # throughput (MLUPS) on Taylor-Green, steps and N
./LBM.exe --xtest --auto    # headless / hands-off GUI soak tests
```

### Interactive controls

| Key | Action |
|-----|--------|
| `Space` / `B` / `N` | run–pause / single step / reset |
| `O` | next scenario |
| `X` | toggle CPU ↔ GPU core (state carried over, works mid-run) |
| `1 2 3` | field: vorticity / $\lvert\mathbf{u}\rvert$ / $\rho$ |
| `8 9` | grid $N$ ÷2 / ×2 |
| `Q` `E` | zoom the view in / out (about the domain centre) |
| `-` `=` | Reynolds number ÷2 / ×2 ($\tau$ recomputed on the fly) |
| `S` | steps per frame 1/10/50/200/1000 |
| `V`, `,` `.` | velocity arrows on/off, arrow length |
| `G` | lattice grid lines overlay |
| `U` | axis scale: physical cell size (0.01 … 1 mm/cell) — see note below |
| `P` | passive tracer particles (massless markers advected by $\mathbf{u}$; LBM has no particles — this is flow visualisation) |
| left click | **cell probe**: draws the nine populations as arrows of the non-equilibrium part $f_i-f_i^{\mathrm{eq}}$ (which carries the viscous stress), plus a table of $f_i$, $f_i^{\mathrm{eq}}$, $\rho$, $\mathbf{u}$. Right click hides it. |
| `[` `]` | colour brightness |
| `K` / `T` | validation table / self-tests (console) |
| `ESC` | quit |

**A note on physical units.** LBM is intrinsically dimensionless: $\Delta x$
and $\Delta t$ are both one lattice unit and the physical regime is fixed only
by the Reynolds number. The size of the domain in metres is therefore a mapping
the user chooses by fixing the physical length of one cell, $\delta x$ (metres
per cell); a single number sets the scale for every scenario, the domain being
$N_x\delta x \times N_y\delta x$. The GUI draws dimensional axis rulers
(auto-switching between µm, mm and m) and reports the domain size; `U` cycles
$\delta x$ from 0.01 to 1 mm/cell (default 0.1 mm/cell, so a 128-cell box is
12.8 mm). To match a physical experiment, choose $\delta x$ so that the
characteristic length agrees with the real one — the flow is unchanged, only
the axis labels are.

---

## 5. Known limitations

- **CUDA context before the GL window.** Creating the CUDA context lazily
  (first `cudaMalloc`) while an OpenGL context is already live crashed inside
  `nvcuda64.dll` on an Optimus laptop (RTX 5070, driver 580.88), whereas the
  identical headless path ran cleanly. It is created eagerly in `main()` via
  `gpuWarmup()` (`cudaFree(0)`) before `glutInit()`. This applies to any
  CUDA + OpenGL application on such systems.
- **$\tau\to\tfrac{1}{2}$ instability.** The single-relaxation-time BGK operator
  loses stability as $\tau\to\tfrac{1}{2}$ (high Reynolds number on a coarse
  grid); the application warns when $\tau<0.51$. Remedies are grid refinement,
  a lower Reynolds number, or a TRT/MRT collision operator (§6).
- **Compressibility.** Errors scale as $\mathcal{O}(\mathrm{Ma}^2)$; keeping
  $u_{\text{lat}}\le0.1$ is required, and raising it visibly degrades the
  Taylor–Green $\nu_{\text{eff}}$.
- **Stair-cased curved walls.** The cylinder surface is voxelised; interpolated
  (Bouzidi) bounce-back would recover its smooth outline and accelerate the
  convergence of $C_d$ and $\mathrm{St}$.

## 6. Possible extensions

Two-relaxation-time / multiple-relaxation-time collision (stability at high
Reynolds number; the porous gate is already its acceptance test), a Smagorinsky
large-eddy closure, a second distribution for temperature (Rayleigh–Bénard
convection, with the critical Rayleigh number $\mathrm{Ra}_c=1708$ as the next
gate), interpolated bounce-back for curved walls, Zou–He velocity/pressure
boundaries, and an extension to three dimensions (D3Q19).

## 7. References

The code is written from scratch; the formulation follows the standard
references below, with conventions matching the Krüger et al. textbook.

1. Y. H. Qian, D. d'Humières, P. Lallemand, *Lattice BGK models for the
   Navier–Stokes equation*, **Europhys. Lett. 17** (1992) 479. — D2Q9 BGK
   model, weights, equilibrium, $\nu=c_s^2(\tau-\tfrac{1}{2})$.
2. T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, E. M. Viggen,
   *The Lattice Boltzmann Method: Principles and Practice*, **Springer**
   (2017). — Chapman–Enskog analysis, verification methodology; companion
   code at `github.com/lbm-principles-practice`.
3. S. Chen, G. D. Doolen, *Lattice Boltzmann method for fluid flows*,
   **Annu. Rev. Fluid Mech. 30** (1998) 329. — review.
4. Z. Guo, C. Zheng, B. Shi, *Discrete lattice effects on the forcing term in
   the lattice Boltzmann method*, **Phys. Rev. E 65** (2002) 046308. — forcing
   scheme (§1.3).
5. A. J. C. Ladd, *Numerical simulations of particulate suspensions via a
   discretized Boltzmann equation. Part 1*, **J. Fluid Mech. 271** (1994) 285;
   R. Mei, D. Yu, W. Shyy, L.-S. Luo, **Phys. Rev. E 65** (2002) 041203. —
   moving-wall bounce-back and momentum-exchange force (§1.4–1.5).
6. U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible flow
   using the Navier–Stokes equations and a multigrid method*, **J. Comput.
   Phys. 48** (1982) 387. — cavity benchmark.
7. G. I. Taylor, A. E. Green, *Mechanism of the production of small eddies from
   large ones*, **Proc. R. Soc. A 158** (1937) 499. — Taylor–Green vortex.
8. C. H. K. Williamson, *Vortex dynamics in the cylinder wake*, **Annu. Rev.
   Fluid Mech. 28** (1996) 477. — cylinder Strouhal number. Confined-channel
   benchmark: M. Schäfer, S. Turek, *Benchmark computations of laminar flow
   around a cylinder* (1996).
9. B. F. Armaly, F. Durst, J. C. F. Pereira, B. Schönung, *Experimental and
   theoretical investigation of backward-facing step flow*, **J. Fluid Mech.
   127** (1983) 473; sudden-expansion variant: D. K. Gartling, **Int. J.
   Numer. Methods Fluids 11** (1990) 953.
10. C. Pan, L.-S. Luo, C. T. Miller, *An evaluation of lattice Boltzmann
    schemes for porous medium flow simulation*, **Comput. Fluids 35** (2006)
    898. — $\tau$-dependence of BGK bounce-back permeability.
