#include "lbm.h"
#include "cuda_kernels.h"
#include <cstring>
#include <chrono>

// Quantitative validation against known solutions (the standard LBM ladder):
//   Taylor-Green : exact transient  u = -U sin(ky) exp(-2 nu k^2 t) at x=0
//   Couette      : exact linear profile  u = U s
//   Poiseuille   : exact parabola       u = 4 U s (1-s)
//   Cavity       : Ghia, Ghia & Shin, J.Comput.Phys. 48 (1982), Re=100 & 1000
//   Cylinder     : St/Cd/Cl in the HUD (no profile panel)
// Plus the self-test gates (--selftest) and the CPU-vs-GPU check (--gpucheck).

// --- Ghia et al. (1982): center-line profiles, lid speed U = 1. ------------
static const int GHIA_N = 17;
static const double ghiaY[GHIA_N] = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
    0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000 };
static const double ghiaU100[GHIA_N] = {
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
    0.78871, 0.84123, 1.00000 };
static const double ghiaU1000[GHIA_N] = {
    0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
    -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
    0.57492, 0.65928, 1.00000 };
static const double ghiaX[GHIA_N] = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
    0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000 };
static const double ghiaV100[GHIA_N] = {
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
    0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
    -0.05906, 0.00000 };
static const double ghiaV1000[GHIA_N] = {
    0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235,
    0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
    -0.21388, 0.00000 };

const char* validationVerdict(double l2rel)
{
    if (l2rel < 0.05) return "GOOD";
    if (l2rel < 0.15) return "FAIR";
    return "POOR";
}

static void finishRange(ValProfile& p, double vmn, double vmx)
{
    if (vmn > 0.0) vmn = 0.0;      // always include the zero axis
    if (vmx < 0.0) vmx = 0.0;
    const double pad = 0.08 * (vmx - vmn + 1e-9);
    p.vmin = vmn - pad;
    p.vmax = vmx + pad;
}

// Couette (kind 0) / Poiseuille (kind 1): u(s)/U at mid-channel x, s in [0,1]
// measured wall plane to wall plane (planes at Y=0.5 and Y=NY-1.5).
static void fillChannelProfile(ValProfile& p, int kind)
{
    const double U = P.ulat, H = (double)(P.NY - 2);
    p.n = 81; p.nref = 0; p.hasExactCurve = true;
    double vmn = 1e30, vmx = -1e30, se2 = 0.0, sx2 = 0.0, linf = 0.0;
    for (int i = 0; i < p.n; i++)
    {
        const double s  = (double)i / (p.n - 1);
        const double ex = kind ? 4.0*s*(1.0-s) : s;
        const double sv = sampleVel(0.5*P.NX, 0.5 + s*H).x / U;
        p.t[i] = s; p.exact[i] = ex; p.solver[i] = sv;
        se2 += (sv-ex)*(sv-ex); sx2 += ex*ex;
        linf = fmax(linf, fabs(sv-ex));
        vmn = fmin(vmn, fmin(sv, ex)); vmx = fmax(vmx, fmax(sv, ex));
    }
    p.l2rel = sqrt(se2 / (sx2 + 1e-30));
    p.linf  = linf;
    finishRange(p, vmn, vmx);
    strcpy(p.title, kind ? "Poiseuille  u(s)/U  (exact parabola)"
                         : "Couette  u(s)/U  (exact linear)");
}

// Taylor-Green: u(y)/U along the column x=0 vs the exact decaying sine.
// Checks the profile shape AND the decay rate (via the analytic amplitude).
static void fillTGProfile(ValProfile& p)
{
    const double U  = P.ulat;
    const double kx = 2.0*PI/P.NX, ky = 2.0*PI/P.NY;
    const double A  = exp(-P.nu * (kx*kx + ky*ky) * (double)stepCount);
    p.n = 81; p.nref = 0; p.hasExactCurve = true;
    double vmn = 1e30, vmx = -1e30, se2 = 0.0, sx2 = 0.0, linf = 0.0;
    for (int i = 0; i < p.n; i++)
    {
        const double t  = (double)i / (p.n - 1);
        const double Y  = t * (P.NY - 1);
        const double ex = -sin(ky * Y) * A;
        const double sv = sampleVel(0.0, Y).x / U;
        p.t[i] = t; p.exact[i] = ex; p.solver[i] = sv;
        se2 += (sv-ex)*(sv-ex); sx2 += ex*ex;
        linf = fmax(linf, fabs(sv-ex));
        vmn = fmin(vmn, fmin(sv, ex)); vmx = fmax(vmx, fmax(sv, ex));
    }
    p.l2rel = sqrt(se2 / (sx2 + 1e-30));
    p.linf  = linf;
    finishRange(p, vmn, vmx);
    strcpy(p.title, "Taylor-Green  u(y)/U at x=0  (exact decay)");
}

// Cavity center-line profile vs the Ghia scatter (reIdx 0: Re=100, 1: Re=1000).
// comp 0: u(y)/U at x*=0.5;  comp 1: v(x)/U at y*=0.5.
// Cavity coords: x* = (X-0.5)/L, y* = (Y-0.5)/L with L = NY-2.
static void fillGhiaProfile(ValProfile& p, int comp, int reIdx)
{
    const double U = P.ulat, L = (double)(P.NY - 2);
    const double* tref = comp ? ghiaX : ghiaY;
    const double* vref = comp ? (reIdx ? ghiaV1000 : ghiaV100)
                              : (reIdx ? ghiaU1000 : ghiaU100);
    p.nref = GHIA_N; p.hasExactCurve = false;
    double vmn = 1e30, vmx = -1e30;
    p.n = 81;
    for (int i = 0; i < p.n; i++)
    {
        const double s = (double)i / (p.n - 1);          // normalized coord
        const double C = 0.5 + s * L;                    // cell coord along line
        const double M = 0.5 + 0.5 * L;                  // mid line
        const double sv = (comp ? sampleVel(C, M).y : sampleVel(M, C).x) / U;
        p.t[i] = s; p.solver[i] = sv; p.exact[i] = 0.0;
        vmn = fmin(vmn, sv); vmx = fmax(vmx, sv);
    }
    double se2 = 0.0, sx2 = 0.0, linf = 0.0;
    for (int i = 0; i < GHIA_N; i++)
    {
        const double C = 0.5 + tref[i] * L;
        const double M = 0.5 + 0.5 * L;
        const double sv = (comp ? sampleVel(C, M).y : sampleVel(M, C).x) / U;
        p.tref[i] = tref[i]; p.vref[i] = vref[i];
        se2 += (sv - vref[i]) * (sv - vref[i]); sx2 += vref[i]*vref[i];
        linf = fmax(linf, fabs(sv - vref[i]));
        vmn = fmin(vmn, vref[i]); vmx = fmax(vmx, vref[i]);
    }
    p.l2rel = sqrt(se2 / (sx2 + 1e-30));
    p.linf  = linf;
    finishRange(p, vmn, vmx);
    sprintf(p.title, "Cavity  %s  vs Ghia Re=%d",
            comp ? "v(x)/U at y*=0.5" : "u(y)/U at x*=0.5", reIdx ? 1000 : 100);
}

int computeValidation(Validation& v)
{
    v.nPanels = 0; v.reMismatch = false; v.refRe = 0.0;
    switch (P.scenario)
    {
    case SCN_TAYLOR_GREEN: fillTGProfile(v.panel[0]);         v.nPanels = 1; break;
    case SCN_COUETTE:      fillChannelProfile(v.panel[0], 0); v.nPanels = 1; break;
    case SCN_POISEUILLE:   fillChannelProfile(v.panel[0], 1); v.nPanels = 1; break;
    case SCN_CAVITY:
    {
        const int reIdx = (fabs(P.Re - 1000.0) < fabs(P.Re - 100.0)) ? 1 : 0;
        v.refRe = reIdx ? 1000.0 : 100.0;
        v.reMismatch = fabs(P.Re - v.refRe) > 1.0;
        fillGhiaProfile(v.panel[0], 0, reIdx);
        fillGhiaProfile(v.panel[1], 1, reIdx);
        v.nPanels = 2;
        break;
    }
    default: break;        // cylinder: HUD metrics + Cl(t) plot instead
    }
    return v.nPanels;
}

void printValidation()
{
    printf("\n=== validation: %s   step %lld   tau=%.4f nu=%.4g Re=%.0f ===\n",
           scenarioName(), stepCount, P.tau, P.nu, P.Re);
    printf("  mass drift (M-M0)/M0 = %.3e\n", massDrift());
    if (P.scenario == SCN_TAYLOR_GREEN)
    {
        const double ne = nuEff();
        if (ne > 0.0)
            printf("  nu_eff = %.6f  vs  nu = %.6f   (err %.2f%%)\n",
                   ne, P.nu, 100.0*fabs(ne - P.nu)/P.nu);
    }
    if (P.scenario == SCN_CYLINDER)
    {
        double St, ClA, Cd;
        if (strouhal(St, ClA, Cd))
            printf("  Cd = %.3f   Cl_amp = %.3f   St = %.4f   (ref St ~ 0.16-0.17 at Re=100)\n",
                   Cd, ClA, St);
        else
            printf("  vortex shedding not developed yet (need a few thousand more steps)\n");
    }
    if (P.scenario == SCN_STEP)
    {
        const double xr = reattachX();
        if (xr > 0.0)
            printf("  reattachment x_r/S = %.2f   (Armaly'83: ~3 @Re=100, ~5 @Re=200, ~8 @Re=400)\n", xr);
        else
            printf("  no recirculation bubble detected yet\n");
    }
    if (P.scenario == SCN_POROUS)
    {
        const double K = porousK();
        printf("  porosity = %.3f   K = %.3f cells^2   (Darcy gate: K invariant under Re/g changes)\n",
               P.poroEps, K);
    }
    Validation v;
    if (!computeValidation(v)) return;
    if (v.reMismatch)
        printf("  WARNING: Re=%.0f matches no Ghia table - set Re=100 or 1000 ('-','=')\n", P.Re);
    for (int k = 0; k < v.nPanels; k++)
    {
        const ValProfile& p = v.panel[k];
        printf("  %-44s  L2rel=%.4f  Linf=%.4f  -> %s\n",
               p.title, p.l2rel, p.linf, validationVerdict(p.l2rel));
    }
}

// ============================ SELF-TEST GATES ==============================

static void setupCase(int scn, int N, double Re)
{
    P.scenario = scn; P.N = N; P.Re = Re;
    applyScenario();
    lbmAllocate();
    buildFlags();
    lbmInitFields();
}

static int gate(const char* name, double val, double tol)
{
    const bool ok = val < tol;
    printf("  [%s] %-46s %.3e  (tol %.1e)\n", ok ? "PASS" : "FAIL", name, val, tol);
    return ok ? 0 : 1;
}

int runSelfTests()
{
    printf("\n===== LBM SELF-TESTS =====\n");
    const Params saveP = P;
    int failed = 0;

    // 1. feq moments: sum feq = rho, sum feq e = rho u, sum feq e e = rho(cs^2 I + u u).
    {
        const int    ex[9] = { 0,1,0,-1,0,1,-1,-1,1 }, ey[9] = { 0,0,1,0,-1,1,1,-1,-1 };
        const double r = 1.13, u = 0.07, v = -0.03;
        double s0 = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (int i = 0; i < 9; i++)
        {
            const double fe = lbmFeq(i, r, u, v);
            s0 += fe; sx += fe*ex[i]; sy += fe*ey[i];
            sxx += fe*ex[i]*ex[i]; syy += fe*ey[i]*ey[i]; sxy += fe*ex[i]*ey[i];
        }
        double err = fabs(s0 - r) + fabs(sx - r*u) + fabs(sy - r*v)
                   + fabs(sxx - r*(1.0/3 + u*u)) + fabs(syy - r*(1.0/3 + v*v))
                   + fabs(sxy - r*u*v);
        failed += gate("feq moments (mass/momentum/stress)", err, 1e-12);
    }

    // 2. Mass conservation: cavity (bounce-back + moving lid), 300 steps.
    {
        setupCase(SCN_CAVITY, 64, 100.0);
        lbmStepsCPU(300);
        failed += gate("mass conservation, cavity 64^2 x300", fabs(massDrift()), 1e-10);
    }

    // 3. Taylor-Green: measured viscosity vs target (collision operator gate).
    {
        setupCase(SCN_TAYLOR_GREEN, 64, 160.0);       // nu = 0.05*64/160 = 0.02
        lbmStepsCPU(3000);
        updateMacro();
        const double ne = nuEff();
        failed += gate("Taylor-Green nu_eff rel err, 64^2 x3000",
                       fabs(ne - P.nu) / P.nu, 0.02);
    }

    // 4. Poiseuille: steady profile vs exact parabola (walls + Guo forcing gate).
    {
        setupCase(SCN_POISEUILLE, 34, 16.0);          // H=32, nu=0.1, tau=0.8
        lbmStepsCPU(8000);
        updateMacro();
        Validation v; computeValidation(v);
        failed += gate("Poiseuille L2 vs exact, H=32", v.panel[0].l2rel, 0.01);
    }

    // 5. Couette: steady linear profile (moving-wall bounce-back gate).
    {
        setupCase(SCN_COUETTE, 34, 16.0);
        lbmStepsCPU(8000);
        updateMacro();
        Validation v; computeValidation(v);
        failed += gate("Couette L2 vs exact, H=32", v.panel[0].l2rel, 0.01);
    }

    // 6. GPU == CPU (only if a CUDA device is present).
    if (gpuAvailable()) failed += runGpuCheck();
    else                printf("  [skip] GPU check: no CUDA device / built without CUDA\n");

    P = saveP;                 // caller re-runs rebuildAll() to restore arrays
    printf("===== self-tests: %s (%d failed) =====\n\n",
           failed ? "FAILED" : "ALL PASS", failed);
    return failed;
}

// CPU vs GPU: identical scenario, identical step count; the update rule is the
// same lattice.h code, so the fields must agree to accumulated roundoff.
int runGpuCheck()
{
    if (!gpuAvailable())
    {
        printf("  gpucheck: no CUDA device available.\n");
        return 0;
    }
    setupCase(SCN_CAVITY, 64, 100.0);
    const int NN = P.NX * P.NY, nsteps = 200;
    std::vector<double> f0 = f;                       // shared IC
    lbmStepsCPU(nsteps);
    std::vector<double> fC = f;                       // CPU result
    f = f0;
    if (!gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
    {
        printf("  gpucheck: gpuLbmInit failed.\n");
        return 1;
    }
    gpuLbmSteps(nsteps, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
    gpuLbmDownloadF(f.data());
    gpuLbmFree();
    double dmax = 0.0, ref = 0.0;
    for (size_t k = 0; k < (size_t)9*NN; k++)
    {
        dmax = fmax(dmax, fabs(f[k] - fC[k]));
        ref  = fmax(ref, fabs(fC[k]));
    }
    printf("  gpucheck [%s]: cavity 64^2 x%d  max|f_gpu-f_cpu| = %.3e  (max|f| = %.3f)\n",
           dmax / ref < 1e-9 ? "PASS" : "FAIL", nsteps, dmax, ref);
    return dmax / ref < 1e-9 ? 0 : 1;
}

// Cavity benchmark gate (--cavity [Re]): run to steady state (on the GPU when
// available), compare the center-line profiles with the Ghia tables.
int runCavityGate(double Re)
{
    setupCase(SCN_CAVITY, 128, Re);
    const int nsteps = (int)((Re >= 500.0 ? 150.0 : 60.0) * P.Lchar / P.ulat);
    printf("cavity gate: Re=%.0f  %dx%d  tau=%.4f  %d steps (%s)...\n",
           Re, P.NX, P.NY, P.tau, nsteps,
           gpuAvailable() ? gpuDeviceName() : "CPU");
    if (gpuAvailable() && gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
    {
        gpuLbmSteps(nsteps, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
        gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
        ghostFillMacro();
        gpuLbmDownloadF(f.data());     // so massDrift() sees the final state
        gpuLbmFree();
        stepCount = nsteps;
    }
    else
    {
        lbmStepsCPU(nsteps);
        updateMacro();
    }
    printValidation();
    Validation v; computeValidation(v);
    int failed = 0;
    for (int k = 0; k < v.nPanels; k++)
        if (v.panel[k].l2rel > 0.10) failed++;    // FAIR-or-better gate
    printf("cavity gate: %s\n", failed ? "FAILED" : "PASS");
    return failed;
}

// Cylinder gate (--cylinder [Re]): develop the Karman street, measure St from
// the Cl(t) zero crossings, compare with the Re=100 reference band.
int runCylinderGate(double Re)
{
    setupCase(SCN_CYLINDER, 128, Re);
    int x0 = (int)(P.cxc - P.D) - 2, x1 = (int)(P.cxc + P.D) + 2;
    int y0 = (int)(P.cyc - P.D) - 2, y1 = (int)(P.cyc + P.D) + 2;
    const int develop = (int)(200.0 * P.D / P.ulat);    // ~200 D/U to lock in
    const int measure = 20000;
    printf("cylinder gate: Re=%.0f  %dx%d  D=%.0f  tau=%.4f  %d+%d steps (%s)...\n",
           Re, P.NX, P.NY, P.D, P.tau, develop, measure,
           gpuAvailable() ? gpuDeviceName() : "CPU");
    if (gpuAvailable() && gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
    {
        std::vector<double> fx(measure), fy(measure);
        gpuLbmSteps(develop, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
        gpuLbmSteps(measure, P.kp(), x0, x1, y0, y1, fx.data(), fy.data());
        for (int s = 0; s < measure; s++) pushForceSample(fx[s], fy[s]);
        gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
        ghostFillMacro();
        gpuLbmFree();
        stepCount = develop + measure;
    }
    else
    {
        lbmStepsCPU(develop + measure);
        updateMacro();
    }
    double St, ClA, Cd;
    if (!strouhal(St, ClA, Cd))
    {
        printf("cylinder gate: FAILED (no shedding detected)\n");
        return 1;
    }
    printf("  Cd = %.3f   Cl_amp = %.3f   St = %.4f\n", Cd, ClA, St);
    const bool ok = (fabs(Re - 100.0) > 1.0) || (St > 0.14 && St < 0.20);
    printf("cylinder gate: %s%s\n", ok ? "PASS" : "FAILED",
           fabs(Re - 100.0) <= 1.0 ? "  (ref band 0.14 < St < 0.20 at Re=100)" : "");
    return ok ? 0 : 1;
}

// --xtest: replicate the GUI 'X' toggle sequence headlessly for every
// scenario: CPU steps -> upload -> GPU steps (+macro download each "frame")
// -> download back -> CPU steps. Localizes the crash path without GLUT.
int runXTest()
{
    if (!gpuAvailable()) { printf("xtest: no CUDA device.\n"); return 0; }
    for (int scn = 0; scn < SCN_COUNT; scn++)
    {
        P.scenario = scn; P.N = 128; P.Re = scenarioDefaultRe(scn);
        printf("xtest %-20s ", scenarioName());
        setupCase(scn, 128, P.Re);
        printf("[cpu50 ");   lbmStepsCPU(50);
        printf("ok] [init ");
        if (!gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
        { printf("FAILED]\n"); return 1; }
        printf("ok] [gpu50x3+macro ");
        for (int frame = 0; frame < 3; frame++)
        {
            if (scn == SCN_CYLINDER)
            {
                int x0 = (int)(P.cxc - P.D) - 2, x1 = (int)(P.cxc + P.D) + 2;
                int y0 = (int)(P.cyc - P.D) - 2, y1 = (int)(P.cyc + P.D) + 2;
                std::vector<double> fx(50), fy(50);
                gpuLbmSteps(50, P.kp(), x0, x1, y0, y1, fx.data(), fy.data());
                for (int s = 0; s < 50; s++) pushForceSample(fx[s], fy[s]);
            }
            else gpuLbmSteps(50, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
            gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
            ghostFillMacro();
        }
        printf("ok] [download ");
        gpuLbmDownloadF(f.data());
        gpuLbmFree();
        printf("ok] [cpu50 ");
        lbmStepsCPU(50);
        updateMacro();
        printf("ok]\n");
    }
    printf("xtest: ALL OK\n");
    return 0;
}

// Step gate (--step [Re]): run to steady state, compare the reattachment
// length with the Armaly et al. (1983) band.
int runStepGate(double Re)
{
    setupCase(SCN_STEP, 128, Re);
    const int nsteps = (int)(15.0 * P.NX / P.ulat);
    printf("step gate: Re=%.0f  %dx%d  S=%.0f  tau=%.4f  %d steps (%s)...\n",
           Re, P.NX, P.NY, P.stepS, P.tau, nsteps,
           gpuAvailable() ? gpuDeviceName() : "CPU");
    if (gpuAvailable() && gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
    {
        gpuLbmSteps(nsteps, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
        gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
        ghostFillMacro();
        gpuLbmFree();
        stepCount = nsteps;
    }
    else { lbmStepsCPU(nsteps); updateMacro(); }
    const double xr = reattachX();
    printf("  x_r/S = %.2f\n", xr);
    // Armaly'83 (laminar 2D range): ~3 at Re=100, ~5 at Re=200. Loose band.
    bool ok = xr > 0.0;
    if (fabs(Re - 100.0) < 1.0) ok = xr > 2.2 && xr < 4.0;
    if (fabs(Re - 200.0) < 1.0) ok = xr > 3.8 && xr < 6.5;
    printf("step gate: %s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}

// Porous gate (--porous): Darcy's law says the permeability is a property of
// the GEOMETRY alone - measure K at two different viscosities (same grains,
// fixed seed) and require they agree. NOTE the known BGK artifact: with plain
// bounce-back the effective wall sits at a tau-DEPENDENT position, so K
// drifts a few percent with viscosity (Pan, Luo & Miller, Comput. Fluids 35
// (2006) - the standard motivation for TRT with the magic Lambda = 3/16,
// which pins the wall for all tau). Measured here: 6.8% between tau = 0.74
// and 0.62, fully steady (identical at 30k and 120k steps). The gate bounds
// the artifact at 10%; switching the collision to TRT should shrink it to
// well under 1% - a ready-made gate for that upgrade.
int runPorousGate()
{
    double K[2] = { 0, 0 };
    const double res[2] = { 10.0, 20.0 };
    for (int r = 0; r < 2; r++)
    {
        setupCase(SCN_POROUS, 128, res[r]);
        const int nsteps = 30000;      // steady: K identical at 30k and 120k
        if (gpuAvailable() && gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
        {
            gpuLbmSteps(nsteps, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
            gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
            ghostFillMacro();
            gpuLbmFree();
        }
        else { lbmStepsCPU(nsteps); updateMacro(); }
        K[r] = porousK();
        printf("porous gate: Re=%.0f  nu=%.3f  porosity=%.3f  ->  K = %.4f cells^2\n",
               res[r], P.nu, P.poroEps, K[r]);
    }
    const double dev = fabs(K[0] - K[1]) / (0.5 * (K[0] + K[1]) + 1e-30);
    const bool ok = K[0] > 0.0 && K[1] > 0.0 && dev < 0.10;
    printf("porous gate [%s]: K deviation between viscosities = %.2f%%  (tol 10%%)\n",
           ok ? "PASS" : "FAILED", 100.0 * dev);
    printf("  (the few-%% drift is the known BGK+bounce-back artifact - wall position\n"
           "   depends on tau; Pan/Luo/Miller 2006. TRT collision would remove it.)\n");
    return ok ? 0 : 1;
}

void runBench(int nsteps)
{
    if (nsteps < 1) nsteps = 1000;
    const double cells = (double)P.NX * P.NY;
    printf("bench: %s %dx%d, %d steps\n", scenarioName(), P.NX, P.NY, nsteps);
    using clk = std::chrono::steady_clock;

    lbmInitFields();
    auto t0 = clk::now();
    lbmStepsCPU(nsteps);
    double dt = std::chrono::duration<double>(clk::now() - t0).count();
    printf("  CPU: %.2f s   %.1f MLUPS\n", dt, cells*nsteps/dt/1e6);

    if (gpuAvailable())
    {
        lbmInitFields();
        gpuLbmInit(f.data(), flag.data(), P.NX, P.NY);
        gpuLbmSteps(50, P.kp(), 0, -1, 0, -1, nullptr, nullptr);   // warm-up
        t0 = clk::now();
        gpuLbmSteps(nsteps, P.kp(), 0, -1, 0, -1, nullptr, nullptr);
        dt = std::chrono::duration<double>(clk::now() - t0).count();
        printf("  GPU (%s): %.2f s   %.1f MLUPS\n", gpuDeviceName(), dt,
               cells*nsteps/dt/1e6);
        gpuLbmFree();
    }
    else printf("  GPU: not available\n");
}
