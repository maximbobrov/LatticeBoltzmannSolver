#include "lbm.h"
#include <algorithm>
#include <cstring>

// CPU core: state arrays, stepping loop, macro fields, integral diagnostics.
// The per-cell math itself lives in lattice.h (shared with the CUDA kernels).

Params P;

std::vector<double>        f, fnew, rho, ux, uy;
std::vector<unsigned char> flag;
long long stepCount = 0;
double    tgE0      = 1.0;
static double massRef = 1.0;

// Ring buffer for the per-step cylinder force history (Cd/Cl time series; the
// Strouhal number is measured from the Cl zero crossings).
static const int FH_MAX = 20000;
static std::vector<double> fhCl(FH_MAX, 0.0), fhCd(FH_MAX, 0.0);
static int fhHead = 0, fhCount = 0;

void   forceHistReset()  { fhHead = 0; fhCount = 0; }
int    forceHistCount()  { return fhCount; }
double forceHistCl(int i){ return fhCl[(fhHead - fhCount + i + 2*FH_MAX) % FH_MAX]; }
double forceHistCd(int i){ return fhCd[(fhHead - fhCount + i + 2*FH_MAX) % FH_MAX]; }

void pushForceSample(double Fx, double Fy)
{
    const double q = 0.5 * P.ulat * P.ulat * P.D;   // dynamic pressure * D (rho0=1)
    fhCd[fhHead] = Fx / q;
    fhCl[fhHead] = Fy / q;
    fhHead = (fhHead + 1) % FH_MAX;
    if (fhCount < FH_MAX) fhCount++;
}

void lbmAllocate()
{
    const int NN = P.NX * P.NY;
    f.assign((size_t)9 * NN, 0.0);
    fnew.assign((size_t)9 * NN, 0.0);
    rho.assign(NN, 1.0);
    ux.assign(NN, 0.0);
    uy.assign(NN, 0.0);
    flag.assign(NN, (unsigned char)CT_FLUID);
}

// Initial condition: f = feq(rho0, u0). Taylor-Green additionally seeds the
// consistent pressure field rho = 1 + 3p (reduces the initial transient); the
// cylinder starts from a uniform stream (the lateral center offset in
// buildFlags() breaks the symmetry and triggers shedding).
void lbmInitFields()
{
    const int    NX = P.NX, NY = P.NY, NN = NX * NY;
    const double U = P.ulat;
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
        {
            const int idx = y*NX + x;
            double r0 = 1.0, u0 = 0.0, v0 = 0.0;
            const unsigned char ct = flag[idx];
            if (ct == CT_FLUID || ct == CT_INLET || ct == CT_OUTLET)
            {
                if (P.scenario == SCN_TAYLOR_GREEN)
                {
                    const double kx = 2.0*PI/NX, ky = 2.0*PI/NY;
                    const double xx = kx*x, yy = ky*y;
                    u0 = -U * cos(xx) * sin(yy);
                    v0 =  U * sin(xx) * cos(yy);
                    r0 = 1.0 - 0.75*U*U*(cos(2.0*xx) + cos(2.0*yy));  // 3p/cs^2 seed
                }
                else if (P.scenario == SCN_CYLINDER) u0 = P.uin;
            }
            for (int i = 0; i < 9; i++) f[(size_t)i*NN + idx] = lbmFeq(i, r0, u0, v0);
        }
    stepCount = 0;
    forceHistReset();
    updateMacro();
    tgE0    = kineticEnergy();
    massRef = totalMass();
}

void lbmStepsCPU(int n)
{
    const LbmParams p = P.kp();
    for (int s = 0; s < n; s++)
    {
        #pragma omp parallel for
        for (int y = 0; y < P.NY; y++)
            for (int x = 0; x < P.NX; x++)
                lbmCellUpdate(f.data(), fnew.data(), flag.data(), p, x, y);
        f.swap(fnew);
        stepCount++;
        if (P.scenario == SCN_CYLINDER)
        {
            double Fx, Fy;
            cylForceCPU(Fx, Fy);
            pushForceSample(Fx, Fy);
        }
    }
}

void updateMacro()
{
    const LbmParams p = P.kp();
    const int NN = P.NX * P.NY;
    #pragma omp parallel for
    for (int idx = 0; idx < NN; idx++)
    {
        double r, u, v;
        lbmMacroAt(f.data(), flag.data(), p, idx, r, u, v);
        rho[idx] = r; ux[idx] = u; uy[idx] = v;
    }
    ghostFillMacro();
}

// Wall-cell macro values = linear mirror of the adjacent fluid about the wall
// plane (u_ghost = 2 u_wall - u_fluid), so that bilinear sampling and finite
// differences see exactly u_wall at the halfway-bounce-back wall plane. Pure
// post-processing for viz/validation; the solver itself never reads these.
void ghostFillMacro()
{
    const int NX = P.NX, NY = P.NY;
    const int dx4[4] = { 1, -1, 0, 0 }, dy4[4] = { 0, 0, 1, -1 };
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
        {
            const int idx = y*NX + x;
            const unsigned char ct = flag[idx];
            if (ct != CT_SOLID && ct != CT_MOVING) continue;
            double su = 0.0, sv = 0.0, sr = 0.0; int c = 0;
            for (int k = 0; k < 4; k++)
            {
                int xn = x + dx4[k], yn = y + dy4[k];
                if (P.perX) xn = lbmWrap(xn, NX); else if (xn < 0 || xn >= NX) continue;
                if (P.perY) yn = lbmWrap(yn, NY); else if (yn < 0 || yn >= NY) continue;
                const int jn = yn*NX + xn;
                const unsigned char cn = flag[jn];
                if (cn == CT_FLUID || cn == CT_INLET || cn == CT_OUTLET)
                { su += ux[jn]; sv += uy[jn]; sr += rho[jn]; c++; }
            }
            const double uwx = (ct == CT_MOVING) ? P.uLid : 0.0;
            if (c) { ux[idx] = 2.0*uwx - su/c; uy[idx] = -sv/c; rho[idx] = sr/c; }
            else   { ux[idx] = uwx;            uy[idx] = 0.0;   rho[idx] = 1.0;  }
        }
}

double totalMass()
{
    const int NN = P.NX * P.NY;
    double s = 0.0;
    for (int idx = 0; idx < NN; idx++)
    {
        const unsigned char ct = flag[idx];
        if (ct == CT_SOLID || ct == CT_MOVING) continue;
        for (int i = 0; i < 9; i++) s += f[(size_t)i*NN + idx];
    }
    return s;
}

double massDrift() { return (totalMass() - massRef) / massRef; }

double kineticEnergy()
{
    const int NN = P.NX * P.NY;
    double s = 0.0;
    for (int idx = 0; idx < NN; idx++)
        if (flag[idx] == CT_FLUID)
            s += 0.5*(ux[idx]*ux[idx] + uy[idx]*uy[idx]);
    return s;
}

// Taylor-Green: E(t) = E0 exp(-2 nu (kx^2+ky^2) t)  =>  measured viscosity.
// The single most direct gate on the collision operator: no walls involved.
double nuEff()
{
    if (P.scenario != SCN_TAYLOR_GREEN || stepCount <= 0) return -1.0;
    const double E = kineticEnergy();
    if (E <= 0.0 || tgE0 <= 0.0) return -1.0;
    const double kx = 2.0*PI/P.NX, ky = 2.0*PI/P.NY;
    return -log(E / tgE0) / (2.0*(kx*kx + ky*ky) * (double)stepCount);
}

Vec2 sampleVel(double X, double Y)
{
    const int NX = P.NX, NY = P.NY;
    if (X < 0.0) X = 0.0; if (X > NX - 1.0) X = NX - 1.0;
    if (Y < 0.0) Y = 0.0; if (Y > NY - 1.0) Y = NY - 1.0;
    int i0 = (int)X, j0 = (int)Y;
    if (i0 > NX - 2) i0 = NX - 2;
    if (j0 > NY - 2) j0 = NY - 2;
    const double a = X - i0, b = Y - j0;
    const int q00 = j0*NX + i0, q10 = q00 + 1, q01 = q00 + NX, q11 = q01 + 1;
    Vec2 u;
    u.x = (1-a)*(1-b)*ux[q00] + a*(1-b)*ux[q10] + (1-a)*b*ux[q01] + a*b*ux[q11];
    u.y = (1-a)*(1-b)*uy[q00] + a*(1-b)*uy[q10] + (1-a)*b*uy[q01] + a*b*uy[q11];
    return u;
}

// Momentum-exchange force on the cylinder: every fluid->solid link transfers
// 2 f*_i e_i per step (static wall). Restricted to a box around the cylinder
// so the channel walls do not contribute.
void cylForceCPU(double& Fx, double& Fy)
{
    const int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
    const int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
    const int NX = P.NX, NY = P.NY, NN = NX * NY;
    int x0 = (int)(P.cxc - P.D) - 2, x1 = (int)(P.cxc + P.D) + 2;
    int y0 = (int)(P.cyc - P.D) - 2, y1 = (int)(P.cyc + P.D) + 2;
    if (x0 < 1) x0 = 1; if (x1 > NX - 2) x1 = NX - 2;
    if (y0 < 1) y0 = 1; if (y1 > NY - 2) y1 = NY - 2;
    Fx = Fy = 0.0;
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++)
        {
            const int idx = y*NX + x;
            if (flag[idx] != CT_FLUID) continue;
            for (int i = 1; i < 9; i++)
                if (flag[(y + ey[i])*NX + (x + ex[i])] == CT_SOLID)
                {
                    Fx += 2.0 * f[(size_t)i*NN + idx] * ex[i];
                    Fy += 2.0 * f[(size_t)i*NN + idx] * ey[i];
                }
        }
}

// Backward-facing step: reattachment length. The recirculation bubble ends
// where the near-floor u_x turns from negative back to positive; we take the
// LAST negative cell along the first fluid row and interpolate the crossing.
// Returned in step heights S, measured from the inlet plane (x = 0.5).
double reattachX()
{
    if (P.scenario != SCN_STEP || P.stepS <= 0.0) return -1.0;
    const int NX = P.NX, j = 1;
    int last = -1;
    for (int x = 2; x < NX - 2; x++)
        if (ux[j*NX + x] < 0.0) last = x;
    if (last < 2 || last >= NX - 3) return -1.0;   // none, or runs off the grid
    const double u0 = ux[j*NX + last], u1 = ux[j*NX + last + 1];
    const double xc = last + (u1 > u0 ? -u0 / (u1 - u0) : 0.5);
    return (xc - 0.5) / P.stepS;
}

// Porous medium: Darcy permeability from the superficial velocity
// <u_sup> = (1/V_total) sum_fluid u_x  and  <u_sup> = K g / nu.
double porousK()
{
    if (P.scenario != SCN_POROUS || P.gx <= 0.0) return -1.0;
    const int NN = P.NX * P.NY;
    double s = 0.0;
    for (int idx = 0; idx < NN; idx++)
        if (flag[idx] == CT_FLUID) s += ux[idx];
    return (s / NN) * P.nu / P.gx;
}

// Strouhal number from the Cl(t) history: mean period between upward zero
// crossings of Cl - <Cl> (linear interpolation), St = D / (U T).
bool strouhal(double& St, double& ClAmp, double& CdMean)
{
    const int n = forceHistCount();
    if (n < 500) return false;
    double mean = 0.0, cdm = 0.0;
    for (int i = 0; i < n; i++) { mean += forceHistCl(i); cdm += forceHistCd(i); }
    mean /= n; cdm /= n;
    double first = -1.0, last = -1.0, amp = 0.0;
    int ncr = 0;
    double prev = forceHistCl(0) - mean;
    for (int i = 1; i < n; i++)
    {
        const double cur = forceHistCl(i) - mean;
        amp = fmax(amp, fabs(cur));
        if (prev < 0.0 && cur >= 0.0)
        {
            const double t = (i - 1) + (-prev) / (cur - prev);
            if (first < 0.0) first = t;
            last = t; ncr++;
        }
        prev = cur;
    }
    if (ncr < 3 || last <= first) return false;
    const double T = (last - first) / (ncr - 1);
    St = P.D / (P.ulat * T);
    ClAmp = amp; CdMean = cdm;
    return true;
}
