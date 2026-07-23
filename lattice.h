#ifndef LATTICE_H
#define LATTICE_H

// D2Q9 lattice + the single-cell update rule, shared VERBATIM between the CPU
// core (lbm_core.cpp) and the CUDA kernels (cuda_kernels.cu): the GPU executes
// literally this same lbmCellUpdate(), so any CPU/GPU mismatch beyond FMA
// roundoff is a bug (checked by --gpucheck).
//
// Velocity set (index i at its offset e_i), opp = bounce-back partner:
//     6 2 5         w0    = 4/9
//     3 0 1         w1..4 = 1/9   (axis)
//     7 4 8         w5..8 = 1/36  (diagonal)
// cs^2 = 1/3, so 1/cs^2 = 3, 1/(2 cs^4) = 4.5, 1/(2 cs^2) = 1.5.
// nu = cs^2 (tau - 1/2)  =>  tau = 3 nu + 1/2.

#ifdef __CUDACC__
#define LBM_HD __host__ __device__ __forceinline__
#else
#define LBM_HD inline
#endif

enum CellType
{
    CT_FLUID  = 0,   // interior: full stream + collide
    CT_SOLID  = 1,   // static wall / obstacle: halfway bounce-back
    CT_MOVING = 2,   // moving wall (cavity/Couette lid): bounce-back + 6 w (e.uw)
    CT_INLET  = 3,   // equilibrium inlet, rho = 1, u = (uin, 0)
    CT_OUTLET = 4    // first-order outflow: copy the x-1 neighbour's populations
};

// Everything the update rule needs, as a plain aggregate (no CUDA types) so it
// can be passed by value to kernels and shared with host code unchanged.
struct LbmParams
{
    int    NX, NY;
    int    perX, perY;    // periodic wrap in x / y (non-periodic edges must be
                          // fully covered by SOLID/MOVING/INLET/OUTLET cells)
    double tau;           // BGK relaxation time
    double gx, gy;        // body force (Guo forcing)
    double uwx, uwy;      // CT_MOVING wall velocity
    double uin;           // CT_INLET velocity (+x): uniform, or the MEAN of
    int    inParabolic;   //   a parabolic profile when inParabolic is set
    double inY0, inY1;    //   (wall planes bounding the inlet span)
};

// Inlet velocity at row y: uniform p.uin, or the parabola with mean p.uin
// (peak 1.5 uin) spanning the wall planes [inY0, inY1] - developed channel
// inflow for the backward-facing step.
LBM_HD double lbmInletU(const LbmParams& p, int y)
{
    if (!p.inParabolic) return p.uin;
    double s = (y - p.inY0) / (p.inY1 - p.inY0 + 1e-30);
    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;
    return 6.0 * p.uin * s * (1.0 - s);
}

LBM_HD int lbmWrap(int v, int n) { return v < 0 ? v + n : (v >= n ? v - n : v); }

LBM_HD double lbmFeq(int i, double rho, double ux, double uy)
{
    const int    ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
    const int    ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
    const double w [9] = { 4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
                           1.0/36, 1.0/36, 1.0/36, 1.0/36 };
    const double eu = ex[i]*ux + ey[i]*uy;
    return w[i]*rho*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*(ux*ux + uy*uy));
}

// Macroscopic (rho, u) of one cell from the distributions, with the Guo
// half-force correction u = (sum f e + F/2)/rho. Boundary-cell types report
// their nominal values (they carry no meaningful f).
LBM_HD void lbmMacroAt(const double* f, const unsigned char* flag, LbmParams p,
                       int idx, double& rho, double& ux, double& uy)
{
    const int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
    const int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
    const unsigned char ct = flag[idx];
    if (ct == CT_SOLID)  { rho = 1.0; ux = 0.0;   uy = 0.0;   return; }
    if (ct == CT_MOVING) { rho = 1.0; ux = p.uwx; uy = p.uwy; return; }
    if (ct == CT_INLET)  { rho = 1.0; ux = lbmInletU(p, idx / p.NX); uy = 0.0; return; }
    const int NN = p.NX * p.NY;
    double r = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < 9; i++)
    {
        const double fi = f[i*NN + idx];
        r += fi; mx += fi*ex[i]; my += fi*ey[i];
    }
    rho = r;
    ux = (mx + 0.5*p.gx) / r;
    uy = (my + 0.5*p.gy) / r;
}

// One fused stream(pull) + collide update of cell (x,y): gather the incoming
// populations from the 8 neighbours (with halfway bounce-back off SOLID/MOVING
// cells resolved during the gather), compute (rho,u), BGK-relax toward
// equilibrium with the Guo forcing term, write the post-collision state to
// fout. fin holds the post-collision state of the previous step; buffers are
// ping-ponged by the caller. Layout: f[i*NX*NY + y*NX + x] (SoA, coalesced).
LBM_HD void lbmCellUpdate(const double* fin, double* fout,
                          const unsigned char* flag, LbmParams p, int x, int y)
{
    const int    ex [9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
    const int    ey [9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
    const int    opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
    const double w  [9] = { 4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
                            1.0/36, 1.0/36, 1.0/36, 1.0/36 };
    const int NN = p.NX * p.NY, idx = y*p.NX + x;
    const unsigned char ct = flag[idx];

    if (ct == CT_SOLID || ct == CT_MOVING)          // geometry cells: carry
    {                                               // state through unchanged
        for (int i = 0; i < 9; i++) fout[i*NN + idx] = fin[i*NN + idx];
        return;
    }
    if (ct == CT_INLET)                             // equilibrium inlet, rho=1
    {
        const double ui = lbmInletU(p, y);
        for (int i = 0; i < 9; i++) fout[i*NN + idx] = lbmFeq(i, 1.0, ui, 0.0);
        return;
    }
    if (ct == CT_OUTLET)                            // first-order outflow
    {
        const int src = y*p.NX + (x - 1);
        for (int i = 0; i < 9; i++) fout[i*NN + idx] = fin[i*NN + src];
        return;
    }

    // --- streaming (pull): f_i comes from x - e_i ---
    double fi[9];
    for (int i = 0; i < 9; i++)
    {
        int xn = x - ex[i], yn = y - ey[i];
        if (p.perX) xn = lbmWrap(xn, p.NX);
        if (p.perY) yn = lbmWrap(yn, p.NY);
        // Non-periodic edges are always fenced by boundary cells, so xn,yn are
        // in range here; clamp defensively so a bad flag field can't crash.
        if (xn < 0) xn = 0; if (xn >= p.NX) xn = p.NX - 1;
        if (yn < 0) yn = 0; if (yn >= p.NY) yn = p.NY - 1;
        const unsigned char cn = flag[yn*p.NX + xn];
        if (cn == CT_SOLID)                          // halfway bounce-back:
            fi[i] = fin[opp[i]*NN + idx];            // own outgoing comes back
        else if (cn == CT_MOVING)                    // + wall-momentum term
            fi[i] = fin[opp[i]*NN + idx]
                  + 6.0*w[i]*(ex[i]*p.uwx + ey[i]*p.uwy);   // 2 w rho0 (e.uw)/cs^2, rho0=1
        else
            fi[i] = fin[i*NN + yn*p.NX + xn];
    }

    // --- macroscopic values (Guo half-force correction) ---
    double r = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < 9; i++) { r += fi[i]; mx += fi[i]*ex[i]; my += fi[i]*ey[i]; }
    const double ux = (mx + 0.5*p.gx) / r;
    const double uy = (my + 0.5*p.gy) / r;

    // --- BGK collision + Guo forcing ---
    const double om  = 1.0 / p.tau;
    const double usq = ux*ux + uy*uy;
    for (int i = 0; i < 9; i++)
    {
        const double eu  = ex[i]*ux + ey[i]*uy;
        const double feq = w[i]*r*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*usq);
        const double Fi  = w[i]*(1.0 - 0.5*om)
                         * ( 3.0*((ex[i]-ux)*p.gx + (ey[i]-uy)*p.gy)
                           + 9.0*eu*(ex[i]*p.gx + ey[i]*p.gy) );
        fout[i*NN + idx] = fi[i] - om*(fi[i] - feq) + Fi;
    }
}

#endif // LATTICE_H
