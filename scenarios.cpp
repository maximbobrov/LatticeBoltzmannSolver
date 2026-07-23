#include "lbm.h"
#include <algorithm>

// Scenario setup: derived parameters (applyScenario) + cell-type field
// (buildFlags). Walls are one-cell fences; with halfway bounce-back the
// physical wall plane sits half a cell inside the last fluid cell, so the
// effective channel/cavity size is N-2 cells.

double scenarioDefaultRe(int scn)
{
    switch (scn)
    {
    case SCN_TAYLOR_GREEN: return 200.0;
    case SCN_POISEUILLE:   return 50.0;
    case SCN_COUETTE:      return 50.0;
    case SCN_CAVITY:       return 100.0;   // matches the Ghia table
    case SCN_CYLINDER:     return 100.0;   // Karman street, St ~ 0.16-0.17
    case SCN_STEP:         return 200.0;   // Armaly range: x_r/S ~ 5
    case SCN_POROUS:       return 10.0;    // creeping pore flow (Re on grain)
    }
    return 100.0;
}

void applyScenario()
{
    const int N = P.N;
    P.gx = 0.0; P.uLid = 0.0; P.uin = 0.0;
    P.perX = 0; P.perY = 0;
    P.inParabolic = 0; P.inY0 = 0.0; P.inY1 = 1.0;
    switch (P.scenario)
    {
    case SCN_TAYLOR_GREEN:
        P.NX = P.NY = N; P.perX = P.perY = 1;
        P.Lchar = N;
        P.nu = P.ulat * N / P.Re;
        break;
    case SCN_POISEUILLE:
    {
        P.NX = N; P.NY = N; P.perX = 1;
        const double H = N - 2;               // wall-plane to wall-plane
        P.Lchar = H;
        P.nu = P.ulat * H / P.Re;
        P.gx = 8.0 * P.nu * P.ulat / (H * H); // makes u_max = ulat exactly
        break;
    }
    case SCN_COUETTE:
    {
        P.NX = N; P.NY = N; P.perX = 1;
        const double H = N - 2;
        P.Lchar = H;
        P.nu = P.ulat * H / P.Re;
        P.uLid = P.ulat;
        break;
    }
    case SCN_CAVITY:
    {
        P.NX = P.NY = N;
        const double L = N - 2;
        P.Lchar = L;
        P.nu = P.ulat * L / P.Re;
        P.uLid = P.ulat;
        break;
    }
    case SCN_CYLINDER:
        P.NY = N; P.NX = 3 * N;
        P.D = N / 8.0;                        // blockage D/H = 1/8
        P.Lchar = P.D;
        P.nu = P.ulat * P.D / P.Re;
        P.uin = P.ulat;
        P.cxc = 0.8 * N;                      // ~17 D of wake to the outlet
        P.cyc = 0.5 * N + 0.31;               // symmetry-breaking offset
        break;
    case SCN_STEP:
    {
        // Gartling-style: sudden 1:2 expansion right at the inlet plane. The
        // lower half of the left boundary is the step face, the upper half a
        // parabolic (developed) inflow with mean u = ulat. Re is built on the
        // hydraulic diameter of the inlet channel, D_h = 2 h_in = NY-2.
        P.NY = N; P.NX = 4 * N;
        const int    S  = (N - 2) / 2;        // step height, cells
        const double Hf = N - 2.0;
        P.stepS = S;
        P.Lchar = Hf;
        P.nu  = P.ulat * Hf / P.Re;
        P.uin = P.ulat;
        P.inParabolic = 1;
        P.inY0 = S + 0.5;                     // step-top wall plane
        P.inY1 = N - 1.5;                     // ceiling wall plane
        break;
    }
    case SCN_POROUS:
        // Fully periodic box of random grains, driven by a body force:
        // Darcy's law <u_sup> = K g / nu. Re is built on the grain diameter.
        P.NX = P.NY = N; P.perX = P.perY = 1;
        P.D = N / 8.0;                        // grain diameter
        P.Lchar = P.D;
        P.nu = P.ulat * P.D / P.Re;
        P.gx = 4.0 * P.nu * P.ulat / (P.D * P.D);
        break;
    }
    P.tau = 3.0 * P.nu + 0.5;
    if (P.tau < 0.51)
        printf("WARNING: tau = %.4f close to 1/2 - BGK may go unstable. "
               "Lower Re ('-') or refine the grid ('9').\n", P.tau);
}

void buildFlags()
{
    const int NX = P.NX, NY = P.NY;
    std::fill(flag.begin(), flag.end(), (unsigned char)CT_FLUID);
    switch (P.scenario)
    {
    case SCN_TAYLOR_GREEN:
        break;                                          // fully periodic
    case SCN_POISEUILLE:
        for (int x = 0; x < NX; x++)
        { flag[x] = CT_SOLID; flag[(NY-1)*NX + x] = CT_SOLID; }
        break;
    case SCN_COUETTE:
        for (int x = 0; x < NX; x++)
        { flag[x] = CT_SOLID; flag[(NY-1)*NX + x] = CT_MOVING; }
        break;
    case SCN_CAVITY:
        for (int x = 0; x < NX; x++)
        { flag[x] = CT_SOLID; flag[(NY-1)*NX + x] = CT_MOVING; }   // lid spans full width
        for (int y = 1; y < NY-1; y++)
        { flag[y*NX] = CT_SOLID; flag[y*NX + NX-1] = CT_SOLID; }
        break;
    case SCN_CYLINDER:
    {
        for (int x = 0; x < NX; x++)
        { flag[x] = CT_SOLID; flag[(NY-1)*NX + x] = CT_SOLID; }
        for (int y = 1; y < NY-1; y++)
        { flag[y*NX] = CT_INLET; flag[y*NX + NX-1] = CT_OUTLET; }
        const double r2 = 0.25 * P.D * P.D;
        const int bx0 = (int)(P.cxc - P.D), bx1 = (int)(P.cxc + P.D) + 1;
        const int by0 = (int)(P.cyc - P.D), by1 = (int)(P.cyc + P.D) + 1;
        for (int y = by0; y <= by1; y++)
            for (int x = bx0; x <= bx1; x++)
            {
                const double dx = x - P.cxc, dy = y - P.cyc;
                if (dx*dx + dy*dy <= r2 && x >= 1 && x < NX-1 && y >= 1 && y < NY-1)
                    flag[y*NX + x] = CT_SOLID;
            }
        break;
    }
    case SCN_STEP:
    {
        const int S = (int)P.stepS;
        for (int x = 0; x < NX; x++)
        { flag[x] = CT_SOLID; flag[(NY-1)*NX + x] = CT_SOLID; }
        for (int y = 1; y < NY-1; y++)
        {
            flag[y*NX] = (y <= S) ? (unsigned char)CT_SOLID    // step face
                                  : (unsigned char)CT_INLET;   // inflow above
            flag[y*NX + NX-1] = CT_OUTLET;
        }
        break;
    }
    case SCN_POROUS:
    {
        // Random overlapping grains until ~28% solid fraction. Fixed seed so
        // the geometry (and hence K) is reproducible across runs/cores.
        srand(20260722);
        const double rad = 0.5 * P.D, r2 = rad * rad;
        const long long total = (long long)NX * NY;
        long long solid = 0;
        int guard = 0;
        while (solid < (long long)(0.28 * total) && guard++ < 4000)
        {
            const double cx = NX * (rand() / (double)RAND_MAX);
            const double cy = NY * (rand() / (double)RAND_MAX);
            const int R = (int)rad + 1;
            for (int dy = -R; dy <= R; dy++)
                for (int dx = -R; dx <= R; dx++)
                    if (dx*dx + dy*dy <= r2)
                    {
                        const int x = lbmWrap((int)cx + dx, NX);
                        const int y = lbmWrap((int)cy + dy, NY);
                        unsigned char& c = flag[y*NX + x];
                        if (c == CT_FLUID) { c = CT_SOLID; solid++; }
                    }
        }
        P.poroEps = 1.0 - (double)solid / total;
        break;
    }
    }
}

const char* scenarioName()
{
    switch (P.scenario)
    {
    case SCN_TAYLOR_GREEN: return "Taylor-Green";
    case SCN_POISEUILLE:   return "Poiseuille";
    case SCN_COUETTE:      return "Couette";
    case SCN_CAVITY:       return "lid-driven cavity";
    case SCN_CYLINDER:     return "cylinder in channel";
    case SCN_STEP:         return "backward-facing step";
    case SCN_POROUS:       return "porous medium";
    }
    return "?";
}

const char* scenarioDesc()
{
    switch (P.scenario)
    {
    case SCN_TAYLOR_GREEN: return "periodic vortex array, exact decay exp(-2 nu k^2 t) - tests collision+streaming, no walls";
    case SCN_POISEUILLE:   return "body-force channel, exact parabola - tests bounce-back walls + Guo forcing";
    case SCN_COUETTE:      return "top lid drags fluid, exact linear profile - tests the moving-wall bounce-back";
    case SCN_CAVITY:       return "moving lid, compare center-line profiles vs Ghia (Re=100/1000)";
    case SCN_CYLINDER:     return "uniform inflow, Karman street at Re=100: St~0.16-0.17, watch Cl(t)";
    case SCN_STEP:         return "1:2 sudden expansion, parabolic inflow: x_r/S vs Armaly'83 (~5 at Re=200)";
    case SCN_POROUS:       return "random grains, force-driven: Darcy law, K in HUD must not depend on Re";
    }
    return "";
}
