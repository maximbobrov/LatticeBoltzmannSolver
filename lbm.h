#ifndef LBM_H
#define LBM_H

// LBM D2Q9 solver — app-wide header.
// Architecture mirrors ../VortexGridFreeSolver: plain C++ + freeglut GUI,
// optional CUDA core (same math via lattice.h). Full description: README.md.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lattice.h"

constexpr double PI = 3.14159265358979323846;

// Validation scenarios, each with its own reference (see README.md). Cycled with 'O'.
enum Scenario
{
    SCN_TAYLOR_GREEN = 0,  // periodic decaying vortex array: exact nu from E(t)
    SCN_POISEUILLE   = 1,  // force-driven channel: exact parabola
    SCN_COUETTE      = 2,  // moving-lid channel: exact linear profile
    SCN_CAVITY       = 3,  // lid-driven cavity: Ghia et al. (1982) tables
    SCN_CYLINDER     = 4,  // channel flow past a cylinder: Karman street, St/Cd/Cl
    SCN_STEP         = 5,  // backward-facing step: reattachment length vs Armaly
    SCN_POROUS       = 6,  // random porous medium: Darcy law, permeability K
    SCN_COUNT        = 7
};

struct Vec2 { double x = 0.0, y = 0.0; };

struct Params
{
    int    scenario = SCN_TAYLOR_GREEN;
    int    N        = 128;     // base resolution (NY); NX derived per scenario
    double Re       = 200.0;   // Reynolds number on the scenario's Lchar
    double ulat     = 0.05;    // velocity scale in lattice units (Ma = ulat*sqrt(3))

    // Derived by applyScenario():
    int    NX = 128, NY = 128;
    int    perX = 1, perY = 1;
    double nu = 0.032, tau = 0.596;
    double gx = 0.0;           // Poiseuille body force
    double uLid = 0.0;         // CT_MOVING wall speed (+x)
    double uin  = 0.0;         // CT_INLET speed (+x)
    double Lchar = 128.0;      // length the Re is built on (cells)
    double D = 16.0, cxc = 0.0, cyc = 0.0;   // cylinder/grain diameter, center
    double stepS = 0.0;        // backward-facing step height (cells)
    double poroEps = 1.0;      // porous medium: measured porosity
    int    inParabolic = 0;    // inlet: parabolic profile (step scenario)
    double inY0 = 0.0, inY1 = 1.0;

    // Physical scale for the axes. LBM is dimensionless (dx = dt = 1 lattice
    // unit); the physical size is a user-chosen mapping - ONE number, the
    // physical length of a single cell, fixes it for EVERY scenario. Default
    // 0.1 mm/cell puts the domains at a plausible lab scale (a 128-cell box is
    // 12.8 mm). Cycled with 'U'. Persisted across scenarios (display-only).
    double dxPhys = 1.0e-4;    // metres per lattice cell

    LbmParams kp() const       // kernel-parameter view of the current state
    {
        LbmParams p;
        p.NX = NX; p.NY = NY; p.perX = perX; p.perY = perY;
        p.tau = tau; p.gx = gx; p.gy = 0.0;
        p.uwx = uLid; p.uwy = 0.0; p.uin = uin;
        p.inParabolic = inParabolic; p.inY0 = inY0; p.inY1 = inY1;
        return p;
    }
};
extern Params P;

// --- lbm_core.cpp: state + CPU core --------------------------------------
extern std::vector<double>        f, fnew;      // [9*NX*NY], f[i*NN+idx]
extern std::vector<double>        rho, ux, uy;  // macro fields (display/validation)
extern std::vector<unsigned char> flag;         // CellType per cell
extern long long stepCount;
extern double    tgE0;                          // Taylor-Green initial energy

void   lbmAllocate();
void   lbmInitFields();       // per-scenario IC; resets step/history/E0/mass
void   lbmStepsCPU(int n);
void   updateMacro();         // rho,u from f (CPU) + ghost fill for walls
void   ghostFillMacro();      // wall-cell macro = mirror of fluid (viz/sampling)
double totalMass();           // sum of f over non-wall cells
double massDrift();           // (M - M0)/M0 since init
double kineticEnergy();       // 0.5 sum u^2 over fluid (from macro arrays)
double nuEff();               // TG: viscosity measured from the energy decay
Vec2   sampleVel(double X, double Y);   // bilinear sample in cell coords
void   cylForceCPU(double& Fx, double& Fy);     // momentum exchange on cylinder
void   pushForceSample(double Fx, double Fy);   // append Cd/Cl to the history
void   forceHistReset();
int    forceHistCount();
double forceHistCl(int i);    // i = 0 .. count-1, oldest first
double forceHistCd(int i);
bool   strouhal(double& St, double& ClAmp, double& CdMean);
double reattachX();           // step: reattachment length x_r/S (-1 if none)
double porousK();             // porous: Darcy permeability K [cells^2]

// --- scenarios.cpp ---------------------------------------------------------
void        applyScenario();          // derive NX/NY/nu/tau/gx/... from P
void        buildFlags();             // fill the CellType field
const char* scenarioName();
const char* scenarioDesc();
double      scenarioDefaultRe(int scn);

// --- validation.cpp --------------------------------------------------------
// One comparison panel: solver profile along a line vs a reference (analytic
// curve for TG/Couette/Poiseuille, Ghia scatter for the cavity).
struct ValProfile
{
    char   title[80];
    int    n;              // solver curve sample count
    double t[160];         // independent coordinate, normalized to [0,1]
    double solver[160];    // solver value along the line (normalized by ulat)
    double exact[160];     // analytic value (valid when hasExactCurve)
    bool   hasExactCurve;
    int    nref;           // benchmark scatter points (Ghia); 0 if none
    double tref[24], vref[24];
    double vmin, vmax;     // plot range (includes 0)
    double l2rel;          // relative L2 error vs reference
    double linf;           // max abs error (velocities already normalized)
};
struct Validation
{
    int    nPanels;        // 0 (cylinder), 1 (channel/TG) or 2 (cavity)
    bool   reMismatch;     // cavity: P.Re matches no Ghia table
    double refRe;          // Re of the table actually used
    ValProfile panel[2];
};
int         computeValidation(Validation& v);
void        printValidation();                 // console table ('K')
const char* validationVerdict(double l2rel);   // "GOOD"/"FAIR"/"POOR"
int         runSelfTests();                    // gates; returns #failed
int         runGpuCheck();                     // CPU vs GPU; returns #failed
void        runBench(int nsteps);              // MLUPS CPU (и GPU если есть)
int         runCavityGate(double Re);          // cavity to steady state vs Ghia
int         runCylinderGate(double Re);        // Karman street, St vs reference
int         runStepGate(double Re);            // reattachment length vs Armaly
int         runPorousGate();                   // Darcy: K invariant under nu
int         runXTest();                        // headless GUI 'X'-toggle repro

// --- viz.cpp ---------------------------------------------------------------
void initGL();
void display();
void reshape(int w, int h);
void mouseClick(int button, int state, int x, int y);  // probe cell picking
extern bool probeOn;            // a cell probe is active (draws f_i anatomy)

// --- main_app.cpp: UI state ------------------------------------------------
void keyboard(unsigned char key, int x, int y);
void rebuildAll();
extern int    fieldMode;        // 0 vorticity, 1 |u|, 2 rho-1
extern bool   showVelocity;
extern bool   showGrid;         // lattice lines overlay ('G')
extern bool   showTracers;      // passive tracer particles ('P')
extern bool   running;
extern bool   useGpu;
extern int    stepsPerFrame;
extern double colorGain;
extern double velArrowScale;
extern double lastMLUPS;

#endif // LBM_H
