#include "lbm.h"
#include "cuda_kernels.h"
#include <chrono>
#include <cstring>

// LBM entry point + UI state. Architecture mirrors VortexGridFreeSolver's
// main_app.cpp: GLUT loop, keyboard-driven scenario/core switching, CLI gates.

int    fieldMode     = 0;      // 0 vorticity, 1 |u|, 2 rho-1
bool   showVelocity  = true;
bool   showGrid      = false;  // 'G': lattice lines
bool   showTracers   = false;  // 'P': passive flow markers
bool   running       = false;
bool   useGpu        = false;
int    stepsPerFrame = 50;
double colorGain     = 1.0;
double velArrowScale = 1.0;
double lastMLUPS     = 0.0;
double viewZoom      = 1.0;

void rebuildAll()
{
    applyScenario();
    lbmAllocate();
    buildFlags();
    lbmInitFields();
    if (useGpu)
    {
        gpuLbmFree();
        if (!gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
        {
            useGpu = false;
            printf("GPU init failed - falling back to CPU.\n");
        }
    }
    printf("case: %-20s %dx%d  Re=%.0f  nu=%.4g  tau=%.4f  u_lat=%.3f  core=%s\n",
           scenarioName(), P.NX, P.NY, P.Re, P.nu, P.tau, P.ulat,
           useGpu ? "GPU" : "CPU");
}

// Refresh the macro fields for display/validation from whichever core is live.
static void updateDisplayFields()
{
    if (useGpu)
    {
        gpuLbmMacro(P.kp(), rho.data(), ux.data(), uy.data());
        ghostFillMacro();
        if (probeOn) gpuLbmDownloadF(f.data());   // probe shows raw f_i
    }
    else updateMacro();
}

static void doSteps(int n)
{
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (useGpu)
    {
        static std::vector<double> fxh, fyh;
        double *pfx = nullptr, *pfy = nullptr;
        int x0 = 0, x1 = -1, y0 = 0, y1 = -1;
        if (P.scenario == SCN_CYLINDER)
        {
            x0 = (int)(P.cxc - P.D) - 2; x1 = (int)(P.cxc + P.D) + 2;
            y0 = (int)(P.cyc - P.D) - 2; y1 = (int)(P.cyc + P.D) + 2;
            if (x0 < 1) x0 = 1; if (x1 > P.NX - 2) x1 = P.NX - 2;
            if (y0 < 1) y0 = 1; if (y1 > P.NY - 2) y1 = P.NY - 2;
            fxh.resize(n); fyh.resize(n);
            pfx = fxh.data(); pfy = fyh.data();
        }
        gpuLbmSteps(n, P.kp(), x0, x1, y0, y1, pfx, pfy);
        stepCount += n;
        if (pfx)
            for (int s = 0; s < n; s++) pushForceSample(fxh[s], fyh[s]);
    }
    else lbmStepsCPU(n);
    const double dt = std::chrono::duration<double>(clk::now() - t0).count();
    if (dt > 1e-9) lastMLUPS = (double)P.NX * P.NY * n / dt / 1e6;
}

static void idle()
{
    if (!running) return;
    doSteps(stepsPerFrame);
    updateDisplayFields();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int, int)
{
    switch (key)
    {
    case '1': fieldMode = 0; break;
    case '2': fieldMode = 1; break;
    case '3': fieldMode = 2; break;
    case 'v': case 'V': showVelocity = !showVelocity; break;
    case 'g': case 'G': showGrid     = !showGrid;     break;
    case 'p': case 'P': showTracers  = !showTracers;  break;
    case ' ': running = !running; break;
    case 'b': case 'B':
        doSteps(1);
        updateDisplayFields();
        break;
    case 'n': case 'N':
        running = false;
        rebuildAll();
        updateDisplayFields();
        break;
    case 'o': case 'O':
        running = false;
        P.scenario = (P.scenario + 1) % SCN_COUNT;
        P.Re = scenarioDefaultRe(P.scenario);
        rebuildAll();
        updateDisplayFields();
        break;
    case 'x': case 'X':
        if (!useGpu)
        {
            if (!gpuAvailable()) { printf("no CUDA device available.\n"); break; }
            if (gpuLbmInit(f.data(), flag.data(), P.NX, P.NY))
            {
                useGpu = true;
                printf("core: GPU (%s)\n", gpuDeviceName());
            }
        }
        else
        {
            gpuLbmDownloadF(f.data());   // carry the state back seamlessly
            gpuLbmFree();
            useGpu = false;
            printf("core: CPU\n");
        }
        updateDisplayFields();
        break;
    case '8':
        running = false;
        P.N = P.N > 32 ? P.N / 2 : 32;
        rebuildAll(); updateDisplayFields();
        break;
    case '9':
        running = false;
        P.N = P.N < 1024 ? P.N * 2 : 1024;
        rebuildAll(); updateDisplayFields();
        break;
    case '-':                       // change Re on the fly (tau/gx recomputed;
    case '=':                       // fields kept except TG, whose reference
    {                               // decay depends on nu from t=0)
        P.Re = key == '-' ? fmax(1.0, 0.5 * P.Re) : fmin(1e5, 2.0 * P.Re);
        applyScenario();
        if (P.scenario == SCN_TAYLOR_GREEN) { running = false; lbmInitFields(); }
        forceHistReset();           // Cd/Cl normalization unchanged, period isn't
        printf("Re = %.0f  ->  nu=%.4g  tau=%.4f\n", P.Re, P.nu, P.tau);
        updateDisplayFields();
        break;
    }
    case 's': case 'S':
    {
        const int cyc[5] = { 1, 10, 50, 200, 1000 };
        int cur = 0;
        for (int i = 0; i < 5; i++) if (stepsPerFrame == cyc[i]) cur = i;
        stepsPerFrame = cyc[(cur + 1) % 5];
        printf("steps per frame: %d\n", stepsPerFrame);
        break;
    }
    case 'u': case 'U':                                 // cycle the physical
    {                                                   // cell size (axis scale)
        const double lv[5] = { 1e-5, 5e-5, 1e-4, 5e-4, 1e-3 };
        int cur = 2;
        for (int i = 0; i < 5; i++) if (fabs(P.dxPhys - lv[i]) < 1e-12) cur = i;
        P.dxPhys = lv[(cur + 1) % 5];
        printf("axis scale: %.4g mm/cell  ->  domain %.4g x %.4g mm\n",
               P.dxPhys*1e3, P.NX*P.dxPhys*1e3, P.NY*P.dxPhys*1e3);
        break;
    }
    case 'q': case 'Q': viewZoom = fmin(20.0, viewZoom * 1.25); break;   // zoom in
    case 'e': case 'E': viewZoom = fmax(0.25, viewZoom * 0.8);  break;   // zoom out
    case ',': velArrowScale *= 0.8;  break;
    case '.': velArrowScale *= 1.25; break;
    case '[': colorGain *= 0.8;  break;
    case ']': colorGain *= 1.25; break;
    case 'k': case 'K':
        updateDisplayFields();
        printValidation();
        break;
    case 't': case 'T':
        running = false;
        runSelfTests();
        rebuildAll();               // self-tests trample the state arrays
        updateDisplayFields();
        break;
    case 'r': case 'R':
        running = false;
        rebuildAll(); updateDisplayFields();
        break;
    case 27: exit(0);
    }
    glutPostRedisplay();
}

// --auto: hands-off GUI soak test - replays the exact interactive sequence
// (GPU toggle, run, cycle every scenario, toggle back mid-run) on timers and
// exits 0 if the app survives. Used to reproduce/verify the nvcuda64 crash.
struct AutoEv { int delayMs; unsigned char key; };
static const AutoEv autoSeq[] = {
    { 2000, 'x' },   // GPU on (paused, Taylor-Green)
    { 2000, ' ' },   // run on GPU
    { 1000, 'g' },   // lattice grid overlay
    { 1000, 'p' },   // tracer particles
    { 800,  'q' },   // zoom in
    { 800,  'q' },
    { 1200, 'e' },   // zoom out
    { 4000, 'o' },   // -> Poiseuille (o pauses)
    { 1000, ' ' },
    { 4000, 'o' },   // -> Couette
    { 1000, ' ' },
    { 4000, 'o' },   // -> cavity
    { 1000, ' ' },
    { 4000, 'o' },   // -> cylinder
    { 1000, ' ' },
    { 6000, 'x' },   // GPU -> CPU mid-run on the cylinder
    { 4000, 'x' },   // CPU -> GPU again
    { 4000, 'o' },   // -> backward-facing step
    { 1000, ' ' },
    { 4000, 'o' },   // -> porous medium
    { 1000, ' ' },
    { 4000, 'o' },   // -> Taylor-Green
    { 1000, ' ' },
    { 6000, 0   }    // survived -> exit 0
};
static int autoIdx = 0;
static void autoTimer(int)
{
    const AutoEv& e = autoSeq[autoIdx];
    if (e.key == 0)
    {
        printf("AUTO: full sequence survived - OK\n");
        exit(0);
    }
    printf("AUTO: key '%c' (stage %d, %s, core=%s)\n",
           e.key, autoIdx, scenarioName(), useGpu ? "GPU" : "CPU");
    keyboard(e.key, 0, 0);
    autoIdx++;
    glutTimerFunc(autoSeq[autoIdx].delayMs, autoTimer, 0);
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);   // gates run redirected - don't lose
                                        // output if something crashes
    // CLI gates (no GUI): --selftest / --gpucheck / --bench [n]
    for (int a = 1; a < argc; a++)
    {
        if (!strcmp(argv[a], "--selftest"))
            return runSelfTests() ? 1 : 0;
        if (!strcmp(argv[a], "--gpucheck"))
            return runGpuCheck() ? 1 : 0;
        if (!strcmp(argv[a], "--cavity"))
        {
            const double re = (a + 1 < argc) ? atof(argv[a + 1]) : 100.0;
            return runCavityGate(re) ? 1 : 0;
        }
        if (!strcmp(argv[a], "--cylinder"))
        {
            const double re = (a + 1 < argc) ? atof(argv[a + 1]) : 100.0;
            return runCylinderGate(re) ? 1 : 0;
        }
        if (!strcmp(argv[a], "--step"))
        {
            const double re = (a + 1 < argc) ? atof(argv[a + 1]) : 200.0;
            return runStepGate(re) ? 1 : 0;
        }
        if (!strcmp(argv[a], "--porous"))
            return runPorousGate() ? 1 : 0;
        if (!strcmp(argv[a], "--order"))
            return runOrderStudy() ? 1 : 0;
        if (!strcmp(argv[a], "--xtest"))
            return runXTest() ? 1 : 0;
        if (!strcmp(argv[a], "--bench"))
        {
            const int n = (a + 1 < argc) ? atoi(argv[a + 1]) : 1000;
            P.scenario = SCN_CAVITY;
            P.Re = scenarioDefaultRe(P.scenario);
            applyScenario(); lbmAllocate(); buildFlags(); lbmInitFields();
            runBench(n);
            return 0;
        }
    }

    bool autoDrive = false;
    for (int a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--auto")) autoDrive = true;

    P.Re = scenarioDefaultRe(P.scenario);

    // CUDA context must exist BEFORE the GL window (see gpuWarmup decl).
    gpuWarmup();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1280, 900);
    glutCreateWindow("LBM D2Q9 - CPU/CUDA solver");
    initGL();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouseClick);
    glutIdleFunc(idle);

    rebuildAll();
    updateDisplayFields();
    printf("\nSpace=run  O=next case  X=CPU/GPU  T=self-tests  K=validation  (see legend)\n");
    if (gpuAvailable()) printf("CUDA device: %s\n", gpuDeviceName());
    else                printf("CUDA: not available (CPU only)\n");

    if (autoDrive) glutTimerFunc(2000, autoTimer, 0);

    glutMainLoop();
    return 0;
}

// --- stubs when built without the CUDA toolkit (HAVE_CUDA undefined) --------
#ifndef HAVE_CUDA
bool        gpuAvailable()  { return false; }
const char* gpuDeviceName() { return "n/a (built without CUDA)"; }
bool        gpuWarmup()     { return false; }
bool gpuLbmInit(const double*, const unsigned char*, int, int) { return false; }
void gpuLbmFree() {}
void gpuLbmSteps(int, const LbmParams&, int, int, int, int, double*, double*) {}
void gpuLbmMacro(const LbmParams&, double*, double*, double*) {}
void gpuLbmDownloadF(double*) {}
#endif
