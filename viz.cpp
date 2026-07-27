#include "lbm.h"
#include "cuda_kernels.h"
#include <cstring>
#include <algorithm>
#include <vector>

// 2D visualization, same layout language as VortexGridFreeSolver/viz.cpp:
// scalar field (here: one textured quad built from the macro arrays), colour
// scale, HUD text block, key legend, validation panels (solver vs reference),
// and for the cylinder a Cl(t) strip with the measured Strouhal number.

static int    winW = 1280, winH = 900;
static double gFieldMax = 1.0;
static bool   gFieldSigned = true;
static GLuint gTex = 0;
static double gOx0, gOx1, gOy0, gOy1;   // current ortho box (mouse -> world)

// --- cell probe: click a cell to see its f_i "anatomy" ---------------------
bool probeOn = false;
static int probeI = 0, probeJ = 0;

// --- passive tracers: massless markers advected by the macro velocity ------
// Pure visualization (LBM itself has no particles - the f_i are distribution
// densities); tracers make the flow topology visible the way vortex-method
// particles do.
static std::vector<double> trX, trY;
static long long trLastStep = 0;
static long long trGeomKey  = -1;

static const char* fieldNames[3] = { "vorticity [1/step]", "|velocity| [lattice]", "rho - 1" };

static void drawText(float x, float y, void* font, const char* s)
{
    glRasterPos2f(x, y);
    for (const char* c = s; *c; ++c) glutBitmapCharacter(font, *c);
}

// blue (neg) - black (0) - yellow (pos), matching the reference project.
static void colorSigned(double val, double maxAbs, unsigned char* rgb)
{
    double t = (maxAbs > 1e-30) ? val / maxAbs * colorGain : 0.0;
    if (t > 1.0) t = 1.0; if (t < -1.0) t = -1.0;
    const double pos = t > 0 ? t : 0.0, neg = t < 0 ? -t : 0.0;
    rgb[0] = (unsigned char)(pos * 255.0);
    rgb[1] = (unsigned char)(pos * 255.0);
    rgb[2] = (unsigned char)(neg * 255.0);
}

static void colorPositive(double val, double maxVal, unsigned char* rgb)
{
    double t = (maxVal > 1e-30) ? val / maxVal * colorGain : 0.0;
    if (t > 1.0) t = 1.0; if (t < 0.0) t = 0.0;
    rgb[0] = (unsigned char)(t * 255.0);
    rgb[1] = (unsigned char)(t * 255.0);
    rgb[2] = 0;
}

static int idxCl(int v, int n, int per)
{
    if (per) return lbmWrap(v, n);
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// Build the field texture from the macro arrays: compute the selected scalar,
// normalize to the 98th percentile of |f| (so single extreme cells - e.g. the
// cavity lid corners - saturate instead of washing everything out), colormap.
static void buildFieldTexture()
{
    const int NX = P.NX, NY = P.NY, NN = NX * NY;
    static std::vector<double> fld;
    static std::vector<unsigned char> rgb;
    static std::vector<double> mag;
    fld.assign(NN, 0.0);
    rgb.assign((size_t)3 * NN, 0);
    mag.clear(); mag.reserve(NN);

    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
        {
            const int idx = y*NX + x;
            if (flag[idx] == CT_SOLID || flag[idx] == CT_MOVING) continue;
            double v = 0.0;
            if (fieldMode == 0)          // vorticity = dv/dx - du/dy, central
            {
                const int xm = idxCl(x-1, NX, P.perX), xp = idxCl(x+1, NX, P.perX);
                const int ym = idxCl(y-1, NY, P.perY), yp = idxCl(y+1, NY, P.perY);
                v = 0.5*(uy[y*NX + xp] - uy[y*NX + xm])
                  - 0.5*(ux[yp*NX + x] - ux[ym*NX + x]);
            }
            else if (fieldMode == 1) v = sqrt(ux[idx]*ux[idx] + uy[idx]*uy[idx]);
            else                     v = rho[idx] - 1.0;
            fld[idx] = v;
            mag.push_back(fabs(v));
        }

    double maxAbs = 1e-30;
    if (!mag.empty())
    {
        const size_t q = (size_t)(0.98 * (mag.size() - 1));
        std::nth_element(mag.begin(), mag.begin() + q, mag.end());
        maxAbs = fmax(mag[q], 1e-30);
    }
    const bool sgn = (fieldMode != 1);
    gFieldMax = maxAbs; gFieldSigned = sgn;

    for (int idx = 0; idx < NN; idx++)
    {
        unsigned char* c = &rgb[(size_t)3*idx];
        if (flag[idx] == CT_SOLID)      { c[0] = 70; c[1] = 70;  c[2] = 80;  }
        else if (flag[idx] == CT_MOVING){ c[0] = 95; c[1] = 95;  c[2] = 110; }
        else if (sgn) colorSigned(fld[idx], maxAbs, c);
        else          colorPositive(fld[idx], maxAbs, c);
    }

    glBindTexture(GL_TEXTURE_2D, gTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, NX, NY, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 rgb.data());
}

static void drawFieldColorScale(float x, float yBar, float w, float h)
{
    char s[48];
    const double sat = gFieldMax / (colorGain > 1e-9 ? colorGain : 1.0);

    glColor4f(0.0f, 0.0f, 0.0f, 0.55f);
    glBegin(GL_QUADS);
    glVertex2f(x - 6, yBar + h + 20); glVertex2f(x + w + 46, yBar + h + 20);
    glVertex2f(x + w + 46, yBar - 18); glVertex2f(x - 6, yBar - 18);
    glEnd();

    glColor3f(0.7f, 0.82f, 1.0f);
    drawText(x, yBar + h + 5, GLUT_BITMAP_HELVETICA_10, fieldNames[fieldMode]);

    const int NB = 64;
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= NB; i++)
    {
        const double fv = gFieldSigned ? (-1.0 + 2.0*i/NB) * sat : (double)i/NB * sat;
        unsigned char c[3];
        if (gFieldSigned) colorSigned(fv, gFieldMax, c);
        else              colorPositive(fv, gFieldMax, c);
        glColor3ub(c[0], c[1], c[2]);
        const float xx = x + w * i / NB;
        glVertex2f(xx, yBar); glVertex2f(xx, yBar + h);
    }
    glEnd();
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, yBar); glVertex2f(x + w, yBar);
    glVertex2f(x + w, yBar + h); glVertex2f(x, yBar + h);
    glEnd();

    glColor3f(0.85f, 0.85f, 0.9f);
    if (gFieldSigned)
    {
        sprintf(s, "%.2e", -sat); drawText(x - 4, yBar - 13, GLUT_BITMAP_HELVETICA_10, s);
        drawText(x + w*0.5f - 4, yBar - 13, GLUT_BITMAP_HELVETICA_10, "0");
        sprintf(s, "%.2e", sat);  drawText(x + w - 30, yBar - 13, GLUT_BITMAP_HELVETICA_10, s);
    }
    else
    {
        drawText(x - 2, yBar - 13, GLUT_BITMAP_HELVETICA_10, "0");
        sprintf(s, "%.2e", sat);  drawText(x + w - 30, yBar - 13, GLUT_BITMAP_HELVETICA_10, s);
    }
}

// One validation panel: solver profile (yellow) vs reference (cyan analytic
// curve, or white crosses for a Ghia scatter), with L2/max error + verdict.
static void drawValProfile(double x, double y, double w, double h, const ValProfile& p)
{
    char s[128];
    glColor4f(0.10f, 0.10f, 0.12f, 0.92f);
    glBegin(GL_QUADS);
    glVertex2d(x, y); glVertex2d(x + w, y); glVertex2d(x + w, y + h); glVertex2d(x, y + h);
    glEnd();

    double vr = p.vmax - p.vmin; if (vr < 1e-9) vr = 1e-9;

    if (p.vmin < 0.0 && p.vmax > 0.0)
    {
        const double yz = y + h * (0.0 - p.vmin) / vr;
        glColor3f(0.4f, 0.4f, 0.45f);
        glBegin(GL_LINES); glVertex2d(x, yz); glVertex2d(x + w, yz); glEnd();
    }

    if (p.hasExactCurve)
    {
        glColor3f(0.3f, 0.85f, 1.0f); glLineWidth(1.5f);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < p.n; i++)
            glVertex2d(x + w * p.t[i], y + h * (p.exact[i] - p.vmin) / vr);
        glEnd();
    }
    if (p.nref > 0)
    {
        glColor3f(0.9f, 0.9f, 0.95f); glLineWidth(1.2f);
        for (int i = 0; i < p.nref; i++)
        {
            const double px = x + w * p.tref[i], py = y + h * (p.vref[i] - p.vmin) / vr, d = 2.5;
            glBegin(GL_LINES);
            glVertex2d(px - d, py); glVertex2d(px + d, py);
            glVertex2d(px, py - d); glVertex2d(px, py + d);
            glEnd();
        }
    }

    glColor3f(1.0f, 0.9f, 0.3f); glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < p.n; i++)
    {
        double py = y + h * (p.solver[i] - p.vmin) / vr;
        if (py < y) py = y; if (py > y + h) py = y + h;
        glVertex2d(x + w * p.t[i], py);
    }
    glEnd();
    glLineWidth(1.0f);

    glColor3f(0.55f, 0.55f, 0.6f);
    glBegin(GL_LINE_LOOP);
    glVertex2d(x, y); glVertex2d(x + w, y); glVertex2d(x + w, y + h); glVertex2d(x, y + h);
    glEnd();

    glColor3f(0.8f, 0.85f, 0.95f);
    drawText((float)x, (float)(y + h + 4), GLUT_BITMAP_HELVETICA_10, p.title);

    const char* vd = validationVerdict(p.l2rel);
    float r, g, b;
    if (p.l2rel < 0.05)      { r = 0.3f; g = 1.0f;  b = 0.4f; }
    else if (p.l2rel < 0.15) { r = 1.0f; g = 0.85f; b = 0.3f; }
    else                     { r = 1.0f; g = 0.4f;  b = 0.3f; }
    sprintf(s, "L2 %.1f%%  max %.3f", p.l2rel * 100.0, p.linf);
    glColor3f(0.85f, 0.85f, 0.9f);
    drawText((float)x, (float)(y - 12), GLUT_BITMAP_HELVETICA_10, s);
    glColor3f(r, g, b);
    drawText((float)(x + 108), (float)(y - 12), GLUT_BITMAP_HELVETICA_10, vd);
    glColor3f(0.6f, 0.65f, 0.72f);
    drawText((float)(x + 146), (float)(y - 12), GLUT_BITMAP_HELVETICA_10,
             p.nref > 0 ? "ylw=solver  +=Ghia" : "ylw=solver  cyan=exact");
}

static bool fluidCell(int i, int j)
{
    if (i < 0 || i >= P.NX || j < 0 || j >= P.NY) return false;
    return flag[j*P.NX + i] == CT_FLUID;
}

static void reseedTracerOne(int k)
{
    for (int guard = 0; guard < 200; guard++)
    {
        double X, Y;
        if (P.scenario == SCN_CYLINDER)                 // feed from the inlet
        { X = 1.5; Y = 1.0 + (P.NY - 3.0) * rand() / (double)RAND_MAX; }
        else
        { X = (P.NX - 1.0) * rand() / (double)RAND_MAX;
          Y = (P.NY - 1.0) * rand() / (double)RAND_MAX; }
        if (fluidCell((int)X, (int)Y)) { trX[k] = X; trY[k] = Y; return; }
    }
}

static void reseedTracers()
{
    const int NT = 3000;
    trX.assign(NT, 1.0); trY.assign(NT, 1.0);
    for (int k = 0; k < NT; k++) reseedTracerOne(k);
    trLastStep = stepCount;
}

// Advance the tracers by however many solver steps passed since the last
// frame (Euler with the frame-frozen velocity field, substepped so a single
// move never exceeds a few cells). Reseeds on geometry change / reset.
static void advectTracers()
{
    const long long key = ((long long)P.scenario << 40)
                        ^ ((long long)P.NX << 20) ^ (long long)P.NY;
    if (key != trGeomKey || stepCount < trLastStep || trX.empty())
    { trGeomKey = key; reseedTracers(); }
    long long ds = stepCount - trLastStep;
    trLastStep = stepCount;
    if (ds <= 0) return;
    if (ds > 5000) ds = 5000;                 // long paused gaps: don't spin
    while (ds > 0)
    {
        const int sub = ds > 50 ? 50 : (int)ds;
        ds -= sub;
        for (size_t k = 0; k < trX.size(); k++)
        {
            const Vec2 u = sampleVel(trX[k], trY[k]);
            double X = trX[k] + u.x * sub, Y = trY[k] + u.y * sub;
            if (P.perX) { while (X < 0) X += P.NX; while (X >= P.NX) X -= P.NX; }
            if (P.perY) { while (Y < 0) Y += P.NY; while (Y >= P.NY) Y -= P.NY; }
            trX[k] = X; trY[k] = Y;
            if (!fluidCell((int)X, (int)Y)) reseedTracerOne((int)k);
        }
    }
}

// Lattice overlay: cell lines, strided so they stay >= ~7 px apart on screen.
static void drawGridLines()
{
    const double pxPerCell = winW / (gOx1 - gOx0 + 1e-30);
    int stride = 1;
    while (stride * pxPerCell < 7.0) stride *= 2;
    glColor4f(0.5f, 0.5f, 0.6f, 0.28f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int i = 0; i <= P.NX; i += stride)
    { glVertex2d(i, 0); glVertex2d(i, P.NY); }
    for (int j = 0; j <= P.NY; j += stride)
    { glVertex2d(0, j); glVertex2d(P.NX, j); }
    glEnd();
}

// Format a length in metres with an adaptive unit (m / mm / um).
static void fmtLen(double m, char* s)
{
    const double a = fabs(m);
    if (a >= 1.0)       sprintf(s, "%.3g m",  m);
    else if (a >= 1e-3) sprintf(s, "%.3g mm", m * 1e3);
    else if (a >= 1e-6) sprintf(s, "%.3g um", m * 1e6);
    else                sprintf(s, "%.2g mm", m * 1e3);
}

// Dimensional axis rulers around the domain: tick marks at 0..1 (5 per side)
// labelled with the physical length cell*dxPhys, plus x/y axis titles. Drawn
// in world (cell) coordinates - the ortho box keeps cells square, so the same
// physical scale applies to both axes for every scenario.
static void drawDimAxes()
{
    const int NX = P.NX, NY = P.NY;
    const double u = (NX < NY ? NX : NY);
    const double tick = 0.02 * u;
    char s[32];
    glColor3f(0.66f, 0.70f, 0.78f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int t = 0; t <= 4; t++)
    {
        const double cx = NX * t / 4.0, cy = NY * t / 4.0;
        glVertex2d(cx, 0.0);  glVertex2d(cx, -tick);       // bottom (x)
        glVertex2d(0.0, cy);  glVertex2d(-tick, cy);       // left (y)
    }
    glEnd();
    for (int t = 0; t <= 4; t++)
    {
        const double cx = NX * t / 4.0, cy = NY * t / 4.0;
        fmtLen(cx * P.dxPhys, s);
        drawText((float)(cx - 0.02 * u), (float)(-0.055 * u), GLUT_BITMAP_HELVETICA_10, s);
        fmtLen(cy * P.dxPhys, s);
        drawText((float)(-0.12 * u), (float)(cy - 0.008 * u), GLUT_BITMAP_HELVETICA_10, s);
    }
    glColor3f(0.8f, 0.84f, 0.92f);
    drawText((float)(0.46 * NX), (float)(-0.085 * u), GLUT_BITMAP_HELVETICA_12, "x");
    drawText((float)(-0.13 * u), (float)(0.48 * NY), GLUT_BITMAP_HELVETICA_12, "y");
}

static void drawArrow(double x, double y, double ex, double ey)
{
    glVertex2d(x, y);
    glVertex2d(x + ex, y + ey);
    const double len = sqrt(ex*ex + ey*ey);
    if (len > 1e-9)
    {
        const double hx = -ex / len, hy = -ey / len;
        const double px = -hy, py = hx;
        const double a = 0.25 * len;
        glVertex2d(x + ex, y + ey);
        glVertex2d(x + ex + a*(hx + 0.5*px), y + ey + a*(hy + 0.5*py));
        glVertex2d(x + ex, y + ey);
        glVertex2d(x + ex + a*(hx - 0.5*px), y + ey + a*(hy - 0.5*py));
    }
}

void reshape(int w, int h)
{
    winW = w; winH = h;
    glViewport(0, 0, w, h);
}

void initGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glGenTextures(1, &gTex);
    glBindTexture(GL_TEXTURE_2D, gTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    const int NX = P.NX, NY = P.NY;

    // Ortho projection: fit [0,NX]x[0,NY] with a margin, preserve aspect.
    {
        const double mx = 0.06 * NX, my = 0.06 * NY;
        double x0 = -mx, x1 = NX + mx, y0 = -my, y1 = NY + my;
        const double aspect = (double)winW / winH;
        const double wx = x1 - x0, wy = y1 - y0;
        if (wx / wy < aspect)
        {
            const double cx = 0.5*(x0 + x1), half = 0.5 * wy * aspect;
            x0 = cx - half; x1 = cx + half;
        }
        else
        {
            const double cy = 0.5*(y0 + y1), half = 0.5 * wx / aspect;
            y0 = cy - half; y1 = cy + half;
        }
        // zoom about the domain centre (Q/E)
        {
            const double cx = 0.5*(x0 + x1), cy = 0.5*(y0 + y1);
            const double hx = 0.5*(x1 - x0)/viewZoom, hy = 0.5*(y1 - y0)/viewZoom;
            x0 = cx - hx; x1 = cx + hx; y0 = cy - hy; y1 = cy + hy;
        }
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(x0, x1, y0, y1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gOx0 = x0; gOx1 = x1; gOy0 = y0; gOy1 = y1;   // for mouse picking
    }

    // --- scalar field as one textured quad ---
    buildFieldTexture();
    glEnable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2d(0,  0);
    glTexCoord2f(1, 0); glVertex2d(NX, 0);
    glTexCoord2f(1, 1); glVertex2d(NX, NY);
    glTexCoord2f(0, 1); glVertex2d(0,  NY);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    if (showGrid) drawGridLines();
    drawDimAxes();

    // --- domain outline + cylinder ---
    glColor3f(0.5f, 0.5f, 0.5f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2d(0, 0); glVertex2d(NX, 0); glVertex2d(NX, NY); glVertex2d(0, NY);
    glEnd();
    if (P.scenario == SCN_CYLINDER)
    {
        glColor3f(0.9f, 0.9f, 0.95f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 40; i++)
        {
            const double a = 2.0*PI*i/40;
            glVertex2d(P.cxc + 0.5*P.D*cos(a) + 0.5, P.cyc + 0.5*P.D*sin(a) + 0.5);
        }
        glEnd();
    }

    // --- velocity arrows on a subsampled grid ---
    if (showVelocity)
    {
        const int astep = (NX > NY ? NX : NY) / 32 > 0 ? (NX > NY ? NX : NY) / 32 : 1;
        glColor3f(1.0f, 1.0f, 0.3f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        for (int y = astep/2; y < NY; y += astep)
            for (int x = astep/2; x < NX; x += astep)
            {
                const int idx = y*NX + x;
                if (flag[idx] != CT_FLUID) continue;
                const double sc = 0.6 * astep / P.ulat * velArrowScale;
                drawArrow(x + 0.5, y + 0.5, ux[idx]*sc, uy[idx]*sc);
            }
        glEnd();
    }

    // --- passive tracers ---
    if (showTracers)
    {
        advectTracers();
        glPointSize(2.0f);
        glColor4f(0.95f, 0.95f, 1.0f, 0.75f);
        glBegin(GL_POINTS);
        for (size_t k = 0; k < trX.size(); k++) glVertex2d(trX[k], trY[k]);
        glEnd();
    }

    // --- cell probe: the 9 populations of the picked cell ---
    // Arrows show the NON-equilibrium part f_i - f_i^eq (the interesting bit:
    // it carries the viscous stress; f_i itself is ~w_i rho and looks equal in
    // all directions). Orange = surplus vs equilibrium, blue = deficit.
    double prRho = 0.0, prU = 0.0, prV = 0.0, prNeq[9] = {0};
    if (probeOn)
    {
        const int exd[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
        const int eyd[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
        const int NN = NX * NY, idx = probeJ*NX + probeI;
        lbmMacroAt(f.data(), flag.data(), P.kp(), idx, prRho, prU, prV);
        double mx = 1e-30;
        for (int i = 0; i < 9; i++)
        {
            prNeq[i] = f[(size_t)i*NN + idx] - lbmFeq(i, prRho, prU, prV);
            mx = fmax(mx, fabs(prNeq[i]));
        }
        glColor3f(1.0f, 1.0f, 1.0f); glLineWidth(1.5f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(probeI, probeJ);     glVertex2d(probeI + 1, probeJ);
        glVertex2d(probeI + 1, probeJ + 1); glVertex2d(probeI, probeJ + 1);
        glEnd();
        const double cx = probeI + 0.5, cy = probeJ + 0.5, sc = 4.0;
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        for (int i = 1; i < 9; i++)
        {
            const double en = sqrt((double)(exd[i]*exd[i] + eyd[i]*eyd[i]));
            const double L = sc * prNeq[i] / mx;      // signed length
            if (prNeq[i] >= 0) glColor4f(1.0f, 0.7f, 0.2f, 0.95f);
            else               glColor4f(0.35f, 0.7f, 1.0f, 0.95f);
            drawArrow(cx, cy, exd[i]/en * L, eyd[i]/en * L);
        }
        glEnd();
        glLineWidth(1.0f);
        // rest population: circle, radius by |neq|
        if (prNeq[0] >= 0) glColor4f(1.0f, 0.7f, 0.2f, 0.9f);
        else               glColor4f(0.35f, 0.7f, 1.0f, 0.9f);
        glBegin(GL_LINE_LOOP);
        for (int a = 0; a < 24; a++)
        {
            const double t = 2.0*PI*a/24, rr = 0.15 + 0.8*fabs(prNeq[0])/mx;
            glVertex2d(cx + rr*cos(t), cy + rr*sin(t));
        }
        glEnd();
    }

    // --- text overlay (screen space) ---
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    char line[256];
    // translucent backing so the HUD text stays legible over a bright field
    {
        float bx1 = (float)winW - 286.0f; if (bx1 < 420.0f) bx1 = 420.0f;
        glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
        glBegin(GL_QUADS);
        glVertex2f(0.0f, (float)winH); glVertex2f(bx1, (float)winH);
        glVertex2f(bx1, (float)winH - 102.0f); glVertex2f(0.0f, (float)winH - 102.0f);
        glEnd();
    }
    glColor3f(0.8f, 0.9f, 1.0f);
    sprintf(line, "field: %s   gain %.2f   grid %dx%d   steps/frame %d   core: %s   %.0f MLUPS   zoom %.2gx (Q/E)",
            fieldNames[fieldMode], colorGain, NX, NY, stepsPerFrame,
            useGpu ? gpuDeviceName() : "CPU", lastMLUPS, viewZoom);
    drawText(10, winH - 20, GLUT_BITMAP_HELVETICA_12, line);
    {
        char sx[32], sy[32], sd[32];
        fmtLen(NX * P.dxPhys, sx); fmtLen(NY * P.dxPhys, sy);
        fmtLen(P.dxPhys, sd);
        glColor3f(0.66f, 0.72f, 0.82f);
        sprintf(line, "domain: %s x %s   (scale %s/cell, 'U' to change; LBM is scale-free, this is a chosen mapping)",
                sx, sy, sd);
        drawText(10, winH - 20 - 16, GLUT_BITMAP_HELVETICA_10, line);
    }
    sprintf(line, "%s   step %lld   tau=%.4f  nu=%.4g  u_lat=%.3f (Ma %.2f)  Re=%.0f   mass drift %.1e",
            running ? "RUNNING" : "paused (Space)", stepCount, P.tau, P.nu,
            P.ulat, P.ulat*sqrt(3.0), P.Re, massDrift());
    drawText(10, winH - 54, GLUT_BITMAP_HELVETICA_12, line);
    glColor3f(0.95f, 0.85f, 0.4f);
    sprintf(line, "case: %s  -  %s", scenarioName(), scenarioDesc());
    drawText(10, winH - 72, GLUT_BITMAP_HELVETICA_12, line);

    // scenario-specific metric line
    glColor3f(0.5f, 0.95f, 0.6f);
    line[0] = 0;
    if (P.scenario == SCN_TAYLOR_GREEN)
    {
        const double ne = nuEff();
        if (ne > 0.0)
            sprintf(line, "nu_eff = %.6f  vs  nu = %.6f   (err %.2f%%)   E/E0 = %.3f",
                    ne, P.nu, 100.0*fabs(ne - P.nu)/P.nu,
                    kineticEnergy() / (tgE0 > 0 ? tgE0 : 1.0));
    }
    else if (P.scenario == SCN_CYLINDER)
    {
        double St, ClA, Cd;
        if (strouhal(St, ClA, Cd))
            sprintf(line, "Cd = %.3f   Cl_amp = %.3f   St = %.4f   (ref St ~ 0.16-0.17 at Re=100)",
                    Cd, ClA, St);
        else
            sprintf(line, "waiting for vortex shedding to develop... (Cl history %d steps)",
                    forceHistCount());
    }
    else if (P.scenario == SCN_CAVITY)
        sprintf(line, "compare vs Ghia tables: '-','=' set Re (100 or 1000); steady state takes ~%d k steps",
                (int)(30.0 * P.Lchar / P.ulat / 1000.0));
    else if (P.scenario == SCN_STEP)
    {
        const double xr = reattachX();
        if (xr > 0.0)
            sprintf(line, "reattachment x_r/S = %.2f   (Armaly'83: ~3 @Re=100, ~5 @Re=200, ~8 @Re=400)", xr);
        else
            sprintf(line, "recirculation bubble developing... (run a few 10k steps)");
    }
    else if (P.scenario == SCN_POROUS)
    {
        const double K = porousK();
        sprintf(line, "porosity = %.3f   <u>_sup = %.2e   K = %.2f cells^2   (Darcy: K must stay put under '-'/'=')",
                P.poroEps, K > 0.0 ? K * P.gx / P.nu : 0.0, K);
    }
    if (line[0]) drawText(10, winH - 92, GLUT_BITMAP_HELVETICA_12, line);

    drawFieldColorScale(14.0f, (float)(winH - 130), 200.0f, 14.0f);

    // --- probe value table (screen space, below the colour scale) ---
    if (probeOn)
    {
        const int exd[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
        const int eyd[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
        const int NN = NX * NY, idx = probeJ*NX + probeI;
        const float px = 14, pw = 320, rowH = 14;
        const float ptop = winH - 175.0f;
        glColor4f(0.0f, 0.0f, 0.0f, 0.65f);
        glBegin(GL_QUADS);
        glVertex2f(px - 6, ptop + 14); glVertex2f(px + pw, ptop + 14);
        glVertex2f(px + pw, ptop - rowH*11 - 4); glVertex2f(px - 6, ptop - rowH*11 - 4);
        glEnd();
        glColor3f(0.95f, 0.9f, 0.5f);
        sprintf(line, "probe (%d,%d) %s   rho=%.5f  u=(%.4f, %.4f)",
                probeI, probeJ,
                flag[idx] == CT_FLUID ? "FLUID" : flag[idx] == CT_SOLID ? "SOLID"
                : flag[idx] == CT_MOVING ? "MOVING" : flag[idx] == CT_INLET ? "INLET" : "OUTLET",
                prRho, prU, prV);
        drawText(px, ptop, GLUT_BITMAP_HELVETICA_10, line);
        glColor3f(0.75f, 0.8f, 0.9f);
        drawText(px, ptop - rowH, GLUT_BITMAP_HELVETICA_10,
                 "i   e_i        f_i          f_i^eq       f-feq (arrows)");
        for (int i = 0; i < 9; i++)
        {
            const double fi = f[(size_t)i*NN + idx];
            sprintf(line, "%d  (%2d,%2d)   %.6f   %.6f   %+.2e",
                    i, exd[i], eyd[i], fi, fi - prNeq[i], prNeq[i]);
            if (prNeq[i] >= 0) glColor3f(1.0f, 0.75f, 0.35f);
            else               glColor3f(0.45f, 0.75f, 1.0f);
            drawText(px, ptop - rowH*(2 + i), GLUT_BITMAP_HELVETICA_10, line);
        }
    }

    // --- legend panel (top-right) ---
    double legendBottomY = winH - 8;
    {
        struct Row { const char* key; const char* act; float r, g, b; };
        static const Row rows[] = {
            {"CONTROLS", "",                                    0.9f,0.8f,0.3f},
            {"X",     "toggle CPU / GPU core",                  0.5f,0.85f,1.0f},
            {"Space", "run / pause",                            1,1,1},
            {"B",     "one step",                               1,1,1},
            {"N",     "reset (re-init fields)",                 1,1,1},
            {"O",     "next case (TG/Pois/Cou/cav/cyl/step/porous)", 1,1,1},
            {"1 2 3", "field: vorticity / |u| / rho",           1,1,1},
            {"8 9",   "grid N /2  x2",                          1,1,1},
            {"Q E",   "zoom in / out (view)",                   1,1,1},
            {"- =",   "Re /2  x2",                              1,1,1},
            {"S",     "steps per frame 1/10/50/200/1000",       1,1,1},
            {"V",     "velocity arrows",                        1,1,1},
            {"G",     "lattice grid lines",                     1,1,1},
            {"P",     "tracer particles (flow markers)",        1,1,1},
            {"U",     "axis scale (mm per cell)",               1,1,1},
            {"LMB",   "probe cell: f_i anatomy (RMB off)",      1,1,1},
            {", .",   "arrow length -/+",                       1,1,1},
            {"[ ]",   "colour brightness -/+",                  1,1,1},
            {"K",     "validation table (console)",             1,1,1},
            {"T",     "self-tests (console)",                   1,1,1},
            {"R",     "rebuild",                                1,1,1},
            {"ESC",   "quit",                                   1,1,1},
            {"LEGEND", "",                                      0.9f,0.8f,0.3f},
            {"yellow", "solver profile",                        1.0f,0.9f,0.3f},
            {"cyan",  "exact solution",                         0.3f,0.85f,1.0f},
            {"+",     "Ghia benchmark points",                  0.9f,0.9f,0.95f},
            {"gray",  "solid cells (walls, cylinder)",          0.6f,0.6f,0.68f},
        };
        const int nrows = (int)(sizeof(rows)/sizeof(rows[0]));
        const float panelW = 262, rowH = 15.0f;
        const float px = winW - panelW - 8, py = winH - 8.0f;
        const float panelH = rowH * nrows + 10;
        legendBottomY = py - panelH;
        glColor4f(0.0f, 0.0f, 0.0f, 0.55f);
        glBegin(GL_QUADS);
        glVertex2f(px - 6, py + 4); glVertex2f(px + panelW, py + 4);
        glVertex2f(px + panelW, py - panelH); glVertex2f(px - 6, py - panelH);
        glEnd();
        for (int i = 0; i < nrows; i++)
        {
            const float y = py - 6 - rowH * i;
            glColor3f(rows[i].r, rows[i].g, rows[i].b);
            drawText(px, y, GLUT_BITMAP_HELVETICA_10, rows[i].key);
            if (rows[i].act[0])
            {
                glColor3f(0.8f, 0.85f, 0.95f);
                drawText(px + 46, y, GLUT_BITMAP_HELVETICA_10, rows[i].act);
            }
        }
    }

    // --- Cl(t) strip (cylinder) or validation panels (right column) ---
    double topY = legendBottomY - 20;
    if (P.scenario == SCN_CYLINDER && forceHistCount() > 10)
    {
        const double plotW = 340, plotH = 95;
        const double plotL = winW - plotW - 12;
        const double plotB = topY - plotH;
        glColor4f(0.12f, 0.12f, 0.12f, 0.9f);
        glBegin(GL_QUADS);
        glVertex2d(plotL, plotB); glVertex2d(plotL + plotW, plotB);
        glVertex2d(plotL + plotW, plotB + plotH); glVertex2d(plotL, plotB + plotH);
        glEnd();
        const int n = forceHistCount();
        double mn = 1e30, mx = -1e30;
        for (int i = 0; i < n; i++)
        { const double c = forceHistCl(i); mn = fmin(mn, c); mx = fmax(mx, c); }
        double vr = mx - mn; if (vr < 1e-12) vr = 1e-12;
        glColor3f(1.0f, 0.5f, 0.3f);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < n; i++)
            glVertex2d(plotL + plotW * i / (n - 1),
                       plotB + plotH * (forceHistCl(i) - mn) / vr);
        glEnd();
        glColor3f(0.6f, 0.6f, 0.6f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(plotL, plotB); glVertex2d(plotL + plotW, plotB);
        glVertex2d(plotL + plotW, plotB + plotH); glVertex2d(plotL, plotB + plotH);
        glEnd();
        glColor3f(0.75f, 0.8f, 0.9f);
        sprintf(line, "Cl(t), last %d steps  [%.3f .. %.3f]", n, mn, mx);
        drawText((float)plotL, (float)(plotB + plotH + 5), GLUT_BITMAP_HELVETICA_10, line);
        topY = plotB - 30;
    }
    {
        Validation val;
        if (computeValidation(val))
        {
            const double vpW = 340, vpL = winW - vpW - 12, vpH = 100;
            if (val.reMismatch)
            {
                glColor3f(1.0f, 0.5f, 0.3f);
                sprintf(line, "Ghia table is Re=%.0f - set Re accordingly", val.refRe);
                drawText((float)vpL, (float)(topY + 2), GLUT_BITMAP_HELVETICA_10, line);
            }
            for (int k = 0; k < val.nPanels; k++)
            {
                const double boxB = topY - 16 - vpH - k * (vpH + 42);
                if (boxB < 34) break;
                drawValProfile(vpL, boxB, vpW, vpH, val.panel[k]);
            }
        }
    }

    glutSwapBuffers();
}

// Left click: probe the cell under the cursor (screen -> world via the stored
// ortho box). Right click: probe off. In GPU mode the distributions live on
// the device, so refresh the host copy for the probe display.
void mouseClick(int button, int state, int x, int y)
{
    if (state != GLUT_DOWN) return;
    if (button == GLUT_RIGHT_BUTTON) { probeOn = false; glutPostRedisplay(); return; }
    if (button != GLUT_LEFT_BUTTON) return;
    const double wx = gOx0 + (gOx1 - gOx0) * x / winW;
    const double wy = gOy0 + (gOy1 - gOy0) * (winH - y) / winH;
    const int i = (int)floor(wx), j = (int)floor(wy);
    if (i < 0 || i >= P.NX || j < 0 || j >= P.NY) { probeOn = false; }
    else
    {
        probeOn = true; probeI = i; probeJ = j;
        if (useGpu) gpuLbmDownloadF(f.data());
    }
    glutPostRedisplay();
}
