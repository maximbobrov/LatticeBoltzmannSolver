#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA LBM core. All physics lives in lattice.h (lbmCellUpdate/lbmMacroAt),
// compiled here as __device__ code - the kernels are thin per-cell wrappers.
// State stays resident on the device across steps; only macro fields (per
// frame) or the force history (per step batch) come back to the host.

static bool  gProbed = false, gOk = false;
static char  gName[256] = "n/a";

static double*        dA    = nullptr;   // ping-pong distribution buffers
static double*        dB    = nullptr;
static unsigned char* dFlag = nullptr;
static double*        dMac  = nullptr;   // rho | ux | uy, 3*NN
static double*        dFh   = nullptr;   // per-step force history, 2*n
static int            gNX = 0, gNY = 0;
static int            gFhCap = 0;

bool gpuAvailable()
{
    if (!gProbed)
    {
        gProbed = true;
        int n = 0;
        gOk = (cudaGetDeviceCount(&n) == cudaSuccess) && n > 0;
        if (gOk)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            snprintf(gName, sizeof(gName), "%s", prop.name);
        }
    }
    return gOk;
}

const char* gpuDeviceName() { return gName; }

bool gpuWarmup()
{
    if (!gpuAvailable()) return false;
    return cudaFree(0) == cudaSuccess;   // forces primary-context creation
}

// ---------------------------------------------------------------------------

__global__ void stepKernel(const double* fin, double* fout,
                           const unsigned char* flag, LbmParams p)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.NX || y >= p.NY) return;
    lbmCellUpdate(fin, fout, flag, p, x, y);
}

__global__ void macroKernel(const double* f, const unsigned char* flag,
                            LbmParams p, double* mac)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int NN = p.NX * p.NY;
    if (idx >= NN) return;
    double r, u, v;
    lbmMacroAt(f, flag, p, idx, r, u, v);
    mac[idx] = r; mac[NN + idx] = u; mac[2*NN + idx] = v;
}

// Momentum-exchange force on CT_SOLID cells inside the given box: every
// fluid->solid link transfers 2 f*_i e_i per step. Same formula as
// cylForceCPU(); accumulated with atomics into out2[2*s].
__global__ void forceKernel(const double* f, const unsigned char* flag,
                            LbmParams p, int x0, int x1, int y0, int y1,
                            double* out2)
{
    const int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
    const int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
    const int x = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x > x1 || y > y1) return;
    const int NN = p.NX * p.NY, idx = y*p.NX + x;
    if (flag[idx] != CT_FLUID) return;
    double fx = 0.0, fy = 0.0;
    for (int i = 1; i < 9; i++)
        if (flag[(y + ey[i])*p.NX + (x + ex[i])] == CT_SOLID)
        {
            fx += 2.0 * f[i*NN + idx] * ex[i];
            fy += 2.0 * f[i*NN + idx] * ey[i];
        }
    if (fx != 0.0) atomicAdd(&out2[0], fx);
    if (fy != 0.0) atomicAdd(&out2[1], fy);
}

// ---------------------------------------------------------------------------

void gpuLbmFree()
{
    if (dA)    cudaFree(dA);
    if (dB)    cudaFree(dB);
    if (dFlag) cudaFree(dFlag);
    if (dMac)  cudaFree(dMac);
    if (dFh)   cudaFree(dFh);
    dA = dB = dMac = dFh = nullptr;
    dFlag = nullptr;
    gNX = gNY = gFhCap = 0;
}

bool gpuLbmInit(const double* f9, const unsigned char* flag, int NX, int NY)
{
    if (!gpuAvailable()) return false;
    gpuLbmFree();
    const size_t NN = (size_t)NX * NY;
    if (cudaMalloc(&dA,    9*NN*sizeof(double))        != cudaSuccess ||
        cudaMalloc(&dB,    9*NN*sizeof(double))        != cudaSuccess ||
        cudaMalloc(&dFlag, NN*sizeof(unsigned char))   != cudaSuccess ||
        cudaMalloc(&dMac,  3*NN*sizeof(double))        != cudaSuccess)
    {
        gpuLbmFree();
        printf("gpuLbmInit: cudaMalloc failed (%dx%d)\n", NX, NY);
        return false;
    }
    cudaMemcpy(dA, f9, 9*NN*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dFlag, flag, NN*sizeof(unsigned char), cudaMemcpyHostToDevice);
    gNX = NX; gNY = NY;
    return true;
}

void gpuLbmSteps(int n, const LbmParams& p,
                 int fx0, int fx1, int fy0, int fy1,
                 double* FxHist, double* FyHist)
{
    if (!dA || n <= 0) return;
    const bool wantForce = (FxHist != nullptr) && (fx1 >= fx0) && (fy1 >= fy0);
    if (wantForce && gFhCap < n)
    {
        if (dFh) cudaFree(dFh);
        cudaMalloc(&dFh, (size_t)2*n*sizeof(double));
        gFhCap = n;
    }
    if (wantForce) cudaMemset(dFh, 0, (size_t)2*n*sizeof(double));

    const dim3 blk(16, 16);
    const dim3 grd((gNX + 15)/16, (gNY + 15)/16);
    const dim3 fgrd((fx1 - fx0 + 16)/16, (fy1 - fy0 + 16)/16);
    for (int s = 0; s < n; s++)
    {
        stepKernel<<<grd, blk>>>(dA, dB, dFlag, p);
        double* t = dA; dA = dB; dB = t;
        if (wantForce)
            forceKernel<<<fgrd, blk>>>(dA, dFlag, p, fx0, fx1, fy0, fy1, dFh + 2*s);
    }
    if (wantForce)
    {
        static double* hFh = nullptr; static int hCap = 0;
        if (hCap < n)
        {
            free(hFh);
            hFh = (double*)malloc((size_t)2*n*sizeof(double));
            hCap = n;
        }
        cudaMemcpy(hFh, dFh, (size_t)2*n*sizeof(double), cudaMemcpyDeviceToHost);
        for (int s = 0; s < n; s++)
        {
            FxHist[s] = hFh[2*s];
            FyHist[s] = hFh[2*s + 1];
        }
    }
    cudaDeviceSynchronize();
}

void gpuLbmMacro(const LbmParams& p, double* rhoH, double* uxH, double* uyH)
{
    if (!dA) return;
    const size_t NN = (size_t)gNX * gNY;
    macroKernel<<<(int)((NN + 255)/256), 256>>>(dA, dFlag, p, dMac);
    cudaMemcpy(rhoH, dMac,        NN*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uxH,  dMac + NN,   NN*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uyH,  dMac + 2*NN, NN*sizeof(double), cudaMemcpyDeviceToHost);
}

void gpuLbmDownloadF(double* f9)
{
    if (!dA) return;
    const size_t NN = (size_t)gNX * gNY;
    cudaMemcpy(f9, dA, 9*NN*sizeof(double), cudaMemcpyDeviceToHost);
}
