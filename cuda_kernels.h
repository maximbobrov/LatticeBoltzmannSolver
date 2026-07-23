#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// Plain C++ interface to the CUDA LBM core. No CUDA types appear here, so
// ordinary MSVC-compiled .cpp files include this without nvcc. Implemented in
// cuda_kernels.cu; when the project is built without the CUDA toolkit
// (HAVE_CUDA undefined) the stubs at the bottom of main_app.cpp are used and
// gpuAvailable() reports false.
//
// The GPU executes the SAME lbmCellUpdate()/lbmMacroAt() from lattice.h as
// the CPU loop - one thread per cell instead of a for-loop. --gpucheck
// verifies the two match to accumulated roundoff.

#include "lattice.h"

bool        gpuAvailable();      // CUDA device found (cached after first call)
const char* gpuDeviceName();     // e.g. "RTX 5070 Laptop GPU", for the HUD

// Create the CUDA primary context NOW. Must be called BEFORE the OpenGL
// window exists: creating the context lazily (first cudaMalloc) while a GL
// context is already live crashes inside nvcuda64.dll on this laptop driver
// (Optimus; reproduced deterministically with --auto, fixed by this warmup).
bool gpuWarmup();

// Upload the initial distributions + cell flags, allocate the ping-pong pair.
// Frees any previous state first. False if no device.
bool gpuLbmInit(const double* f9, const unsigned char* flag, int NX, int NY);
void gpuLbmFree();

// Run n fused stream+collide steps entirely on the device. If FxHist/FyHist
// are non-null they receive the per-step momentum-exchange force on solid
// cells inside the box [fx0..fx1]x[fy0..fy1] (the cylinder), length n each.
void gpuLbmSteps(int n, const LbmParams& p,
                 int fx0, int fx1, int fy0, int fy1,
                 double* FxHist, double* FyHist);

// Compute rho,u on the device and download just the three macro fields
// (NX*NY each) - 3x less traffic than pulling all 9 distributions.
void gpuLbmMacro(const LbmParams& p, double* rho, double* ux, double* uy);

// Download the full distribution state (switching GPU -> CPU, --gpucheck).
void gpuLbmDownloadF(double* f9);

#endif // CUDA_KERNELS_H
