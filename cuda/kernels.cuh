// CUDA kernels for FLIP fluid simulation
#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

constexpr int GFLUID=0, GAIR=1, GSOLID=2;

// helpers for device
__device__ inline float clampf_d(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}
__device__ inline int clampi_d(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

// integrate particles
__global__ void k_integrate(float* px, float* py, float* vx, float* vy,
                            int n, float dt, float gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vy[i] += dt * gravity;
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
}

// handle collisions with walls and obstacle
__global__ void k_collisions(float* px, float* py, float* vx, float* vy,
                             int n, float obsX, float obsY, float obsR,
                             float obsVX, float obsVY, float pRad,
                             float hh, int fNumX, int fNumY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = pRad;
    float minDist = obsR + r;
    float x = px[i], y = py[i];
    float dx = x - obsX, dy = y - obsY;
    if (dx*dx + dy*dy < minDist*minDist) { vx[i] = obsVX; vy[i] = obsVY; }
    float mnX = hh+r, mxX = (fNumX-1)*hh-r;
    float mnY = hh+r, mxY = (fNumY-1)*hh-r;
    if (x < mnX) { x = mnX; vx[i] = 0; }
    if (x > mxX) { x = mxX; vx[i] = 0; }
    if (y < mnY) { y = mnY; vy[i] = 0; }
    if (y > mxY) { y = mxY; vy[i] = 0; }
    px[i] = x; py[i] = y;
}

// count particles in each cell for hashing
__global__ void k_hashCount(const float* px, const float* py, int n,
                            int* counts, float pInvSp, int pNX, int pNY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xi = clampi_d((int)floorf(px[i]*pInvSp), 0, pNX-1);
    int yi = clampi_d((int)floorf(py[i]*pInvSp), 0, pNY-1);
    atomicAdd(&counts[xi*pNY+yi], 1);
}

// put particle ids into sorted order
__global__ void k_hashScatter(const float* px, const float* py, int n,
                              int* first, int* ids,
                              float pInvSp, int pNX, int pNY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xi = clampi_d((int)floorf(px[i]*pInvSp), 0, pNX-1);
    int yi = clampi_d((int)floorf(py[i]*pInvSp), 0, pNY-1);
    int slot = atomicSub(&first[xi*pNY+yi], 1) - 1;
    ids[slot] = i;
}

// push particles apart to avoid overlap
__global__ void k_separate(float* px, float* py,
                           float* cr, float* cg, float* cb,
                           const int* first, const int* ids,
                           int n, float pInvSp, int pNX, int pNY,
                           float minDist, float colorDiff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = px[i], y = py[i];
    float c0r = cr[i], c0g = cg[i], c0b = cb[i];
    int pxi = (int)floorf(x * pInvSp);
    int pyi = (int)floorf(y * pInvSp);
    int x0 = max(pxi-1,0), y0 = max(pyi-1,0);
    int x1 = min(pxi+1,pNX-1), y1 = min(pyi+1,pNY-1);
    float minD2 = minDist*minDist;
    float dxA=0, dyA=0;
    float drA=0, dgA=0, dbA=0;
    for (int xi=x0; xi<=x1; xi++) {
        for (int yi=y0; yi<=y1; yi++) {
            int cell = xi*pNY+yi;
            int fst = first[cell], lst = (cell+1 < pNX*pNY) ? first[cell+1] : n;
            for (int j=fst; j<lst; j++) {
                int idn = ids[j];
                if (idn == i) continue;
                float dx = px[idn]-x, dy = py[idn]-y;
                float d2 = dx*dx+dy*dy;
                if (d2 > minD2 || d2 == 0.0f) continue;
                float d = sqrtf(d2);
                float s = 0.5f*(minDist-d)/d;
                dxA -= dx*s; dyA -= dy*s;
                
                // Color diffusion
                float c1r = cr[idn], c1g = cg[idn], c1b = cb[idn];
                float avgR = (c0r + c1r) * 0.5f;
                float avgG = (c0g + c1g) * 0.5f;
                float avgB = (c0b + c1b) * 0.5f;
                drA += (avgR - c0r) * colorDiff;
                dgA += (avgG - c0g) * colorDiff;
                dbA += (avgB - c0b) * colorDiff;
            }
        }
    }
    atomicAdd(&px[i], dxA);
    atomicAdd(&py[i], dyA);
    atomicAdd(&cr[i], drA);
    atomicAdd(&cg[i], dgA);
    atomicAdd(&cb[i], dbA);
}

// compute particle density for each cell
__global__ void k_particleDensity(const float* px, const float* py, int n,
                                  float* pDens, float hh, float h1, float h2,
                                  int fNX, int fNY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int nn = fNY;
    float x = clampf_d(px[i], hh, (fNX-1)*hh);
    float y = clampf_d(py[i], hh, (fNY-1)*hh);
    int x0 = (int)floorf((x-h2)*h1); float tx = ((x-h2)-x0*hh)*h1;
    int x1 = min(x0+1, fNX-2);
    int y0 = (int)floorf((y-h2)*h1); float ty = ((y-h2)-y0*hh)*h1;
    int y1 = min(y0+1, fNY-2);
    float sx=1-tx, sy=1-ty;
    if(x0<fNX&&y0<fNY) atomicAdd(&pDens[x0*nn+y0], sx*sy);
    if(x1<fNX&&y0<fNY) atomicAdd(&pDens[x1*nn+y0], tx*sy);
    if(x1<fNX&&y1<fNY) atomicAdd(&pDens[x1*nn+y1], tx*ty);
    if(x0<fNX&&y1<fNY) atomicAdd(&pDens[x0*nn+y1], sx*ty);
}

// set cell types
__global__ void k_classifyInit(int* ct, const float* s, int nCells) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    ct[i] = (s[i]==0.0f) ? GSOLID : GAIR;
}
__global__ void k_classifyParticles(int* ct, const float* px, const float* py,
                                    int n, float fInvSp, int fNX, int fNY) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    int xi = clampi_d((int)floorf(px[i]*fInvSp), 0, fNX-1);
    int yi = clampi_d((int)floorf(py[i]*fInvSp), 0, fNY-1);
    int c = xi*fNY+yi;
    if (ct[c] == GAIR) ct[c] = GFLUID; // fluid if particle is inside
}

// save velocities before update
__global__ void k_savePrev(float* u, float* v, float* du, float* dv,
                           float* pu, float* pv, int nCells) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    pu[i]=u[i]; pv[i]=v[i]; du[i]=0; dv[i]=0; u[i]=0; v[i]=0;
}

// transfer particle velocities to grid
__global__ void k_p2g(const float* px, const float* py,
                      const float* pvel, float* fld, float* fldD,
                      int n, int comp, float hh, float h1, float h2,
                      int fNX, int fNY) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    int nn = fNY;
    float dxOff = (comp==0)?0.0f:h2, dyOff = (comp==0)?h2:0.0f;
    float x = clampf_d(px[i], hh, (fNX-1)*hh);
    float y = clampf_d(py[i], hh, (fNY-1)*hh);
    int x0 = min((int)floorf((x-dxOff)*h1), fNX-2); float tx=((x-dxOff)-x0*hh)*h1;
    int x1 = min(x0+1, fNX-2);
    int y0 = min((int)floorf((y-dyOff)*h1), fNY-2); float ty=((y-dyOff)-y0*hh)*h1;
    int y1 = min(y0+1, fNY-2);
    float sx=1-tx, sy=1-ty;
    float d0=sx*sy, d1=tx*sy, d2=tx*ty, d3=sx*ty;
    float pv = pvel[i];
    atomicAdd(&fld[x0*nn+y0], pv*d0); atomicAdd(&fldD[x0*nn+y0], d0);
    atomicAdd(&fld[x1*nn+y0], pv*d1); atomicAdd(&fldD[x1*nn+y0], d1);
    atomicAdd(&fld[x1*nn+y1], pv*d2); atomicAdd(&fldD[x1*nn+y1], d2);
    atomicAdd(&fld[x0*nn+y1], pv*d3); atomicAdd(&fldD[x0*nn+y1], d3);
}

// normalize grid velocities
__global__ void k_p2gNorm(float* fld, const float* fldD, int nCells) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    if (fldD[i] > 0.0f) fld[i] /= fldD[i];
}

// restore velocities for solid cells
__global__ void k_restoreSolid(float* u, float* v, const float* pu, const float* pv,
                               const int* ct, int fNX, int fNY) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int n = fNY;
    int total = fNX*fNY;
    if (idx >= total) return;
    int i = idx/n, j = idx%n;
    bool solid = (ct[i*n+j]==GSOLID);
    if (solid || (i>0 && ct[(i-1)*n+j]==GSOLID)) u[i*n+j] = pu[i*n+j];
    if (solid || (j>0 && ct[i*n+j-1]==GSOLID))   v[i*n+j] = pv[i*n+j];
}

// red-black pressure solver
__global__ void k_jacobiRB(float* u, float* v, float* p,
                           const float* s, const int* ct,
                           const float* pDens, float rest, int cd,
                           float overRelax, float cp,
                           int fNX, int fNY, int color) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int n = fNY;
    // map idx to (i,j) for this color
    int totalInner = (fNX-2)*(fNY-2);
    if (idx >= totalInner) return;
    int i = idx/(fNY-2)+1, j = idx%(fNY-2)+1;
    if (((i+j)&1) != color) return;
    if (ct[i*n+j]!=GFLUID) return;
    int center=i*n+j, left=(i-1)*n+j, right=(i+1)*n+j, bot=i*n+j-1, top=i*n+j+1;
    float sx0=s[left], sx1=s[right], sy0=s[bot], sy1=s[top];
    float sSum=sx0+sx1+sy0+sy1;
    if (sSum==0.0f) return;
    float div = u[right]-u[center]+v[top]-v[center];
    if (rest>0.0f && cd) {
        float comp = pDens[center]-rest;
        if (comp>0.0f) div -= comp;
    }
    float pVal = -div/sSum * overRelax;
    p[center] += cp*pVal;
    u[center] -= sx0*pVal;
    u[right]  += sx1*pVal;
    v[center] -= sy0*pVal;
    v[top]    += sy1*pVal;
}

// grid to particle transfer
__global__ void k_g2p(const float* px, const float* py,
                      float* pvel, const float* fld, const float* pfld,
                      const int* ct, int n_particles, int comp, float flipR,
                      float hh, float h1, float h2, int fNX, int fNY) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n_particles) return;
    int nn = fNY;
    float dxOff=(comp==0)?0.0f:h2, dyOff=(comp==0)?h2:0.0f;
    int offset=(comp==0)?nn:1;
    float x = clampf_d(px[i], hh, (fNX-1)*hh);
    float y = clampf_d(py[i], hh, (fNY-1)*hh);
    int x0=min((int)floorf((x-dxOff)*h1),fNX-2); float tx=((x-dxOff)-x0*hh)*h1;
    int x1=min(x0+1,fNX-2);
    int y0=min((int)floorf((y-dyOff)*h1),fNY-2); float ty=((y-dyOff)-y0*hh)*h1;
    int y1=min(y0+1,fNY-2);
    float sx=1-tx, sy=1-ty;
    float d0=sx*sy, d1=tx*sy, d2=tx*ty, d3=sx*ty;
    int nr0=x0*nn+y0, nr1=x1*nn+y0, nr2=x1*nn+y1, nr3=x0*nn+y1;
    float v0=(ct[nr0]!=GAIR||ct[nr0-offset]!=GAIR)?1.0f:0.0f;
    float v1=(ct[nr1]!=GAIR||ct[nr1-offset]!=GAIR)?1.0f:0.0f;
    float v2=(ct[nr2]!=GAIR||ct[nr2-offset]!=GAIR)?1.0f:0.0f;
    float v3=(ct[nr3]!=GAIR||ct[nr3-offset]!=GAIR)?1.0f:0.0f;
    float vOld = pvel[i];
    float d = v0*d0+v1*d1+v2*d2+v3*d3;
    if (d > 0.0f) {
        float picV=(v0*d0*fld[nr0]+v1*d1*fld[nr1]+v2*d2*fld[nr2]+v3*d3*fld[nr3])/d;
        float corr=(v0*d0*(fld[nr0]-pfld[nr0])+v1*d1*(fld[nr1]-pfld[nr1])
                   +v2*d2*(fld[nr2]-pfld[nr2])+v3*d3*(fld[nr3]-pfld[nr3]))/d;
        pvel[i] = (1.0f-flipR)*picV + flipR*(vOld+corr);
    }
}

// update particle colors
__global__ void k_updateParticleColors(float* cr, float* cg, float* cb,
                                       const float* px, const float* py,
                                       const float* pDens, float d0,
                                       float h1, int numP, int fNX, int fNY) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= numP) return;
    cr[i] = clampf_d(cr[i]-0.01f, 0, 1);
    cg[i] = clampf_d(cg[i]-0.01f, 0, 1);
    cb[i] = clampf_d(cb[i]+0.01f, 0, 1);
    int xi = clampi_d((int)floorf(px[i]*h1), 1, fNX-1);
    int yi = clampi_d((int)floorf(py[i]*h1), 1, fNY-1);
    int cellNr = xi*fNY+yi;
    if (d0 > 0.0f) {
        float rel = pDens[cellNr]/d0;
        if (rel < 0.7f) { cr[i]=0.8f; cg[i]=0.8f; cb[i]=1.0f; }
    }
}

// color cells for drawing
__global__ void k_updateCellColors(float* cc, const int* ct,
                                   const float* pDens, float rest,
                                   int nCells) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    cc[3*i]=0; cc[3*i+1]=0; cc[3*i+2]=0;
    if (ct[i]==GSOLID) { cc[3*i]=0.5f; cc[3*i+1]=0.5f; cc[3*i+2]=0.5f; }
    else if (ct[i]==GFLUID) {
        float d = pDens[i];
        if (rest>0) d /= rest;
        // sci color scheme
        float val = fminf(fmaxf(d, 0.0f), 1.9999f);
        float dd = 2.0f;
        val = val / dd;
        float m = 0.25f;
        int num = (int)floorf(val/m);
        float sl = (val-num*m)/m;
        float r=0,g=0,b=0;
        if (num==0) { g=sl; b=1; }
        else if (num==1) { g=1; b=1-sl; }
        else if (num==2) { r=sl; g=1; }
        else { r=1; g=1-sl; }
        cc[3*i]=r; cc[3*i+1]=g; cc[3*i+2]=b;
    }
}

// carve obstacle out of the fluid grid
__global__ void k_carveObstacle(float* s, float* u, float* v,
                                float ox, float oy, float orad,
                                float ovx, float ovy,
                                float hh, int fNX, int fNY) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int n = fNY;
    int total = fNX*fNY;
    if (idx >= total) return;
    int i = idx/n, j = idx%n;
    if (i<1||i>=fNX-2||j<1||j>=fNY-2) return;
    s[i*n+j] = 1.0f;
    float dx = (i+0.5f)*hh - ox;
    float dy = (j+0.5f)*hh - oy;
    if (dx*dx+dy*dy < orad*orad) {
        s[i*n+j] = 0.0f;
        u[i*n+j] = ovx; u[(i+1)*n+j] = ovx;
        v[i*n+j] = ovy; v[i*n+j+1] = ovy;
    }
}

// parallel prefix scan using hillis-steele
__global__ void k_scanBlockInclusive(const int* in, int* out,
                                    int* blockSums, int n) {
    extern __shared__ int tmp[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tmp[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int add = (tid >= off) ? tmp[tid - off] : 0;
        __syncthreads();
        tmp[tid] += add;
        __syncthreads();
    }
    if (gid < n) out[gid] = tmp[tid];
    if (tid == blockDim.x - 1 && blockSums) blockSums[blockIdx.x] = tmp[tid];
}

// add block offsets for final scan result
__global__ void k_addBlockOffsets(int* out, const int* scannedBlockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (blockIdx.x > 0) out[gid] += scannedBlockSums[blockIdx.x - 1];
}

// zero out integer arrays
__global__ void k_zeroInt(int* arr, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) arr[i] = 0;
}
// zero out float arrays
__global__ void k_zeroFloat(float* arr, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) arr[i] = 0.0f;
}

// sum up fluid density for rest density calculation
__global__ void k_sumFluidDensity(const float* pDens, const int* ct,
                                  int nCells, float* outSum, int* outCount) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    if (ct[i] == GFLUID) {
        atomicAdd(outSum, pDens[i]);
        atomicAdd(outCount, 1);
    }
}

// setup for pressure solver
__global__ void k_prepPressure(float* p, float* prevU, float* prevV,
                               const float* u, const float* v, int nCells) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= nCells) return;
    p[i] = 0.0f;
    prevU[i] = u[i];
    prevV[i] = v[i];
}

// pack particle data for opengl interop
__global__ void k_packParticles(const float* px, const float* py,
                                const float* cr, const float* cg, const float* cb,
                                float2* outPos, float3* outCol, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    outPos[i] = make_float2(px[i], py[i]);
    outCol[i] = make_float3(cr[i], cg[i], cb[i]);
}
