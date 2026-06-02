// FlipFluidCUDA class implementation - launches kernels from kernels.cuh
#include "flip_fluid_cuda.cuh"
#include "kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLK 256
#define GRID(n) (((n)+BLK-1)/BLK)

// ── Constructor ───────────────────────────────────────────────
FlipFluidCUDA::FlipFluidCUDA(float density_, float width, float height,
                             float spacing, float particle_radius, int max_particles) {
    density = density_;
    fNumX = (int)floorf(width/spacing)+1;
    fNumY = (int)floorf(height/spacing)+1;
    h = fmaxf(width/fNumX, height/fNumY);
    fInvSpacing = 1.0f/h;
    fNumCells = fNumX*fNumY;

    maxParticles = max_particles;
    numParticles = 0;
    particleRestDensity = 0.0f;
    particleRadius = particle_radius;
    pInvSpacing = 1.0f/(2.2f*particleRadius);
    pNumX = (int)floorf(width*pInvSpacing)+1;
    pNumY = (int)floorf(height*pInvSpacing)+1;
    pNumCells = pNumX*pNumY;

    // Allocate device grid arrays
    CUDA_CHECK(cudaMalloc(&d_u, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_du, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prevU, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prevV, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cellType, fNumCells*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellColor, 3*fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleDensity, fNumCells*sizeof(float)));

    // Allocate device particle arrays
    CUDA_CHECK(cudaMalloc(&d_particlePosX, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particlePosY, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleVelX, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleVelY, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleColorR, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleColorG, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particleColorB, maxParticles*sizeof(float)));

    // Spatial hash
    CUDA_CHECK(cudaMalloc(&d_numCellParticles, pNumCells*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_firstCellParticle, (pNumCells+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellParticleIds, maxParticles*sizeof(int)));

    // Reduction temporaries
    CUDA_CHECK(cudaMalloc(&d_reductionBuf, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_reductionIntBuf, sizeof(int)));

    // Zero everything
    CUDA_CHECK(cudaMemset(d_u, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_du, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_prevU, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_prevV, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cellColor, 0, 3*fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_particleDensity, 0, fNumCells*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_particleVelX, 0, maxParticles*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_particleVelY, 0, maxParticles*sizeof(float)));

    // Host staging
    h_particlePosX = new float[maxParticles];
    h_particlePosY = new float[maxParticles];
    h_particleColorR = new float[maxParticles];
    h_particleColorG = new float[maxParticles];
    h_particleColorB = new float[maxParticles];
    h_cellColor = new float[3*fNumCells];

    // Timing events (one pair per stage per substep)
    for (int i = 0; i < NUM_TIMING_STAGES; i++) {
        for (int s = 0; s < MAX_SUBSTEPS; s++) {
            CUDA_CHECK(cudaEventCreate(&evStart[i][s]));
            CUDA_CHECK(cudaEventCreate(&evStop[i][s]));
        }
        accumMs[i] = 0.0f;
    }
    CUDA_CHECK(cudaEventCreate(&evFrameStart));
    CUDA_CHECK(cudaEventCreate(&evFrameStop));
    accumFrames = 0;
    lastNumPressureIters = 0;
    lastNumSubSteps = 1;
}

FlipFluidCUDA::~FlipFluidCUDA() {
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_du); cudaFree(d_dv);
    cudaFree(d_prevU); cudaFree(d_prevV); cudaFree(d_p); cudaFree(d_s);
    cudaFree(d_cellType); cudaFree(d_cellColor); cudaFree(d_particleDensity);
    cudaFree(d_particlePosX); cudaFree(d_particlePosY);
    cudaFree(d_particleVelX); cudaFree(d_particleVelY);
    cudaFree(d_particleColorR); cudaFree(d_particleColorG); cudaFree(d_particleColorB);
    cudaFree(d_numCellParticles); cudaFree(d_firstCellParticle); cudaFree(d_cellParticleIds);
    cudaFree(d_reductionBuf); cudaFree(d_reductionIntBuf);
    delete[] h_particlePosX; delete[] h_particlePosY;
    delete[] h_particleColorR; delete[] h_particleColorG; delete[] h_particleColorB;
    delete[] h_cellColor;
    for (int i = 0; i < NUM_TIMING_STAGES; i++) {
        for (int s = 0; s < MAX_SUBSTEPS; s++) {
            cudaEventDestroy(evStart[i][s]); cudaEventDestroy(evStop[i][s]);
        }
    }
    cudaEventDestroy(evFrameStart); cudaEventDestroy(evFrameStop);
}

// ── Upload ────────────────────────────────────────────────────
void FlipFluidCUDA::uploadParticles(const float* posX, const float* posY,
                                    const float* velX, const float* velY,
                                    const float* colR, const float* colG, const float* colB,
                                    int nP) {
    numParticles = nP;
    size_t sz = nP*sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_particlePosX, posX, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particlePosY, posY, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleVelX, velX, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleVelY, velY, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleColorR, colR, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleColorG, colG, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleColorB, colB, sz, cudaMemcpyHostToDevice));
}

void FlipFluidCUDA::uploadGrid(const float* s_host) {
    CUDA_CHECK(cudaMemcpy(d_s, s_host, fNumCells*sizeof(float), cudaMemcpyHostToDevice));
}

// ── Timing ────────────────────────────────────────────────────
void FlipFluidCUDA::startTiming(TimingStage s, int sub) {
    if (sub < 0 || sub >= MAX_SUBSTEPS) sub = 0;
    CUDA_CHECK(cudaEventRecord(evStart[s][sub]));
}
void FlipFluidCUDA::stopTiming(TimingStage s, int sub) {
    if (sub < 0 || sub >= MAX_SUBSTEPS) sub = 0;
    CUDA_CHECK(cudaEventRecord(evStop[s][sub]));
}
void FlipFluidCUDA::resetTiming() {
    for (int i = 0; i < NUM_TIMING_STAGES; i++) accumMs[i] = 0;
    accumFrames = 0;
}
void FlipFluidCUDA::printTiming() {
    if (accumFrames == 0) return;
    printf("\n=== CUDA Timing (avg over %d frames) ===\n", accumFrames);
    printf("  grid: %dx%d (%d cells)  particles: %d\n",
           fNumX, fNumY, fNumCells, numParticles);
    printf("  numPressureIters=%d  numSubSteps=%d  effective pressure iters/frame=%d\n",
           lastNumPressureIters, lastNumSubSteps,
           lastNumPressureIters * lastNumSubSteps);
    for (int i = 0; i < NUM_TIMING_STAGES; i++)
        printf("  %-16s: %8.3f ms\n", timingStageNames[i], accumMs[i]/accumFrames);
    printf("=========================================\n\n");
}

// ── Simulation stages ─────────────────────────────────────────
void FlipFluidCUDA::integrateParticles(float dt, float gravity) {
    k_integrate<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                            d_particleVelX, d_particleVelY,
                                            numParticles, dt, gravity);
}

void FlipFluidCUDA::buildSpatialHash() {
    // Zero counts
    k_zeroInt<<<GRID(pNumCells),BLK>>>(d_numCellParticles, pNumCells);
    // Count
    k_hashCount<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                            numParticles, d_numCellParticles,
                                            pInvSpacing, pNumX, pNumY);
    // Inclusive prefix sum (parallel scan via Thrust/CUB) — replaces the old
    // single-thread kernel that serialized ~O(pNumCells) work each frame.
    // Matches the CPU semantics: firstCellParticle[i] = sum(counts[0..i]).
    thrust::device_ptr<int> cptr(d_numCellParticles);
    thrust::device_ptr<int> fptr(d_firstCellParticle);
    thrust::inclusive_scan(thrust::device, cptr, cptr + pNumCells, fptr);
    // Sentinel firstCellParticle[pNumCells] = total (= last inclusive sum).
    // Never decremented by the scatter, so it stays as the end offset.
    CUDA_CHECK(cudaMemcpy(d_firstCellParticle + pNumCells,
                          d_firstCellParticle + (pNumCells - 1),
                          sizeof(int), cudaMemcpyDeviceToDevice));
    // Scatter
    k_hashScatter<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                              numParticles, d_firstCellParticle,
                                              d_cellParticleIds,
                                              pInvSpacing, pNumX, pNumY);
}

void FlipFluidCUDA::pushParticlesApart(int numIters) {
    buildSpatialHash();
    float minDist = 2.0f * particleRadius;
    for (int iter = 0; iter < numIters; iter++) {
        k_separate<<<GRID(numParticles),BLK>>>(
            d_particlePosX, d_particlePosY,
            d_particleColorR, d_particleColorG, d_particleColorB,
            d_firstCellParticle, d_cellParticleIds,
            numParticles, pInvSpacing, pNumX, pNumY,
            minDist, 0.001f);
    }
}

void FlipFluidCUDA::handleParticleCollisions(float obsX, float obsY, float obsR,
                                             float obsVX, float obsVY) {
    k_collisions<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                             d_particleVelX, d_particleVelY,
                                             numParticles, obsX, obsY, obsR,
                                             obsVX, obsVY, particleRadius,
                                             h, fNumX, fNumY);
}

void FlipFluidCUDA::updateParticleDensity() {
    k_zeroFloat<<<GRID(fNumCells),BLK>>>(d_particleDensity, fNumCells);
    float h2 = 0.5f*h;
    k_particleDensity<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                                   numParticles, d_particleDensity,
                                                   h, fInvSpacing, h2, fNumX, fNumY);
}

float FlipFluidCUDA::computeRestDensity() {
    float zero = 0.0f; int zeroI = 0;
    CUDA_CHECK(cudaMemcpy(d_reductionBuf, &zero, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reductionIntBuf, &zeroI, sizeof(int), cudaMemcpyHostToDevice));
    k_sumFluidDensity<<<GRID(fNumCells),BLK>>>(d_particleDensity, d_cellType,
                                                fNumCells, d_reductionBuf, d_reductionIntBuf);
    float sumD; int cnt;
    CUDA_CHECK(cudaMemcpy(&sumD, d_reductionBuf, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cnt, d_reductionIntBuf, sizeof(int), cudaMemcpyDeviceToHost));
    return (cnt > 0) ? sumD/cnt : 0.0f;
}

void FlipFluidCUDA::classifyCells() {
    k_classifyInit<<<GRID(fNumCells),BLK>>>(d_cellType, d_s, fNumCells);
    k_classifyParticles<<<GRID(numParticles),BLK>>>(d_cellType, d_particlePosX, d_particlePosY,
                                                     numParticles, fInvSpacing, fNumX, fNumY);
}

void FlipFluidCUDA::savePrevVelocities() {
    k_savePrev<<<GRID(fNumCells),BLK>>>(d_u, d_v, d_du, d_dv, d_prevU, d_prevV, fNumCells);
}

void FlipFluidCUDA::p2gComponent(int comp) {
    float h2 = 0.5f*h;
    float* pvel = (comp==0) ? d_particleVelX : d_particleVelY;
    float* fld  = (comp==0) ? d_u : d_v;
    float* fldD = (comp==0) ? d_du : d_dv;
    k_p2g<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                       pvel, fld, fldD,
                                       numParticles, comp, h, fInvSpacing, h2, fNumX, fNumY);
}

void FlipFluidCUDA::p2gNormalize(int comp) {
    float* fld  = (comp==0) ? d_u : d_v;
    float* fldD = (comp==0) ? d_du : d_dv;
    k_p2gNorm<<<GRID(fNumCells),BLK>>>(fld, fldD, fNumCells);
}

void FlipFluidCUDA::restoreSolidCells() {
    k_restoreSolid<<<GRID(fNumCells),BLK>>>(d_u, d_v, d_prevU, d_prevV,
                                             d_cellType, fNumX, fNumY);
}

void FlipFluidCUDA::g2pComponent(int comp, float flipRatio) {
    float h2 = 0.5f*h;
    float* pvel = (comp==0) ? d_particleVelX : d_particleVelY;
    float* fld  = (comp==0) ? d_u : d_v;
    float* pfld = (comp==0) ? d_prevU : d_prevV;
    k_g2p<<<GRID(numParticles),BLK>>>(d_particlePosX, d_particlePosY,
                                       pvel, fld, pfld, d_cellType,
                                       numParticles, comp, flipRatio,
                                       h, fInvSpacing, h2, fNumX, fNumY);
}

void FlipFluidCUDA::transferVelocities(bool toGrid, float flipRatio) {
    if (toGrid) {
        savePrevVelocities();
        classifyCells();
        p2gComponent(0); p2gComponent(1);
        p2gNormalize(0); p2gNormalize(1);
        restoreSolidCells();
    } else {
        g2pComponent(0, flipRatio);
        g2pComponent(1, flipRatio);
    }
}

void FlipFluidCUDA::solveIncompressibility(int numIters, float dt,
                                           float overRelaxation, bool compensateDrift) {
    k_prepPressure<<<GRID(fNumCells),BLK>>>(d_p, d_prevU, d_prevV, d_u, d_v, fNumCells);
    float cp = density * h / dt;
    int cd = compensateDrift ? 1 : 0;
    int totalInner = (fNumX-2)*(fNumY-2);
    for (int iter = 0; iter < numIters; iter++) {
        // Red-Black: color 0 then color 1
        k_jacobiRB<<<GRID(totalInner),BLK>>>(d_u, d_v, d_p, d_s, d_cellType,
                                              d_particleDensity, particleRestDensity, cd,
                                              overRelaxation, cp, fNumX, fNumY, 0);
        k_jacobiRB<<<GRID(totalInner),BLK>>>(d_u, d_v, d_p, d_s, d_cellType,
                                              d_particleDensity, particleRestDensity, cd,
                                              overRelaxation, cp, fNumX, fNumY, 1);
    }
}

void FlipFluidCUDA::updateParticleColors() {
    k_updateParticleColors<<<GRID(numParticles),BLK>>>(
        d_particleColorR, d_particleColorG, d_particleColorB,
        d_particlePosX, d_particlePosY, d_particleDensity,
        particleRestDensity, fInvSpacing, numParticles, fNumX, fNumY);
}

void FlipFluidCUDA::updateCellColors() {
    k_updateCellColors<<<GRID(fNumCells),BLK>>>(d_cellColor, d_cellType,
                                                 d_particleDensity, particleRestDensity,
                                                 fNumCells);
}

void FlipFluidCUDA::carveObstacle(float x, float y, float r, float vx, float vy) {
    k_carveObstacle<<<GRID(fNumCells),BLK>>>(d_s, d_u, d_v, x, y, r, vx, vy, h, fNumX, fNumY);
}

void FlipFluidCUDA::simulate(float dt, float gravity, float flipRatio,
                             int numPressureIters, int numParticleIters,
                             float overRelaxation, bool compensateDrift,
                             bool separateParticles,
                             float obstacleX, float obstacleY, float obstacleRadius,
                             float obstacleVelX, float obstacleVelY,
                             int numSubSteps) {
    if (numSubSteps < 1) numSubSteps = 1;
    if (numSubSteps > MAX_SUBSTEPS) numSubSteps = MAX_SUBSTEPS;
    float sdt = dt / numSubSteps;

    lastNumPressureIters = numPressureIters;
    lastNumSubSteps = numSubSteps;

    for (int step = 0; step < numSubSteps; step++) {
        startTiming(T1_INTEGRATE, step);
        integrateParticles(sdt, gravity);
        stopTiming(T1_INTEGRATE, step);

        startTiming(T2_PUSH_APART, step);
        if (separateParticles) pushParticlesApart(numParticleIters);
        stopTiming(T2_PUSH_APART, step);

        startTiming(T3_COLLISIONS, step);
        handleParticleCollisions(obstacleX, obstacleY, obstacleRadius,
                                 obstacleVelX, obstacleVelY);
        stopTiming(T3_COLLISIONS, step);

        startTiming(T4_P2G, step);
        transferVelocities(true);
        stopTiming(T4_P2G, step);

        startTiming(T5_DENSITY, step);
        updateParticleDensity();
        if (particleRestDensity == 0.0f)
            particleRestDensity = computeRestDensity();
        stopTiming(T5_DENSITY, step);

        startTiming(T6_PRESSURE, step);
        solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
        stopTiming(T6_PRESSURE, step);

        startTiming(T7_G2P, step);
        transferVelocities(false, flipRatio);
        stopTiming(T7_G2P, step);
    }

    // Colors run once per frame (outside the substep loop) — store at slot 0.
    startTiming(T8_COLORS, 0);
    updateParticleColors();
    updateCellColors();
    stopTiming(T8_COLORS, 0);

    // Single sync point: once the colors stop-event completes, all the sim
    // kernels above (same default stream) are guaranteed finished.
    CUDA_CHECK(cudaEventRecord(evFrameStop));
    CUDA_CHECK(cudaEventSynchronize(evFrameStop));

    // Accumulate per-stage GPU time. For T1..T7 sum across every substep so
    // the reported number is the per-frame cost (not just the last substep).
    // T9_RENDER / T10_TRANSFER / T_TOTAL are measured by the caller (host
    // wall-clock) because they include OpenGL + D2H work outside this method.
    for (int i = 0; i < NUM_TIMING_STAGES; i++) {
        if (i == T9_RENDER || i == T10_TRANSFER || i == T_TOTAL) continue;
        int nsub = (i == T8_COLORS) ? 1 : numSubSteps;
        float stageMs = 0.0f;
        for (int s = 0; s < nsub; s++) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, evStart[i][s], evStop[i][s]));
            stageMs += ms;
        }
        accumMs[i] += stageMs;
    }
    accumFrames++;
}

void FlipFluidCUDA::packParticlesToBuffers(float2* dPos, float3* dCol) {
    k_packParticles<<<GRID(numParticles),BLK>>>(
        d_particlePosX, d_particlePosY,
        d_particleColorR, d_particleColorG, d_particleColorB,
        dPos, dCol, numParticles);
}

void FlipFluidCUDA::downloadForRender() {
    size_t szP = numParticles*sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_particlePosX, d_particlePosX, szP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particlePosY, d_particlePosY, szP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particleColorR, d_particleColorR, szP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particleColorG, d_particleColorG, szP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particleColorB, d_particleColorB, szP, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cellColor, d_cellColor, 3*fNumCells*sizeof(float), cudaMemcpyDeviceToHost));
}
