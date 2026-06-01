// CUDA FLIP Fluid Simulator
// GPU port of the CPU flip_fluid.cpp implementation
// All simulation stages run as CUDA kernels on the GPU.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ── Error checking macros ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Constants (same as CPU) ────────────────────────────────────────────────
constexpr int GPU_U_FIELD = 0;
constexpr int GPU_V_FIELD = 1;

constexpr int GPU_FLUID_CELL = 0;
constexpr int GPU_AIR_CELL   = 1;
constexpr int GPU_SOLID_CELL = 2;

// ── Timing stages ──────────────────────────────────────────────────────────
enum TimingStage {
    T1_INTEGRATE = 0,
    T2_PUSH_APART,
    T3_COLLISIONS,
    T4_P2G,
    T5_DENSITY,
    T6_PRESSURE,
    T7_G2P,
    T8_COLORS,
    T9_RENDER,
    T10_TRANSFER,   // D2H or interop map/unmap
    T_TOTAL,
    NUM_TIMING_STAGES
};

static const char* timingStageNames[] = {
    "T1_integrate",
    "T2_pushApart",
    "T3_collisions",
    "T4_p2g",
    "T5_density",
    "T6_pressure",
    "T7_g2p",
    "T8_colors",
    "T9_render",
    "T10_d2h",
    "T_total"
};

// ── FlipFluidCUDA class ────────────────────────────────────────────────────
class FlipFluidCUDA {
public:
    // Grid dimensions (host copies for launch config)
    float density;
    int   fNumX, fNumY;
    float h;
    float fInvSpacing;
    int   fNumCells;

    // Particle grid (spatial hash)
    float particleRadius;
    float pInvSpacing;
    int   pNumX, pNumY, pNumCells;
    int   maxParticles;
    int   numParticles;
    float particleRestDensity;

    // ── Device pointers: Grid arrays ──
    float *d_u, *d_v, *d_du, *d_dv, *d_prevU, *d_prevV;
    float *d_p, *d_s;
    int   *d_cellType;
    float *d_cellColor;        // 3 * fNumCells

    // ── Device pointers: Particle arrays ──
    float *d_particlePosX, *d_particlePosY;
    float *d_particleVelX, *d_particleVelY;
    float *d_particleColorR, *d_particleColorG, *d_particleColorB;
    float *d_particleDensity; // fNumCells

    // ── Device pointers: Spatial hash ──
    int *d_numCellParticles;
    int *d_firstCellParticle;  // pNumCells + 1
    int *d_cellParticleIds;    // maxParticles

    // ── Temporary buffers for prefix sum / reduction ──
    float *d_reductionBuf;     // for computeRestDensity
    int   *d_reductionIntBuf;

    // ── Host staging buffers (for D2H copy to render) ──
    float *h_particlePosX, *h_particlePosY;
    float *h_particleColorR, *h_particleColorG, *h_particleColorB;
    float *h_cellColor;

    // ── Timing ──
    cudaEvent_t evStart[NUM_TIMING_STAGES], evStop[NUM_TIMING_STAGES];
    cudaEvent_t evFrameStart, evFrameStop;
    float  accumMs[NUM_TIMING_STAGES];
    int    accumFrames;

    // ── Constructor / Destructor ──
    FlipFluidCUDA(float density, float width, float height,
                  float spacing, float particleRadius, int maxParticles);
    ~FlipFluidCUDA();

    // ── Upload initial data from host ──
    void uploadParticles(const float* posX, const float* posY,
                         const float* velX, const float* velY,
                         const float* colR, const float* colG, const float* colB,
                         int nParticles);
    void uploadGrid(const float* s_host);

    // ── Simulation stages (kernel launchers) ──
    void integrateParticles(float dt, float gravity);
    void pushParticlesApart(int numIters);
    void handleParticleCollisions(float obstacleX, float obstacleY,
                                  float obstacleRadius,
                                  float obstacleVelX, float obstacleVelY);
    void updateParticleDensity();
    void transferVelocities(bool toGrid, float flipRatio = 0.0f);
    void solveIncompressibility(int numIters, float dt,
                                float overRelaxation, bool compensateDrift);
    void updateParticleColors();
    void updateCellColors();

    // Full simulation step
    void simulate(float dt, float gravity, float flipRatio,
                  int numPressureIters, int numParticleIters,
                  float overRelaxation, bool compensateDrift,
                  bool separateParticles,
                  float obstacleX, float obstacleY, float obstacleRadius,
                  float obstacleVelX, float obstacleVelY,
                  int numSubSteps = 1);

    // Copy results back to host for rendering
    void downloadForRender();

    // ── Obstacle carving (runs on GPU) ──
    void carveObstacle(float x, float y, float r, float vx, float vy);

    // ── Timing helpers ──
    void startTiming(TimingStage stage);
    void stopTiming(TimingStage stage);
    void resetTiming();
    void printTiming();

private:
    // Internal kernel launchers
    void buildSpatialHash();
    void separateKernel(int numIters);
    void p2gComponent(int component);
    void p2gNormalize(int component);
    void g2pComponent(int component, float flipRatio);
    void classifyCells();
    void savePrevVelocities();
    void restoreSolidCells();
    float computeRestDensity();
};
