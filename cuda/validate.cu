// Numerical validation: CPU (reference) vs CUDA port.
//
// Runs both simulators from an IDENTICAL initial state for N frames with the
// same parameters, then reports max-abs and RMS (L2) error per field. Errors
// are expected to be small but nonzero — the orderings differ on purpose
// (sequential Gauss-Seidel vs red-black; sequential scatter vs atomic). The
// point is to show the deviation stays bounded / small relative to the field.
//
// Headless (no OpenGL). Build: `make flip_validate`. Run: ./flip_validate [opts]

#include "../flip_fluid.h"          // CPU reference (namespace flipcpu)
#include "flip_fluid_cuda.cuh"      // CUDA port

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Scene constants mirror main.cpp / main_cuda.cu exactly.
static const float simHeight = 3.0f;
static const float simWidth  = 900.0f / (700.0f / simHeight);   // ~3.857

struct SetupResult {
    int numX, numY, maxParticles, numSubSteps, numPressureIters;
    float h, density, r, dx, dy, tankWidth, tankHeight;
};

static SetupResult computeSetup(int res) {
    SetupResult s;
    s.tankHeight = simHeight;
    s.tankWidth  = simWidth;
    s.h = s.tankHeight / res;
    s.density = 1000.0f;
    float relWaterHeight = 0.8f, relWaterWidth = 0.6f;
    s.r  = 0.3f * s.h;
    s.dx = 2.0f * s.r;
    s.dy = std::sqrt(3.0f) / 2.0f * s.dx;
    s.numX = int(std::floor((relWaterWidth  * s.tankWidth  - 2.0f * s.h - 2.0f * s.r) / s.dx));
    s.numY = int(std::floor((relWaterHeight * s.tankHeight - 2.0f * s.h - 2.0f * s.r) / s.dy));
    if (s.numX < 1) s.numX = 1;
    if (s.numY < 1) s.numY = 1;
    s.maxParticles = s.numX * s.numY;
    if      (res <= 100) s.numSubSteps = 1;
    else if (res <= 140) s.numSubSteps = 2;
    else if (res <= 180) s.numSubSteps = 3;
    else                 s.numSubSteps = 4;
    s.numPressureIters = 50 + std::max(0, (res - 100)) / 2;
    return s;
}

// Seed particle positions identically to the apps.
static void seed(std::vector<float>& px, std::vector<float>& py,
                 const SetupResult& s) {
    px.assign(s.maxParticles, 0.0f);
    py.assign(s.maxParticles, 0.0f);
    for (int i = 0; i < s.numX; ++i)
        for (int j = 0; j < s.numY; ++j) {
            int pid = i * s.numY + j;
            float off = (j % 2 == 0) ? 0.0f : s.r;
            px[pid] = s.h + s.r + s.dx * i + off;
            py[pid] = s.h + s.r + s.dy * j;
        }
}

// Border-solid tank `s` grid, matching setupTank().
static void buildTank(std::vector<float>& sgrid, int fNumX, int fNumY) {
    sgrid.assign(fNumX * fNumY, 1.0f);
    for (int i = 0; i < fNumX; ++i)
        for (int j = 0; j < fNumY; ++j)
            if (i == 0 || i == fNumX - 1 || j == 0)
                sgrid[i * fNumY + j] = 0.0f;
}

// Carve a static obstacle into CPU arrays (mirrors carveObstacle in main.cpp).
static void carveCPU(flipcpu::FlipFluid& f, float ox, float oy, float r) {
    int n = f.fNumY;
    for (int i = 1; i < f.fNumX - 2; ++i)
        for (int j = 1; j < f.fNumY - 2; ++j) {
            f.s[i * n + j] = 1.0f;
            float dx = (i + 0.5f) * f.h - ox;
            float dy = (j + 0.5f) * f.h - oy;
            if (dx * dx + dy * dy < r * r) {
                f.s[i * n + j] = 0.0f;
                f.u[i * n + j] = 0.0f; f.u[(i + 1) * n + j] = 0.0f;
                f.v[i * n + j] = 0.0f; f.v[i * n + j + 1] = 0.0f;
            }
        }
}

struct ErrStat { double maxAbs, rms, fieldMax; };

static ErrStat compare(const float* a, const float* b, int n) {
    double se = 0.0, me = 0.0, fmax = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = double(a[i]) - double(b[i]);
        double ad = std::fabs(d);
        se += d * d;
        if (ad > me) me = ad;
        if (std::fabs((double)a[i]) > fmax) fmax = std::fabs((double)a[i]);
    }
    ErrStat e;
    e.maxAbs = me;
    e.rms = (n > 0) ? std::sqrt(se / n) : 0.0;
    e.fieldMax = fmax;
    return e;
}

static void printRow(const char* name, const ErrStat& e) {
    double rel = (e.fieldMax > 1e-12) ? e.maxAbs / e.fieldMax : 0.0;
    std::printf("  %-12s  max|err|=%.3e   rms=%.3e   (CPU max|val|=%.3e, rel=%.2e)\n",
                name, e.maxAbs, e.rms, e.fieldMax, rel);
}

int main(int argc, char** argv) {
    int res = 100, frames = 5;
    float gravity = -9.81f, flipRatio = 0.9f, dt = 1.0f / 60.0f;
    float overRelax = 1.9f;
    int numParticleIters = 2;
    bool compensateDrift = true, separate = true;
    float obsX = 3.0f, obsY = 2.0f, obsR = 0.15f;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--res") && i + 1 < argc) res = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::printf("Usage: %s [--res N] [--frames N]\n", argv[0]);
            return 0;
        }
    }

    SetupResult s = computeSetup(res);
    std::vector<float> px, py;
    seed(px, py, s);
    std::vector<float> vx(s.maxParticles, 0.0f), vy(s.maxParticles, 0.0f);
    std::vector<float> cr(s.maxParticles, 0.0f), cg(s.maxParticles, 0.0f),
                       cb(s.maxParticles, 1.0f);

    // ── CPU sim ──
    flipcpu::FlipFluid cpu(s.density, s.tankWidth, s.tankHeight, s.h, s.r, s.maxParticles);
    cpu.numParticles = s.maxParticles;
    cpu.particleRestDensity = 0.0f;
    for (int i = 0; i < s.maxParticles; ++i) {
        cpu.particlePosX[i] = px[i]; cpu.particlePosY[i] = py[i];
        cpu.particleVelX[i] = 0.0f;  cpu.particleVelY[i] = 0.0f;
        cpu.particleColorR[i] = cr[i]; cpu.particleColorG[i] = cg[i]; cpu.particleColorB[i] = cb[i];
    }
    {
        std::vector<float> sgrid;
        buildTank(sgrid, cpu.fNumX, cpu.fNumY);
        cpu.s = sgrid;
    }
    carveCPU(cpu, obsX, obsY, obsR);

    // ── CUDA sim (identical init) ──
    FlipFluidCUDA gpu(s.density, s.tankWidth, s.tankHeight, s.h, s.r, s.maxParticles);
    {
        std::vector<float> sgrid;
        buildTank(sgrid, gpu.fNumX, gpu.fNumY);
        gpu.uploadGrid(sgrid.data());
    }
    gpu.uploadParticles(px.data(), py.data(), vx.data(), vy.data(),
                        cr.data(), cg.data(), cb.data(), s.maxParticles);
    gpu.carveObstacle(obsX, obsY, obsR, 0.0f, 0.0f);

    std::printf("=== Numerical validation CPU vs CUDA ===\n");
    std::printf("  res=%d  grid=%dx%d  particles=%d  frames=%d  numSubSteps=%d  numPressureIters=%d\n",
                res, cpu.fNumX, cpu.fNumY, s.maxParticles, frames, s.numSubSteps, s.numPressureIters);

    for (int fr = 0; fr < frames; ++fr) {
        cpu.simulate(dt, gravity, flipRatio, s.numPressureIters, numParticleIters,
                     overRelax, compensateDrift, separate,
                     obsX, obsY, obsR, 0.0f, 0.0f, s.numSubSteps);
        gpu.simulate(dt, gravity, flipRatio, s.numPressureIters, numParticleIters,
                     overRelax, compensateDrift, separate,
                     obsX, obsY, obsR, 0.0f, 0.0f, s.numSubSteps);
    }

    // ── Pull CUDA state back and compare ──
    int nP = s.maxParticles, nC = cpu.fNumCells;
    std::vector<float> gpx(nP), gpy(nP), gvx(nP), gvy(nP), gu(nC), gv(nC);
    CUDA_CHECK(cudaMemcpy(gpx.data(), gpu.d_particlePosX, nP * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpy.data(), gpu.d_particlePosY, nP * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gvx.data(), gpu.d_particleVelX, nP * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gvy.data(), gpu.d_particleVelY, nP * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gu.data(),  gpu.d_u, nC * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gv.data(),  gpu.d_v, nC * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("after %d frame(s):\n", frames);
    printRow("particlePosX", compare(cpu.particlePosX.data(), gpx.data(), nP));
    printRow("particlePosY", compare(cpu.particlePosY.data(), gpy.data(), nP));
    printRow("particleVelX", compare(cpu.particleVelX.data(), gvx.data(), nP));
    printRow("particleVelY", compare(cpu.particleVelY.data(), gvy.data(), nP));
    printRow("grid u",       compare(cpu.u.data(), gu.data(), nC));
    printRow("grid v",       compare(cpu.v.data(), gv.data(), nC));
    std::printf("========================================\n");
    std::printf("Note: nonzero error is expected (GS vs red-black, atomic ordering).\n");
    std::printf("Small RMS relative to the field magnitude => the port is correct.\n");
    return 0;
}
