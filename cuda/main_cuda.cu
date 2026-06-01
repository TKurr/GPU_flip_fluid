#include "flip_fluid_cuda.cuh"
#include "../ui.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace flipcpu_ui;

struct Scene {
    float gravity         = -9.81f;
    float dt              = 1.0f / 60.0f;
    float flipRatio       = 0.9f;
    int   numPressureIters= 50;
    int   numParticleIters= 2;
    long  frameNr         = 0;
    float overRelaxation  = 1.9f;
    bool  compensateDrift = true;
    bool  separateParticles = true;
    float obstacleX       = 0.0f;
    float obstacleY       = 0.0f;
    float obstacleRadius  = 0.15f;
    bool  paused          = true;
    bool  showObstacle    = true;
    float obstacleVelX    = 0.0f;
    float obstacleVelY    = 0.0f;
    bool  showParticles   = true;
    bool  showGrid        = false;
    int   resolution      = 100;
    int   numSubSteps     = 1;
    FlipFluidCUDA* fluid  = nullptr;
};

static Scene scene;

constexpr int CANVAS_W = 900;
constexpr int CANVAS_H = 700;
constexpr float simHeight = 3.0f;
constexpr float cScale = float(CANVAS_H) / simHeight;
constexpr float simWidth = float(CANVAS_W) / cScale;

// ── CPU Setup Helpers (Staging before uploading to GPU) ───────────────
static void setObstacle(float x, float y, bool reset) {
    float vx = 0.0f, vy = 0.0f;
    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }
    scene.obstacleX = x;
    scene.obstacleY = y;
    
    if (scene.fluid) {
        scene.fluid->carveObstacle(x, y, scene.obstacleRadius, vx, vy);
    }
    
    scene.showObstacle  = true;
    scene.obstacleVelX  = vx;
    scene.obstacleVelY  = vy;
}

static void seedParticlesHost(std::vector<float>& px, std::vector<float>& py,
                              std::vector<float>& cr, std::vector<float>& cg, std::vector<float>& cb,
                              int numX, int numY, float h, float r, float dx, float dy) {
    px.assign(numX * numY, 0.0f);
    py.assign(numX * numY, 0.0f);
    cr.assign(numX * numY, 0.0f);
    cg.assign(numX * numY, 0.0f);
    cb.assign(numX * numY, 1.0f); // initial color blue

    for (int i = 0; i < numX; ++i) {
        for (int j = 0; j < numY; ++j) {
            int pid = i * numY + j;
            float offset = (j % 2 == 0) ? 0.0f : r;
            px[pid] = h + r + dx * i + offset;
            py[pid] = h + r + dy * j;
        }
    }
}

static void setupScene() {
    scene.obstacleRadius   = 0.15f;
    scene.overRelaxation   = 1.9f;
    scene.dt               = 1.0f / 60.0f;
    scene.numParticleIters = 2;

    int res = scene.resolution;
    if      (res <= 100) scene.numSubSteps = 1;
    else if (res <= 140) scene.numSubSteps = 2;
    else if (res <= 180) scene.numSubSteps = 3;
    else                 scene.numSubSteps = 4;
    scene.numPressureIters = 50 + std::max(0, (res - 100)) / 2;

    float tankHeight = 1.0f * simHeight;
    float tankWidth  = 1.0f * simWidth;
    float h          = tankHeight / res;
    float density    = 1000.0f;

    float relWaterHeight = 0.8f;
    float relWaterWidth  = 0.6f;

    float r  = 0.3f * h;
    float dx = 2.0f * r;
    float dy = std::sqrt(3.0f) / 2.0f * dx;

    int numX = int(std::floor((relWaterWidth  * tankWidth  - 2.0f * h - 2.0f * r) / dx));
    int numY = int(std::floor((relWaterHeight * tankHeight - 2.0f * h - 2.0f * r) / dy));
    if (numX < 1) numX = 1;
    if (numY < 1) numY = 1;
    int maxParticles = numX * numY;

    if (scene.fluid) delete scene.fluid;
    scene.fluid = new FlipFluidCUDA(density, tankWidth, tankHeight, h, r, maxParticles);

    // Initial grid state (borders are SOLID)
    std::vector<float> s_host(scene.fluid->fNumCells, 1.0f);
    int fnX = scene.fluid->fNumX, fnY = scene.fluid->fNumY;
    for (int i = 0; i < fnX; ++i) {
        for (int j = 0; j < fnY; ++j) {
            if (i == 0 || i == fnX - 1 || j == 0) 
                s_host[i * fnY + j] = 0.0f;
        }
    }
    scene.fluid->uploadGrid(s_host.data());

    // Initial particles state
    std::vector<float> px, py, cr, cg, cb;
    seedParticlesHost(px, py, cr, cg, cb, numX, numY, h, r, dx, dy);
    std::vector<float> vx(maxParticles, 0.0f), vy(maxParticles, 0.0f);
    
    scene.fluid->uploadParticles(px.data(), py.data(), vx.data(), vy.data(),
                                 cr.data(), cg.data(), cb.data(), maxParticles);

    setObstacle(3.0f, 2.0f, true);
    scene.frameNr = 0;
    
    scene.fluid->resetTiming();
}

// ── Rendering ──────────────────────────────────────────────────
static void setProjection(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, simWidth, 0.0, simHeight, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void drawGrid(FlipFluidCUDA* f) {
    float h = f->h;
    glBegin(GL_QUADS);
    for (int i = 0; i < f->fNumX; ++i) {
        for (int j = 0; j < f->fNumY; ++j) {
            int idx = i * f->fNumY + j;
            float r = f->h_cellColor[3 * idx + 0];
            float g = f->h_cellColor[3 * idx + 1];
            float b = f->h_cellColor[3 * idx + 2];
            float x0 = i * h, y0 = j * h;
            float x1 = x0 + h, y1 = y0 + h;
            glColor3f(r, g, b);
            glVertex2f(x0, y0); glVertex2f(x1, y0);
            glVertex2f(x1, y1); glVertex2f(x0, y1);
        }
    }
    glEnd();
}

static void drawParticles(FlipFluidCUDA* f, int viewportH) {
    float pxPerSimUnit = float(viewportH) / simHeight;
    float diameterPx = 2.0f * f->particleRadius * pxPerSimUnit;
    if (diameterPx < 1.0f) diameterPx = 1.0f;
    glPointSize(diameterPx);
    glBegin(GL_POINTS);
    for (int i = 0; i < f->numParticles; ++i) {
        glColor3f(f->h_particleColorR[i], f->h_particleColorG[i], f->h_particleColorB[i]);
        glVertex2f(f->h_particlePosX[i], f->h_particlePosY[i]);
    }
    glEnd();
}

static void drawObstacle(float ox, float oy, float orad, float particleRadius) {
    const int N = 48;
    float drawR = orad + particleRadius;
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(ox, oy);
    for (int i = 0; i <= N; ++i) {
        float a = (float)i / N * 2.0f * (float)M_PI;
        glVertex2f(ox + drawR * std::cos(a), oy + drawR * std::sin(a));
    }
    glEnd();
}

// ── X11 Window Setup ───────────────────────────────────────────
static int s_glxAttrs[] = {
    GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24,
    GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None
};

struct AppWindow {
    Display* dpy = nullptr;
    ::Window xwin = 0;
    GLXContext glc = nullptr;
    XVisualInfo* vi = nullptr;
    Atom wm_delete = 0;
    int width = CANVAS_W, height = CANVAS_H;
    bool running = true;
};

static bool createWindow(AppWindow& w, const char* title) {
    w.dpy = XOpenDisplay(nullptr);
    if (!w.dpy) return false;
    int screen = DefaultScreen(w.dpy);
    w.vi = glXChooseVisual(w.dpy, screen, s_glxAttrs);
    if (!w.vi) return false;
    ::Window root = RootWindow(w.dpy, screen);
    Colormap cmap = XCreateColormap(w.dpy, root, w.vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask |
                     ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask;
    w.xwin = XCreateWindow(w.dpy, root, 0, 0, w.width, w.height, 0,
                           w.vi->depth, InputOutput, w.vi->visual,
                           CWColormap | CWEventMask, &swa);
    XStoreName(w.dpy, w.xwin, title);
    w.wm_delete = XInternAtom(w.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(w.dpy, w.xwin, &w.wm_delete, 1);
    XMapWindow(w.dpy, w.xwin);

    w.glc = glXCreateContext(w.dpy, w.vi, nullptr, GL_TRUE);
    glXMakeCurrent(w.dpy, w.xwin, w.glc);

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    return true;
}

// ── Main Loop ──────────────────────────────────────────────────
int main(int argc, char** argv) {
    bool noVsync = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vsync") == 0) noVsync = true;
    }

    std::printf("[flip-cuda] starting (GPU sim, GPU render)\n");
    setupScene();
    scene.paused = true;

    AppWindow w;
    if (!createWindow(w, "FLIP Fluid (CUDA GPU sim)")) return 1;

    if (noVsync) {
        typedef int (*PFNGLXSWAPINTERVAL)(int);
        auto pfn = (PFNGLXSWAPINTERVAL)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalMESA");
        if (pfn) pfn(0);
    }

    bool mouseDownPrev = false, mouseDown = false, mousePressedEdge = false, mouseReleasedEdge = false;
    float mouseSimX = 0.0f, mouseSimY = 0.0f;
    int mousePxX = 0, mousePxY = 0;
    bool dragOwnedByUI = false;

    auto fpsT0 = std::chrono::steady_clock::now();
    int fpsFrames = 0;
    double lastFps = 0.0;
    bool gravityOn = true;

    while (w.running) {
        mousePressedEdge = false; mouseReleasedEdge = false;

        while (XPending(w.dpy) > 0) {
            XEvent e; XNextEvent(w.dpy, &e);
            if (e.type == ClientMessage) {
                if ((Atom)e.xclient.data.l[0] == w.wm_delete) w.running = false;
            } else if (e.type == ConfigureNotify) {
                w.width = e.xconfigure.width; w.height = e.xconfigure.height;
            } else if (e.type == KeyPress) {
                KeySym ks = XLookupKeysym(&e.xkey, 0);
                if (ks == XK_space || ks == XK_p || ks == XK_P) scene.paused = !scene.paused;
                else if (ks == XK_g || ks == XK_G) scene.showGrid = !scene.showGrid;
                else if (ks == XK_r || ks == XK_R) setupScene();
                else if (ks == XK_q || ks == XK_Q || ks == XK_Escape) w.running = false;
                else if (ks == XK_t || ks == XK_T) scene.fluid->printTiming(); // Press 'T' to print timings
            } else if (e.type == ButtonPress && e.xbutton.button == Button1) {
                mouseDown = true; mousePressedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
                mouseSimX = float(mousePxX) / w.width * simWidth;
                mouseSimY = (1.0f - float(mousePxY) / w.height) * simHeight;
            } else if (e.type == ButtonRelease && e.xbutton.button == Button1) {
                mouseDown = false; mouseReleasedEdge = true;
                mousePxX = e.xbutton.x; mousePxY = e.xbutton.y;
                mouseSimX = float(mousePxX) / w.width * simWidth;
                mouseSimY = (1.0f - float(mousePxY) / w.height) * simHeight;
            } else if (e.type == MotionNotify) {
                mousePxX = e.xmotion.x; mousePxY = e.xmotion.y;
                mouseSimX = float(mousePxX) / w.width * simWidth;
                mouseSimY = (1.0f - float(mousePxY) / w.height) * simHeight;
            }
        }

        const int kPanelX = 10, kPanelY = 10, kPanelW = 160, kPanelH = 260;
        bool mouseOnPanel = (mousePxX >= kPanelX && mousePxX < kPanelX + kPanelW &&
                             mousePxY >= kPanelY && mousePxY < kPanelY + kPanelH);
        if (mousePressedEdge && mouseOnPanel) dragOwnedByUI = true;
        if (!mouseDown) dragOwnedByUI = false;

        if (mouseDown && !dragOwnedByUI) {
            if (!mouseDownPrev) { setObstacle(mouseSimX, mouseSimY, true); scene.paused = false; }
            else { setObstacle(mouseSimX, mouseSimY, false); }
            mouseDownPrev = true;
        } else {
            if (mouseDownPrev) { scene.obstacleVelX = 0; scene.obstacleVelY = 0; }
            mouseDownPrev = false;
        }

        FlipFluidCUDA* f = scene.fluid;

        if (!scene.paused) {
            f->simulate(scene.dt, scene.gravity, scene.flipRatio, scene.numPressureIters,
                        scene.numParticleIters, scene.overRelaxation, scene.compensateDrift,
                        scene.separateParticles, scene.obstacleX, scene.obstacleY,
                        scene.obstacleRadius, scene.obstacleVelX, scene.obstacleVelY,
                        scene.numSubSteps);
            scene.frameNr++;
        }

        // T10: Memory transfer for rendering (D2H)
        f->startTiming(T10_TRANSFER);
        f->downloadForRender();
        f->stopTiming(T10_TRANSFER);
        
        // Wait for D2H so we can measure it
        cudaEventSynchronize(f->evStop[T10_TRANSFER]);
        float transferMs = 0;
        cudaEventElapsedTime(&transferMs, f->evStart[T10_TRANSFER], f->evStop[T10_TRANSFER]);
        f->accumMs[T10_TRANSFER] += transferMs;

        // T9: Rendering
        f->startTiming(T9_RENDER);
        
        glClear(GL_COLOR_BUFFER_BIT);
        setProjection(w.width, w.height);

        if (scene.showGrid) drawGrid(f);
        if (scene.showParticles) drawParticles(f, w.height);
        if (scene.showObstacle) drawObstacle(scene.obstacleX, scene.obstacleY, scene.obstacleRadius, f->particleRadius);

        flipcpu_ui::setProjectionToPixels(w.width, w.height);
        flipcpu_ui::Input uin;
        uin.screenW = w.width; uin.screenH = w.height;
        uin.mouseX = mousePxX; uin.mouseY = mousePxY;
        uin.mouseDown = mouseDown;
        uin.mousePressed = mousePressedEdge && mouseOnPanel;
        uin.mouseReleased = mouseReleasedEdge;
        
        flipcpu_ui::begin(uin);
        flipcpu_ui::beginPanel(kPanelX, kPanelY, kPanelW, kPanelH, "CUDA Controls");
        flipcpu_ui::text("FPS: %.1f", lastFps);
        flipcpu_ui::text("Particles: %d", f->numParticles);
        flipcpu_ui::text("Frame: %ld", scene.frameNr);
        flipcpu_ui::checkbox("Particles", &scene.showParticles);
        flipcpu_ui::checkbox("Grid", &scene.showGrid);
        flipcpu_ui::checkbox("Compensate Drift", &scene.compensateDrift);
        flipcpu_ui::checkbox("Separate Particles", &scene.separateParticles);
        if (flipcpu_ui::checkbox("Gravity", &gravityOn)) scene.gravity = gravityOn ? -9.81f : 0.0f;
        flipcpu_ui::slider("PIC <-> FLIP", &scene.flipRatio, 0.0f, 1.0f);
        float resFloat = (float)scene.resolution;
        flipcpu_ui::slider("Grid Res", &resFloat, 30.0f, 200.0f);
        int newRes = (int)(resFloat + 0.5f);
        if (newRes != scene.resolution) { scene.resolution = newRes; setupScene(); }
        flipcpu_ui::checkbox("Pause", &scene.paused);
        if (flipcpu_ui::button("Reset")) setupScene();
        if (flipcpu_ui::button("Print Timing")) { f->printTiming(); }
        flipcpu_ui::endPanel();
        flipcpu_ui::restoreProjection();

        glXSwapBuffers(w.dpy, w.xwin);
        
        f->stopTiming(T9_RENDER);
        cudaEventSynchronize(f->evStop[T9_RENDER]);
        float renderMs = 0;
        cudaEventElapsedTime(&renderMs, f->evStart[T9_RENDER], f->evStop[T9_RENDER]);
        f->accumMs[T9_RENDER] += renderMs;

        fpsFrames++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fpsT0).count();
        if (elapsed >= 0.5) {
            lastFps = fpsFrames / elapsed;
            fpsT0 = now; fpsFrames = 0;
            // Also print average timing every 0.5s if not paused
            if (!scene.paused && scene.frameNr % 60 == 0) {
                 f->printTiming();
            }
        }
    }

    if (w.glc) { glXMakeCurrent(w.dpy, None, nullptr); glXDestroyContext(w.dpy, w.glc); }
    if (w.xwin) XDestroyWindow(w.dpy, w.xwin);
    if (w.dpy) XCloseDisplay(w.dpy);
    
    delete scene.fluid;
    return 0;
}
