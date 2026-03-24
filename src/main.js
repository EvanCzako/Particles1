import computeSrc from './compute.wgsl?raw';
import renderSrc  from './render.wgsl?raw';

// ── Simulation parameters ────────────────────────────────────────────────────
const MAX_PARTICLES    = 10000;  // buffer always allocated for this many
const DEFAULT_PARTICLES = 5000;
const RMIN             = 20;
const RMAX             = 300;
const REP_STRENGTH     = 3.0;
const ATTRACT_STRENGTH = 0.01;
const FRICTION         = 0.9;

// 8 colors — must match typeColor() in render.wgsl (same order).
const TYPE_COLORS = [
    '#ff3838', // 0 red
    '#33ff4d', // 1 green
    '#338cff', // 2 blue
    '#ffe619', // 3 yellow
    '#ff6600', // 4 orange
    '#cc33ff', // 5 purple
    '#00e5e5', // 6 cyan
    '#ff99cc', // 7 pink
];

const MAX_TYPES       = 8;    // hard maximum; force matrix buffer uses this as stride
const PARTICLE_RADIUS = 3.5;  // base visual radius in world units (scaled by worldScale())

// World scale: keeps the longest screen dimension equivalent to ~3840 world units
// (2× a 1920px reference). On smaller screens (mobile) the scale grows, so the
// simulation world never becomes disproportionately tiny.
function worldScale() {
    return Math.max(2, 3840 / Math.max(window.innerWidth, window.innerHeight));
}
// ─────────────────────────────────────────────────────────────────────────────

const STEP_DT             = 1 / 60;
const MAX_STEPS           = 8;
const FLOATS_PER_PARTICLE = 6;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Uniform buffer: 48 bytes, 16-byte aligned (WGSL uniform address-space requirement).
// Layout must match struct Uniforms in both WGSL shaders exactly.
//   0: width  4: height  8: numParticles  12: numTypes
//  16: rMin  20: rMax   24: repStrength  28: attractStrength
//  32: friction  36: particleRadius  40–44: padding
// width/height are the WORLD dimensions (canvas * worldScale()), not the pixel dims.
function writeUniforms(device, buf, width, height, numTypes, numParticles) {
    const ab = new ArrayBuffer(48);
    const dv = new DataView(ab);
    dv.setFloat32( 0, width  * worldScale(),          true);
    dv.setFloat32( 4, height * worldScale(),          true);
    dv.setUint32 ( 8, numParticles,                  true);
    dv.setUint32 (12, numTypes,                      true);
    dv.setFloat32(16, RMIN,                          true);
    dv.setFloat32(20, RMAX,                          true);
    dv.setFloat32(24, REP_STRENGTH,                  true);
    dv.setFloat32(28, ATTRACT_STRENGTH,              true);
    dv.setFloat32(32, FRICTION,                      true);
    dv.setFloat32(36, PARTICLE_RADIUS * worldScale(), true);
    // bytes 40–47: padding (ArrayBuffer is zero-initialized)
    device.queue.writeBuffer(buf, 0, ab);
}

// Interpolates near-black (v≈0) → green (v=+1, attract) or red (v=−1, repel).
function valueToColor(v) {
    const t   = Math.abs(v);
    const base = [14, 14, 18];
    const pos  = [28, 200, 70];
    const neg  = [215, 32,  32];
    const c    = v >= 0 ? pos : neg;
    return `rgb(${Math.round(base[0] + t*(c[0]-base[0]))},${Math.round(base[1] + t*(c[1]-base[1]))},${Math.round(base[2] + t*(c[2]-base[2]))})`;
}

// Builds the NxN matrix grid inside #matrix-grid.
// forceMatrix uses a fixed stride of MAX_TYPES (8) so the GPU buffer layout
// never changes when numTypes changes.
// Returns { refresh() } to repaint all cells from current forceMatrix values.
function createMatrixUI({ numTypes, typeColors, forceMatrix, onAnyChange }) {
    const grid = document.getElementById('matrix-grid');
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `18px ${'30px '.repeat(numTypes)}`.trim();

    const setCellFns = new Array(MAX_TYPES * MAX_TYPES);

    for (let row = 0; row <= numTypes; row++) {
        for (let col = 0; col <= numTypes; col++) {
            const el = document.createElement('div');

            if (row === 0 && col === 0) {
                el.style.cssText = 'width:18px;height:18px';

            } else if (row === 0) {
                el.className = 'type-dot';
                el.style.background = typeColors[col - 1];
                el.title = `target type ${col - 1}`;

            } else if (col === 0) {
                el.className = 'type-dot';
                el.style.background = typeColors[row - 1];
                el.title = `source type ${row - 1}`;

            } else {
                const i   = row - 1;
                const j   = col - 1;
                const idx = i * MAX_TYPES + j;  // fixed stride = 8

                el.className = 'matrix-cell';

                const paint = (v) => {
                    forceMatrix[idx]    = v;
                    el.style.background = valueToColor(v);
                    el.title            = (v >= 0 ? '+' : '') + v.toFixed(2);
                };

                setCellFns[idx] = paint;
                paint(forceMatrix[idx]);

                // ── Pointer interaction ──────────────────────────────────────
                // Drag up/down adjusts value. A release with no (or minimal)
                // movement is treated as a click → set value to zero.
                let dragging = false, moved = false, startY = 0, startVal = 0;

                el.addEventListener('pointerdown', (e) => {
                    dragging = true;
                    moved    = false;
                    startY   = e.clientY;
                    startVal = forceMatrix[idx];
                    el.setPointerCapture(e.pointerId);
                    e.preventDefault();
                });

                el.addEventListener('pointermove', (e) => {
                    if (!dragging) return;
                    if (Math.abs(e.clientY - startY) > 3) moved = true;
                    if (!moved) return;
                    const v = Math.max(-1, Math.min(1, startVal + (startY - e.clientY) * 0.01));
                    paint(v);
                    onAnyChange();
                });

                el.addEventListener('pointerup', () => {
                    if (dragging && !moved) {   // click with no drag → zero
                        paint(0);
                        onAnyChange();
                    }
                    dragging = false;
                    moved    = false;
                });

                el.addEventListener('pointercancel', () => {
                    dragging = false;
                    moved    = false;
                });
            }

            grid.appendChild(el);
        }
    }

    return {
        refresh() {
            for (let i = 0; i < numTypes; i++) {
                for (let j = 0; j < numTypes; j++) {
                    setCellFns[i * MAX_TYPES + j]?.(forceMatrix[i * MAX_TYPES + j]);
                }
            }
        },
    };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
    if (!navigator.gpu) {
        showError('WebGPU not supported. Use Chrome 113+ or Edge 113+.');
        return;
    }

    const canvas = document.getElementById('canvas');
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { showError('No WebGPU adapter found.'); return; }
    const device  = await adapter.requestDevice();
    const context = canvas.getContext('webgpu');
    const format  = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    // ── Force matrix ──────────────────────────────────────────────────────────
    // Always MAX_TYPES × MAX_TYPES (8×8 = 64 floats). The shader reads with
    // fixed stride MAX_TYPES, so this buffer never needs to be resized.
    // forceMatrix[i * MAX_TYPES + j] = force type-i feels toward type-j.
    let numTypes     = 2;
    let numParticles = DEFAULT_PARTICLES;
    const forceMatrix = new Float32Array(MAX_TYPES * MAX_TYPES);
    for (let i = 0; i < numTypes; i++)
        for (let j = 0; j < numTypes; j++)
            forceMatrix[i * MAX_TYPES + j] = Math.random() * 2 - 1;

    // ── GPU buffers ───────────────────────────────────────────────────────────
    const bufSize = MAX_PARTICLES * FLOATS_PER_PARTICLE * 4;
    const bufA = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufB = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE });

    const uniformBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // fmBuf is always MAX_TYPES² floats — never reallocated.
    const fmBuf = device.createBuffer({
        size:  forceMatrix.byteLength,   // 256 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── Pipelines ─────────────────────────────────────────────────────────────
    // numTypes is now a uniform — no pipeline-level constant needed.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: computeSrc }), entryPoint: 'main' },
    });

    const renderModule = device.createShaderModule({ code: renderSrc });
    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex:   { module: renderModule, entryPoint: 'vs_main' },
        fragment: {
            module: renderModule,
            entryPoint: 'fs_main',
            targets: [{
                format,
                blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one',                operation: 'add' },
                    alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
                },
            }],
        },
        primitive: { topology: 'triangle-list' },
    });

    // ── Bind groups (ping-pong) ───────────────────────────────────────────────
    const cLayout = computePipeline.getBindGroupLayout(0);
    const rLayout = renderPipeline.getBindGroupLayout(0);

    const mkComputeBG = (src, dst) => device.createBindGroup({
        layout: cLayout,
        entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: dst } },
            { binding: 2, resource: { buffer: uniformBuf } },
            { binding: 3, resource: { buffer: fmBuf } },
        ],
    });
    const mkRenderBG = (buf) => device.createBindGroup({
        layout: rLayout,
        entries: [
            { binding: 0, resource: { buffer: buf } },
            { binding: 1, resource: { buffer: uniformBuf } },
        ],
    });

    const computeBG = [mkComputeBG(bufA, bufB), mkComputeBG(bufB, bufA)];
    const renderBG  = [mkRenderBG(bufA),        mkRenderBG(bufB)];

    // ── Simulation state ──────────────────────────────────────────────────────
    let physicsStep = 0;
    let accumulator = 0;

    function initParticles() {
        const data = new Float32Array(numParticles * FLOATS_PER_PARTICLE);
        for (let i = 0; i < numParticles; i++) {
            const b = i * FLOATS_PER_PARTICLE;
            data[b + 0] = Math.random() * canvas.width  * worldScale();
            data[b + 1] = Math.random() * canvas.height * worldScale();
            data[b + 4] = Math.floor(Math.random() * numTypes);
        }
        device.queue.writeBuffer(bufA, 0, data);
        physicsStep = 0;
        accumulator = 0;
    }

    // ── Matrix UI ─────────────────────────────────────────────────────────────
    let matrixUI = createMatrixUI({
        numTypes, typeColors: TYPE_COLORS, forceMatrix,
        onAnyChange: () => device.queue.writeBuffer(fmBuf, 0, forceMatrix),
    });

    function applyNumTypes(newNumTypes) {
        // Expand matrix with random values for any newly added type rows/cols.
        if (newNumTypes > numTypes) {
            for (let i = 0; i < newNumTypes; i++) {
                for (let j = 0; j < newNumTypes; j++) {
                    if (i >= numTypes || j >= numTypes) {
                        forceMatrix[i * MAX_TYPES + j] = Math.random() * 2 - 1;
                    }
                }
            }
        }

        numTypes = newNumTypes;
        device.queue.writeBuffer(fmBuf, 0, forceMatrix);
        writeUniforms(device, uniformBuf, canvas.width, canvas.height, numTypes, numParticles);

        initParticles();

        // Rebuild grid
        matrixUI = createMatrixUI({
            numTypes, typeColors: TYPE_COLORS, forceMatrix,
            onAnyChange: () => device.queue.writeBuffer(fmBuf, 0, forceMatrix),
        });

        const countEl = document.getElementById('types-count');
        if (countEl) countEl.textContent = numTypes;
        document.getElementById('types-minus').disabled = numTypes <= 1;
        document.getElementById('types-plus').disabled  = numTypes >= MAX_TYPES;
    }

    document.getElementById('types-minus').addEventListener('click', () => {
        if (numTypes > 1) applyNumTypes(numTypes - 1);
    });
    document.getElementById('types-plus').addEventListener('click', () => {
        if (numTypes < MAX_TYPES) applyNumTypes(numTypes + 1);
    });

    document.getElementById('zero-btn').addEventListener('click', () => {
        forceMatrix.fill(0);
        device.queue.writeBuffer(fmBuf, 0, forceMatrix);
        matrixUI.refresh();
    });

    document.getElementById('randomize-btn').addEventListener('click', () => {
        for (let i = 0; i < numTypes; i++)
            for (let j = 0; j < numTypes; j++)
                forceMatrix[i * MAX_TYPES + j] = Math.random() * 2 - 1;
        device.queue.writeBuffer(fmBuf, 0, forceMatrix);
        matrixUI.refresh();
    });

    // ── First frame: write initial GPU state ──────────────────────────────────
    initParticles();
    writeUniforms(device, uniformBuf, canvas.width, canvas.height, numTypes, numParticles);
    device.queue.writeBuffer(fmBuf, 0, forceMatrix);

    // ── Time control ──────────────────────────────────────────────────────────
    let timeMultiplier = 1.0;
    let lastTime       = null;

    const slider = document.getElementById('speed-slider');
    const label  = document.getElementById('speed-label');
    slider.addEventListener('input', () => {
        timeMultiplier = parseFloat(slider.value);
        label.textContent = timeMultiplier === 0 ? 'Paused' : timeMultiplier.toFixed(2) + '×';
    });

    const countSlider = document.getElementById('count-slider');
    const countLabel  = document.getElementById('count-label');
    countSlider.addEventListener('input', () => {
        numParticles = parseInt(countSlider.value, 10);
        countLabel.textContent = numParticles.toLocaleString();
        initParticles();
        writeUniforms(device, uniformBuf, canvas.width, canvas.height, numTypes, numParticles);
    });

    // ── Render loop ───────────────────────────────────────────────────────────
    function tick(nowMs) {
        if (lastTime === null) lastTime = nowMs;
        const realDt = Math.min((nowMs - lastTime) / 1000, 0.1);
        lastTime = nowMs;

        accumulator += realDt * timeMultiplier;

        const encoder = device.createCommandEncoder();

        let steps = 0;
        while (accumulator >= STEP_DT && steps < MAX_STEPS) {
            const f = physicsStep & 1;
            const cp = encoder.beginComputePass();
            cp.setPipeline(computePipeline);
            cp.setBindGroup(0, computeBG[f]);
            cp.dispatchWorkgroups(Math.ceil(numParticles / 64));
            cp.end();
            accumulator -= STEP_DT;
            physicsStep++;
            steps++;
        }

        const rp = encoder.beginRenderPass({
            colorAttachments: [{
                view:       context.getCurrentTexture().createView(),
                clearValue: { r: 0.04, g: 0.04, b: 0.08, a: 1 },
                loadOp:     'clear',
                storeOp:    'store',
            }],
        });
        rp.setPipeline(renderPipeline);
        rp.setBindGroup(0, renderBG[physicsStep & 1]);
        rp.draw(numParticles * 6);
        rp.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(tick);
    }

    window.addEventListener('resize', () => {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        writeUniforms(device, uniformBuf, canvas.width, canvas.height, numTypes, numParticles);
    });

    requestAnimationFrame(tick);
}

function showError(msg) {
    const el = document.getElementById('error');
    el.style.display = 'block';
    el.textContent   = msg;
}

main();
