import { useEffect, useLayoutEffect, useRef, useState, useCallback } from 'react';
import computeSrc from './compute.wgsl?raw';
import renderSrc  from './render.wgsl?raw';
import './App.css';

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_PARTICLES     = 10000;
const DEFAULT_PARTICLES = 5000;
const RMIN              = 25;
const RMAX              = 320;
const REP_STRENGTH      = 3.0;
const ATTRACT_STRENGTH  = 0.005;
const FRICTION          = 0.9;
const MAX_TYPES         = 8;
const PARTICLE_RADIUS   = 3.5;
const STEP_DT           = 1 / 60;
const MAX_STEPS         = 8;
const FLOATS_PER_PARTICLE = 6;

const TYPE_COLORS = [
  '#ff3838', '#33ff4d', '#338cff', '#ffe619',
  '#ff6600', '#cc33ff', '#00e5e5', '#ff99cc',
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function worldScale() {
  return Math.max(2, 3840 / Math.max(window.innerWidth, window.innerHeight));
}

function valueToColor(v) {
  const t    = Math.abs(v);
  const base = [14, 14, 18];
  const pos  = [28, 200, 70];
  const neg  = [215, 32,  32];
  const c    = v >= 0 ? pos : neg;
  return `rgb(${Math.round(base[0]+t*(c[0]-base[0]))},${Math.round(base[1]+t*(c[1]-base[1]))},${Math.round(base[2]+t*(c[2]-base[2]))})`;
}

function writeUniforms(device, buf, width, height, numTypes, numParticles) {
  const ab = new ArrayBuffer(48);
  const dv = new DataView(ab);
  dv.setFloat32( 0, width  * worldScale(), true);
  dv.setFloat32( 4, height * worldScale(), true);
  dv.setUint32 ( 8, numParticles,          true);
  dv.setUint32 (12, numTypes,              true);
  dv.setFloat32(16, RMIN,                  true);
  dv.setFloat32(20, RMAX,                  true);
  dv.setFloat32(24, REP_STRENGTH,          true);
  dv.setFloat32(28, ATTRACT_STRENGTH,      true);
  dv.setFloat32(32, FRICTION,              true);
  dv.setFloat32(36, PARTICLE_RADIUS * worldScale(), true);
  device.queue.writeBuffer(buf, 0, ab);
}

// ── MatrixCell ────────────────────────────────────────────────────────────────
function MatrixCell({ value, size, onChange }) {
  const dragRef = useRef({ active: false, startY: 0, startVal: 0, moved: false });
  const elRef   = useRef(null);

  const onPointerDown = (e) => {
    dragRef.current = { active: true, startY: e.clientY, startVal: value, moved: false };
    elRef.current.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onPointerMove = (e) => {
    const d = dragRef.current;
    if (!d.active) return;
    if (Math.abs(e.clientY - d.startY) > 3) d.moved = true;
    if (!d.moved) return;
    onChange(Math.max(-1, Math.min(1, d.startVal + (d.startY - e.clientY) * 0.01)));
  };

  const onPointerUp = () => {
    const d = dragRef.current;
    if (d.active && !d.moved) onChange(0);
    d.active = false;
  };

  return (
    <div
      ref={elRef}
      className="matrix-cell"
      style={{ width: size, height: size, background: valueToColor(value) }}
      title={(value >= 0 ? '+' : '') + value.toFixed(2)}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => { dragRef.current.active = false; }}
    />
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const canvasRef     = useRef(null);
  const gpuRef        = useRef(null);
  const animRef       = useRef(null);
  const matrixWrapRef = useRef(null);

  const [error,          setError]          = useState(null);
  const [numTypes,       setNumTypes]       = useState(3);
  const [numParticles,   setNumParticles]   = useState(DEFAULT_PARTICLES);
  const [timeMultiplier, setTimeMultiplier] = useState(1.0);
  const [panelWidth,     setPanelWidth]     = useState(300);
  const [matrixValues,   setMatrixValues]   = useState(() => {
    const a = new Array(MAX_TYPES * MAX_TYPES).fill(0);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        a[i * MAX_TYPES + j] = Math.random() * 2 - 1;
    return a;
  });

  // ── WebGPU init (once on mount) ──────────────────────────────────────────
  useEffect(() => {
    if (!navigator.gpu) {
      setError('WebGPU not supported. Use Chrome 113+ or Edge 113+.');
      return;
    }

    async function init() {
      const canvas = canvasRef.current;
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { setError('No WebGPU adapter found.'); return; }
      const device  = await adapter.requestDevice();
      const ctx     = canvas.getContext('webgpu');
      const format  = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format, alphaMode: 'opaque' });

      // Seed forceMatrix from React state captured at mount time
      const forceMatrix = new Float32Array(MAX_TYPES * MAX_TYPES);
      for (let i = 0; i < MAX_TYPES * MAX_TYPES; i++) forceMatrix[i] = matrixValues[i];

      const bufSize = MAX_PARTICLES * FLOATS_PER_PARTICLE * 4;
      const bufA    = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const bufB    = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE });
      const uBuf    = device.createBuffer({ size: 48,      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const fmBuf   = device.createBuffer({ size: forceMatrix.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

      const cPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: computeSrc }), entryPoint: 'main' },
      });
      const rModule   = device.createShaderModule({ code: renderSrc });
      const rPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex:   { module: rModule, entryPoint: 'vs_main' },
        fragment: {
          module: rModule, entryPoint: 'fs_main',
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

      const cLayout = cPipeline.getBindGroupLayout(0);
      const rLayout = rPipeline.getBindGroupLayout(0);
      const mkCBG = (src, dst) => device.createBindGroup({ layout: cLayout, entries: [
        { binding: 0, resource: { buffer: src   } },
        { binding: 1, resource: { buffer: dst   } },
        { binding: 2, resource: { buffer: uBuf  } },
        { binding: 3, resource: { buffer: fmBuf } },
      ]});
      const mkRBG = buf => device.createBindGroup({ layout: rLayout, entries: [
        { binding: 0, resource: { buffer: buf  } },
        { binding: 1, resource: { buffer: uBuf } },
      ]});

      gpuRef.current = {
        device, ctx, bufA, bufB, uBuf, fmBuf,
        cPipeline, rPipeline,
        cBGs: [mkCBG(bufA, bufB), mkCBG(bufB, bufA)],
        rBGs: [mkRBG(bufA), mkRBG(bufB)],
        forceMatrix,
        physicsStep: 0, accumulator: 0, lastTime: null,
        timeMultiplier: 1.0,
        numParticles:   DEFAULT_PARTICLES,
        numTypes:       3,
      };

      spawnParticles(DEFAULT_PARTICLES, 3);
      writeUniforms(device, uBuf, canvas.width, canvas.height, 3, DEFAULT_PARTICLES);
      device.queue.writeBuffer(fmBuf, 0, forceMatrix);

      // Tick reads exclusively from gpuRef — no stale closure risk
      function tick(nowMs) {
        const g = gpuRef.current;
        if (!g) return;

        if (g.lastTime === null) g.lastTime = nowMs;
        const dt = Math.min((nowMs - g.lastTime) / 1000, 0.1);
        g.lastTime     = nowMs;
        g.accumulator += dt * g.timeMultiplier;

        const enc = device.createCommandEncoder();
        let steps = 0;
        const n   = g.numParticles;
        while (g.accumulator >= STEP_DT && steps < MAX_STEPS) {
          const f  = g.physicsStep & 1;
          const cp = enc.beginComputePass();
          cp.setPipeline(g.cPipeline);
          cp.setBindGroup(0, g.cBGs[f]);
          cp.dispatchWorkgroups(Math.ceil(n / 64));
          cp.end();
          g.accumulator -= STEP_DT;
          g.physicsStep++;
          steps++;
        }

        const rp = enc.beginRenderPass({
          colorAttachments: [{
            view:       ctx.getCurrentTexture().createView(),
            clearValue: { r: 0.04, g: 0.04, b: 0.08, a: 1 },
            loadOp:     'clear',
            storeOp:    'store',
          }],
        });
        rp.setPipeline(g.rPipeline);
        rp.setBindGroup(0, g.rBGs[g.physicsStep & 1]);
        rp.draw(n * 6);
        rp.end();

        device.queue.submit([enc.finish()]);
        animRef.current = requestAnimationFrame(tick);
      }

      animRef.current = requestAnimationFrame(tick);
    }

    init(); // eslint-disable-line react-hooks/exhaustive-deps
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function spawnParticles(n, nt) {
    const g = gpuRef.current;
    if (!g) return;
    const canvas = canvasRef.current;
    const data   = new Float32Array(n * FLOATS_PER_PARTICLE);
    for (let i = 0; i < n; i++) {
      const b = i * FLOATS_PER_PARTICLE;
      data[b]   = Math.random() * canvas.width  * worldScale();
      data[b+1] = Math.random() * canvas.height * worldScale();
      data[b+4] = Math.floor(Math.random() * nt);
    }
    g.device.queue.writeBuffer(g.bufA, 0, data);
    g.physicsStep = 0;
    g.accumulator = 0;
  }

  // ── Sync forceMatrix → GPU ────────────────────────────────────────────────
  useEffect(() => {
    const g = gpuRef.current;
    if (!g) return;
    for (let i = 0; i < MAX_TYPES * MAX_TYPES; i++) g.forceMatrix[i] = matrixValues[i];
    g.device.queue.writeBuffer(g.fmBuf, 0, g.forceMatrix);
  }, [matrixValues]);

  // ── Respawn + sync uniforms on count / type change ────────────────────────
  useEffect(() => {
    const g = gpuRef.current;
    if (!g) return;
    g.numTypes     = numTypes;
    g.numParticles = numParticles;
    const canvas   = canvasRef.current;
    spawnParticles(numParticles, numTypes);
    writeUniforms(g.device, g.uBuf, canvas.width, canvas.height, numTypes, numParticles);
  }, [numTypes, numParticles]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Resize: registered once, always reads fresh values from gpuRef ────────
  useEffect(() => {
    function onResize() {
      const g = gpuRef.current;
      if (!g) return;
      const canvas  = canvasRef.current;
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      writeUniforms(g.device, g.uBuf, canvas.width, canvas.height, g.numTypes, g.numParticles);
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // ── Panel width measurement ───────────────────────────────────────────────
  useLayoutEffect(() => {
    if (!matrixWrapRef.current) return;
    const ro = new ResizeObserver(([e]) => setPanelWidth(e.contentRect.width));
    ro.observe(matrixWrapRef.current);
    return () => ro.disconnect();
  }, []);

  // ── UI handlers ───────────────────────────────────────────────────────────
  const onSpeedChange = (e) => {
    const v = parseFloat(e.target.value);
    setTimeMultiplier(v);
    // timeMultiplier is used every tick frame so update gpuRef directly
    if (gpuRef.current) gpuRef.current.timeMultiplier = v;
  };

  const onNumTypesChange = (delta) => {
    const next = Math.max(1, Math.min(MAX_TYPES, numTypes + delta));
    if (next === numTypes) return;
    setMatrixValues(prev => {
      const a = [...prev];
      if (next > numTypes) {
        for (let i = 0; i < next; i++)
          for (let j = 0; j < next; j++)
            if (i >= numTypes || j >= numTypes)
              a[i * MAX_TYPES + j] = Math.random() * 2 - 1;
      }
      return a;
    });
    setNumTypes(next);
  };

  const onCellChange = useCallback((idx, v) => {
    setMatrixValues(prev => {
      const a = [...prev];
      a[idx]  = v;
      return a;
    });
  }, []);

  const onZero = () => setMatrixValues(new Array(MAX_TYPES * MAX_TYPES).fill(0));

  const onRandomize = () => {
    setMatrixValues(prev => {
      const a = [...prev];
      for (let i = 0; i < numTypes; i++)
        for (let j = 0; j < numTypes; j++)
          a[i * MAX_TYPES + j] = Math.random() * 2 - 1;
      return a;
    });
  };

  // ── Reactive matrix cell sizing ───────────────────────────────────────────
  const GAP    = 4;
  const cellPx = Math.max(36, Math.min(64, Math.floor((panelWidth - 20 - GAP * numTypes) / numTypes)));
  const dotPx  = Math.round(cellPx * 0.55);

  if (error) return (
    <div style={{ color: '#f66', fontFamily: 'monospace', padding: 24 }}>{error}</div>
  );

  return (
    <>
      <canvas ref={canvasRef} style={{ display: 'block', position: 'fixed', inset: 0 }} />

      <div id="control-panel">

        {/* Non-matrix controls — left of matrix on portrait, above on landscape/desktop */}
        <div className="side-controls">

          <div className="slider-row">
            <span>Speed</span>
            <input type="range" min="0" max="10" step="0.1"
              value={timeMultiplier} onChange={onSpeedChange} />
            <span className="slider-val">
              {timeMultiplier === 0 ? 'Paused' : timeMultiplier.toFixed(2) + '×'}
            </span>
          </div>

          <div className="slider-row">
            <span>Particles</span>
            <input type="range" min="1000" max="10000" step="1000"
              value={numParticles} onChange={e => setNumParticles(parseInt(e.target.value, 10))} />
            <span className="slider-val">{numParticles.toLocaleString()}</span>
          </div>

          <hr className="divider" />

          <div className="panel-header">
            <span>Types</span>
            <div className="type-counter">
              <button className="icon-btn" onClick={() => onNumTypesChange(-1)} disabled={numTypes <= 1}>−</button>
              <span>{numTypes}</span>
              <button className="icon-btn" onClick={() => onNumTypesChange(+1)} disabled={numTypes >= MAX_TYPES}>+</button>
            </div>
          </div>

          <div className="header-actions">
            <button className="text-btn" onClick={onZero}>Zero</button>
            <button className="text-btn" onClick={onRandomize}>Randomize</button>
          </div>

        </div>

        {/* NxN attraction matrix */}
        <div className="matrix-wrap" ref={matrixWrapRef}>
          <div style={{
            display: 'grid',
            gap: GAP,
            gridTemplateColumns: `${dotPx}px ${Array(numTypes).fill(`${cellPx}px`).join(' ')}`,
          }}>
            {Array.from({ length: (numTypes + 1) * (numTypes + 1) }, (_, k) => {
              const row = Math.floor(k / (numTypes + 1));
              const col = k % (numTypes + 1);

              if (row === 0 && col === 0) return <div key={k} style={{ width: dotPx, height: dotPx }} />;

              if (row === 0) return (
                <div key={k} className="type-dot"
                  style={{ width: dotPx, height: dotPx, background: TYPE_COLORS[col - 1] }}
                  title={`target type ${col - 1}`} />
              );

              if (col === 0) return (
                <div key={k} className="type-dot"
                  style={{ width: dotPx, height: dotPx, background: TYPE_COLORS[row - 1] }}
                  title={`source type ${row - 1}`} />
              );

              const idx = (row - 1) * MAX_TYPES + (col - 1);
              return (
                <MatrixCell key={`${row}-${col}`}
                  value={matrixValues[idx]}
                  size={cellPx}
                  onChange={v => onCellChange(idx, v)}
                />
              );
            })}
          </div>
        </div>

      </div>
    </>
  );
}
