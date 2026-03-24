(function(){const u=document.createElement("link").relList;if(u&&u.supports&&u.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))l(o);new MutationObserver(o=>{for(const e of o)if(e.type==="childList")for(const i of e.addedNodes)i.tagName==="LINK"&&i.rel==="modulepreload"&&l(i)}).observe(document,{childList:!0,subtree:!0});function t(o){const e={};return o.integrity&&(e.integrity=o.integrity),o.referrerPolicy&&(e.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?e.credentials="include":o.crossOrigin==="anonymous"?e.credentials="omit":e.credentials="same-origin",e}function l(o){if(o.ep)return;o.ep=!0;const e=t(o);fetch(o.href,e)}})();const $=`// Maximum number of particle types (buffer stride — never changes at runtime).
const MAX_TYPES: u32 = 8u;

struct Particle {
    pos:   vec2<f32>,
    vel:   vec2<f32>,
    ptype: f32,
    _pad:  f32,
}

// Must match writeUniforms() layout in main.js exactly (48 bytes, 16-byte aligned).
struct Uniforms {
    width:           f32,  //  0
    height:          f32,  //  4
    numParticles:    u32,  //  8
    numTypes:        u32,  // 12 — active type count (1–8); shader wraps particle types to this range
    rMin:            f32,  // 16
    rMax:            f32,  // 20
    repStrength:     f32,  // 24
    attractStrength: f32,  // 28
    friction:        f32,  // 32
    _pad0:           f32,  // 36
    _pad1:           f32,  // 40
    _pad2:           f32,  // 44
}

@group(0) @binding(0) var<storage, read>       particlesIn:  array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform>             uniforms:     Uniforms;
@group(0) @binding(3) var<storage, read>       forceMatrix:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.numParticles) { return; }

    let p = particlesIn[i];
    var acc = vec2<f32>(0.0, 0.0);

    // Wrap stored type into active range so old particles stay valid after
    // the user reduces numTypes.
    let typeI = u32(p.ptype) % uniforms.numTypes;

    for (var j = 0u; j < uniforms.numParticles; j++) {
        if (j == i) { continue; }
        let q = particlesIn[j];

        // Shortest-path delta across periodic boundaries
        var diff = p.pos - q.pos;
        diff.x -= uniforms.width  * round(diff.x / uniforms.width);
        diff.y -= uniforms.height * round(diff.y / uniforms.height);

        let dist = length(diff);
        if (dist < 0.001 || dist > uniforms.rMax) { continue; }

        let dir   = diff / dist;
        let typeJ = u32(q.ptype) % uniforms.numTypes;

        if (dist < uniforms.rMin) {
            // Smooth universal repulsion: 0 at rMin, increases quadratically closer.
            let t = 1.0 - dist / uniforms.rMin;
            acc += dir * uniforms.repStrength * t * t;
        } else {
            // Type-based force with triangular envelope (0 at rMin, 0 at rMax).
            let t      = (dist - uniforms.rMin) / (uniforms.rMax - uniforms.rMin);
            let shape  = 1.0 - abs(2.0 * t - 1.0);
            // forceMatrix uses a fixed stride of MAX_TYPES regardless of numTypes,
            // so the buffer layout never needs to change at runtime.
            let attrac = forceMatrix[typeI * MAX_TYPES + typeJ];
            acc -= dir * uniforms.attractStrength * attrac * shape;
        }
    }

    let newVel = p.vel * uniforms.friction + acc;
    var newPos = p.pos + newVel;

    newPos.x = ((newPos.x % uniforms.width)  + uniforms.width)  % uniforms.width;
    newPos.y = ((newPos.y % uniforms.height) + uniforms.height) % uniforms.height;

    particlesOut[i] = Particle(newPos, newVel, p.ptype, 0.0);
}
`,J=`struct Particle {
    pos:   vec2<f32>,
    vel:   vec2<f32>,
    ptype: f32,
    _pad:  f32,
}

// Same layout as compute.wgsl — only width/height are used here.
struct Uniforms {
    width:           f32,
    height:          f32,
    numParticles:    u32,
    numTypes:        u32,
    rMin:            f32,
    rMax:            f32,
    repStrength:     f32,
    attractStrength: f32,
    friction:        f32,
    particleRadius:  f32,  // 36 — base radius × WORLD_SCALE, set by writeUniforms()
    _pad1:           f32,
    _pad2:           f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform>       uniforms:  Uniforms;

struct VertexOut {
    @builtin(position) pos:   vec4<f32>,
    @location(0)       color: vec3<f32>,
    @location(1)       uv:    vec2<f32>,
}

// 8 distinct colors — must match TYPE_COLORS in main.js.
fn typeColor(t: u32) -> vec3<f32> {
    switch (t % 8u) {
        case 0u: { return vec3<f32>(1.00, 0.22, 0.22); } // red
        case 1u: { return vec3<f32>(0.20, 1.00, 0.30); } // green
        case 2u: { return vec3<f32>(0.20, 0.55, 1.00); } // blue
        case 3u: { return vec3<f32>(1.00, 0.90, 0.10); } // yellow
        case 4u: { return vec3<f32>(1.00, 0.40, 0.00); } // orange
        case 5u: { return vec3<f32>(0.80, 0.20, 1.00); } // purple
        case 6u: { return vec3<f32>(0.00, 0.90, 0.90); } // cyan
        default: { return vec3<f32>(1.00, 0.60, 0.80); } // pink
    }
}

fn quadCorner(idx: u32) -> vec2<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    return corners[idx];
}

@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32) -> VertexOut {
    let pIdx   = vIdx / 6u;
    let cIdx   = vIdx % 6u;
    let p      = particles[pIdx];
    let corner = quadCorner(cIdx);

    let pixelPos = p.pos + corner * uniforms.particleRadius;
    let ndc = vec2<f32>(
         pixelPos.x / uniforms.width  * 2.0 - 1.0,
        -pixelPos.y / uniforms.height * 2.0 + 1.0,
    );

    var out: VertexOut;
    out.pos   = vec4<f32>(ndc, 0.0, 1.0);
    out.color = typeColor(u32(p.ptype));
    out.uv    = corner;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    let alpha = 1.0 - smoothstep(0.4, 1.0, r);
    return vec4<f32>(in.color * alpha, alpha);
}
`,K=1e4,Q=5e3,Z=20,ee=300,te=3,ne=.01,re=.9,j=["#ff3838","#33ff4d","#338cff","#ffe619","#ff6600","#cc33ff","#00e5e5","#ff99cc"],m=8,ie=3.5;function x(){return Math.max(2,3840/Math.max(window.innerWidth,window.innerHeight))}const D=1/60,oe=8,O=6;function C(r,u,t,l,o,e){const i=new ArrayBuffer(48),n=new DataView(i);n.setFloat32(0,t*x(),!0),n.setFloat32(4,l*x(),!0),n.setUint32(8,e,!0),n.setUint32(12,o,!0),n.setFloat32(16,Z,!0),n.setFloat32(20,ee,!0),n.setFloat32(24,te,!0),n.setFloat32(28,ne,!0),n.setFloat32(32,re,!0),n.setFloat32(36,ie*x(),!0),r.queue.writeBuffer(u,0,i)}function ae(r){const u=Math.abs(r),t=[14,14,18],l=[28,200,70],o=[215,32,32],e=r>=0?l:o;return`rgb(${Math.round(t[0]+u*(e[0]-t[0]))},${Math.round(t[1]+u*(e[1]-t[1]))},${Math.round(t[2]+u*(e[2]-t[2]))})`}function V({numTypes:r,typeColors:u,forceMatrix:t,onAnyChange:l}){const o=document.getElementById("matrix-grid");o.innerHTML="",o.style.gridTemplateColumns=`18px ${"30px ".repeat(r)}`.trim();const e=new Array(m*m);for(let i=0;i<=r;i++)for(let n=0;n<=r;n++){const s=document.createElement("div");if(i===0&&n===0)s.style.cssText="width:18px;height:18px";else if(i===0)s.className="type-dot",s.style.background=u[n-1],s.title=`target type ${n-1}`;else if(n===0)s.className="type-dot",s.style.background=u[i-1],s.title=`source type ${i-1}`;else{const w=i-1,P=n-1,g=w*m+P;s.className="matrix-cell";const d=f=>{t[g]=f,s.style.background=ae(f),s.title=(f>=0?"+":"")+f.toFixed(2)};e[g]=d,d(t[g]);let y=!1,h=!1,b=0,E=0;s.addEventListener("pointerdown",f=>{y=!0,h=!1,b=f.clientY,E=t[g],s.setPointerCapture(f.pointerId),f.preventDefault()}),s.addEventListener("pointermove",f=>{if(!y||(Math.abs(f.clientY-b)>3&&(h=!0),!h))return;const M=Math.max(-1,Math.min(1,E+(b-f.clientY)*.01));d(M),l()}),s.addEventListener("pointerup",()=>{y&&!h&&(d(0),l()),y=!1,h=!1}),s.addEventListener("pointercancel",()=>{y=!1,h=!1})}o.appendChild(s)}return{refresh(){var i;for(let n=0;n<r;n++)for(let s=0;s<r;s++)(i=e[n*m+s])==null||i.call(e,t[n*m+s])}}}async function se(){if(!navigator.gpu){N("WebGPU not supported. Use Chrome 113+ or Edge 113+.");return}const r=document.getElementById("canvas");r.width=window.innerWidth,r.height=window.innerHeight;const u=await navigator.gpu.requestAdapter();if(!u){N("No WebGPU adapter found.");return}const t=await u.requestDevice(),l=r.getContext("webgpu"),o=navigator.gpu.getPreferredCanvasFormat();l.configure({device:t,format:o,alphaMode:"opaque"});let e=2,i=Q;const n=new Float32Array(m*m);for(let a=0;a<e;a++)for(let c=0;c<e;c++)n[a*m+c]=Math.random()*2-1;const s=K*O*4,w=t.createBuffer({size:s,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),P=t.createBuffer({size:s,usage:GPUBufferUsage.STORAGE}),g=t.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),d=t.createBuffer({size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),y=t.createComputePipeline({layout:"auto",compute:{module:t.createShaderModule({code:$}),entryPoint:"main"}}),h=t.createShaderModule({code:J}),b=t.createRenderPipeline({layout:"auto",vertex:{module:h,entryPoint:"vs_main"},fragment:{module:h,entryPoint:"fs_main",targets:[{format:o,blend:{color:{srcFactor:"src-alpha",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),E=y.getBindGroupLayout(0),f=b.getBindGroupLayout(0),M=(a,c)=>t.createBindGroup({layout:E,entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:g}},{binding:3,resource:{buffer:d}}]}),R=a=>t.createBindGroup({layout:f,entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:g}}]}),k=[M(w,P),M(P,w)],z=[R(w),R(P)];let B=0,S=0;function A(){const a=new Float32Array(i*O);for(let c=0;c<i;c++){const p=c*O;a[p+0]=Math.random()*r.width*x(),a[p+1]=Math.random()*r.height*x(),a[p+4]=Math.floor(Math.random()*e)}t.queue.writeBuffer(w,0,a),B=0,S=0}let U=V({numTypes:e,typeColors:j,forceMatrix:n,onAnyChange:()=>t.queue.writeBuffer(d,0,n)});function F(a){if(a>e)for(let p=0;p<a;p++)for(let v=0;v<a;v++)(p>=e||v>=e)&&(n[p*m+v]=Math.random()*2-1);e=a,t.queue.writeBuffer(d,0,n),C(t,g,r.width,r.height,e,i),A(),U=V({numTypes:e,typeColors:j,forceMatrix:n,onAnyChange:()=>t.queue.writeBuffer(d,0,n)});const c=document.getElementById("types-count");c&&(c.textContent=e),document.getElementById("types-minus").disabled=e<=1,document.getElementById("types-plus").disabled=e>=m}document.getElementById("types-minus").addEventListener("click",()=>{e>1&&F(e-1)}),document.getElementById("types-plus").addEventListener("click",()=>{e<m&&F(e+1)}),document.getElementById("zero-btn").addEventListener("click",()=>{n.fill(0),t.queue.writeBuffer(d,0,n),U.refresh()}),document.getElementById("randomize-btn").addEventListener("click",()=>{for(let a=0;a<e;a++)for(let c=0;c<e;c++)n[a*m+c]=Math.random()*2-1;t.queue.writeBuffer(d,0,n),U.refresh()}),A(),C(t,g,r.width,r.height,e,i),t.queue.writeBuffer(d,0,n);let I=1,T=null;const G=document.getElementById("speed-slider"),W=document.getElementById("speed-label");G.addEventListener("input",()=>{I=parseFloat(G.value),W.textContent=I===0?"Paused":I.toFixed(2)+"×"});const q=document.getElementById("count-slider"),X=document.getElementById("count-label");q.addEventListener("input",()=>{i=parseInt(q.value,10),X.textContent=i.toLocaleString(),A(),C(t,g,r.width,r.height,e,i)});function Y(a){T===null&&(T=a);const c=Math.min((a-T)/1e3,.1);T=a,S+=c*I;const p=t.createCommandEncoder();let v=0;for(;S>=D&&v<oe;){const H=B&1,L=p.beginComputePass();L.setPipeline(y),L.setBindGroup(0,k[H]),L.dispatchWorkgroups(Math.ceil(i/64)),L.end(),S-=D,B++,v++}const _=p.beginRenderPass({colorAttachments:[{view:l.getCurrentTexture().createView(),clearValue:{r:.04,g:.04,b:.08,a:1},loadOp:"clear",storeOp:"store"}]});_.setPipeline(b),_.setBindGroup(0,z[B&1]),_.draw(i*6),_.end(),t.queue.submit([p.finish()]),requestAnimationFrame(Y)}window.addEventListener("resize",()=>{r.width=window.innerWidth,r.height=window.innerHeight,C(t,g,r.width,r.height,e,i)}),requestAnimationFrame(Y)}function N(r){const u=document.getElementById("error");u.style.display="block",u.textContent=r}se();
