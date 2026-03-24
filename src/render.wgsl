struct Particle {
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
