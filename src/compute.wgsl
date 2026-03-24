// Maximum number of particle types (buffer stride — never changes at runtime).
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
