struct Config {
    width: u32,
    height: u32,
    depth: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> config: Config;



fn hash31(p: vec3<f32>) -> f32 {
    let dotp = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(dotp) * 43758.5453123);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Smoothstep (Hermite)
    let u = f * f * (3.0 - 2.0 * f);

    // 8 narożników kostki
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));

    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear interpolation
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}

fn random(x:u32,y:u32,z:u32) -> f32 {
    let xf = (f32(x)+0.01)*1.4;
    let yf = f32(y)/1.003;
    let zf = f32(z)*1.002;
    return noise3(vec3<f32>(xf, yf, zf)) * 2.0 - 1.0;
}

// Ustawiamy rozmiar grupy roboczej (np. 8x8x4, co daje 256 wątków)
@compute @workgroup_size(8, 8, 4) 
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x < config.width && y < config.height && z < config.depth) {
        let index = z * (config.width * config.height)
          + y * config.width
          + x;

        data[index] = random(x,y,z);
    }
}