struct Config {
    width: u32,
    height: u32,
    depth: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<uniform> token: array<f32, 128>;

@compute @workgroup_size(8, 8, 4) // 8*8*4 = 256
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= config.width || y >= config.height || z >= config.depth) {
        return;
    }

    let index = z * (config.width * config.height)
              + y * config.width
              + x;

    data[index] = data[index] + f32(index) + token[0]; // albo random(x,y,z)
}