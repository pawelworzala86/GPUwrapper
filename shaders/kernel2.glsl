// workgroup: [vocab, d_model]
@group(0) @binding(0) var<storage, read> dLogits : array<f32>;
@group(0) @binding(1) var<storage, read> hLast : array<f32>;
@group(0) @binding(2) var<storage, read_write> Wout : array<f32>; // flattened [vocab * d_model]
@group(0) @binding(3) var<uniform> dims : vec2<u32>; // dims.x = vocab, dims.y = d_model
@group(0) @binding(4) var<uniform> lr : f32;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    let d = gid.y;

    let vocab = dims.x;
    let d_model = dims.y;

    if (v >= vocab || d >= d_model) { return; }

    let idx = v * d_model + d;

    let g = dLogits[v];
    let h = hLast[d];

    // SGD update
    Wout[idx] = Wout[idx] - lr * g * h;
}
