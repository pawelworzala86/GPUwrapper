// workgroup: [d_model]
@group(0) @binding(0) var<storage, read> dLogits : array<f32>;
@group(0) @binding(1) var<storage, read> Wout : array<f32>; // [vocab * d_model]
@group(0) @binding(2) var<storage, read_write> dHlast : array<f32>;
@group(0) @binding(3) var<storage, read_write> W_in_last : array<f32>; // embedding last token
@group(0) @binding(4) var<uniform> dims : vec2<u32>; // dims.x = vocab, dims.y = d_model
@group(0) @binding(5) var<uniform> lr : f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let d = gid.x;
    let vocab = dims.x;
    let d_model = dims.y;

    if (d >= d_model) { return; }

    var sum : f32 = 0.0;

    // dH_last[d] = Σ_v dLogits[v] * Wout[v,d]
    for (var v = 0u; v < vocab; v++) {
        let idx = v * d_model + d;
        sum += dLogits[v] * Wout[idx];
    }

    dHlast[d] = sum;

    // update embeddingu ostatniego tokena
    W_in_last[d] = W_in_last[d] - lr * sum;
}
