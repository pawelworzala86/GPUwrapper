// workgroup: [vocab]
@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> probs : array<f32>;
@group(0) @binding(2) var<storage, read_write> dLogits : array<f32>;
@group(0) @binding(3) var<uniform> targetU : u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&logits)) { return; }

    // 1) softmax (stabilized)
    // najpierw max
    var maxVal : f32 = -1e30;
    for (var k = 0u; k < arrayLength(&logits); k++) {
        maxVal = max(maxVal, logits[k]);
    }

    // exp + sum
    var sumVal : f32 = 0.0;
    for (var k = 0u; k < arrayLength(&logits); k++) {
        let e = exp(logits[k] - maxVal);
        probs[k] = e;
        sumVal += e;
    }

    // normalizacja
    probs[i] = probs[i] / sumVal;

    // 2) dLogits = probs - one_hot
    let t = targetU;
    dLogits[i] = probs[i] - (select(0.0, 1.0, i == t));
}
