struct MatrixDims {
    M : u32, // rows A, C
    N : u32, // cols B, C
    K : u32, // cols A, rows B
};

@group(0) @binding(0)
var<storage, read_write> A : array<f32>;

@group(0) @binding(1)
var<storage, read_write> B : array<f32>;

@group(0) @binding(2)
var<storage, read_write> C : array<f32>;

@group(0) @binding(3)
var<uniform> dims : MatrixDims;

fn getA(m : u32, k : u32) -> f32 {
    return A[m * dims.K + k];
}

fn getB(k : u32, n : u32) -> f32 {
    return B[k * dims.N + n];
}

fn setC(m : u32, n : u32, v : f32) {
    C[m * dims.N + n] = v;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let m = gid.y;
    let n = gid.x;

    if (m >= dims.M || n >= dims.N) {
        return;
    }

    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < dims.K; k = k + 1u) {
        acc = acc + getA(m, k) * getB(k, n);
    }

    setC(m, n, acc);
}
