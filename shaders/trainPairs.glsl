struct Config {
    width: u32,
    height: u32,
    depth: u32,
};

/*struct Pairs {
    pairA: u32,
    pairB: u32,
};*/

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<uniform> pairs: array<i32,1024>;

fn loadMat16(base: u32) -> array<f32, 16> {
    return array<f32, 16>(
        data[base+0], data[base+1], data[base+2], data[base+3],
        data[base+4], data[base+5], data[base+6], data[base+7],
        data[base+8], data[base+9], data[base+10], data[base+11],
        data[base+12], data[base+13], data[base+14], data[base+15]
    );
}

fn storeMat16(base: u32, m: array<f32, 16>) {
    for (var r = 0u; r < 4u; r++) {
        for (var c = 0u; c < 4u; c++) {
            data[base + r*4u + c] = m[r*4u + c];
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    for (var p = 0u; p < 50u; p = p + 2u) {
        let A = pairs[p];//.pairA;
        let B = pairs[p+1u];//.pairB;
        if(A == -1i || B == -1i){
            continue;
        }

        // Każda mat4 to 16 elementów
        let stride = 16u;

        let baseA = u32(A) * stride;
        let baseB = u32(B) * stride;

        let mA = loadMat16(baseA);
        let mB = loadMat16(baseB);

        var resultA = mA;

        for (var i = 0u; i < 16u; i = i + 1u) {
            resultA[i] += ( mB[i] - mA[i] ) * 0.01;
        }
        // Przykładowa operacja: mA = mA * mB
        //let resultA = mA;

        storeMat16(baseA, resultA);
    }
}